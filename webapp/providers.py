"""Per-account provider configuration: bring your own endpoint, model and key.

The deployment-wide settings in app_settings stay the administrator's surface.
This table layers one row per account in front of them, so a signed-in user can
point their own turns at their own backend — a local Ollama, an OpenAI-compatible
gateway, a paid key — without reconfiguring what everyone else gets. No row means
"use the deployment default", which is why clearing it is a delete.

The api key is encrypted with SYSML_SECRET_KEY (sha256 -> Fernet key) because a
database dump must not hand out working credentials. With no secret set the key
field is refused with a readable 400 rather than written in plaintext; everything
else in the row still works, since a local Ollama needs no key at all.

A user-supplied base url is fetched by this server on that account's turns, so it
is validated before it is ever stored: http(s) only, no embedded credentials, and
no address that resolves into the container's own network unless the operator
opts in with SYSML_ALLOW_PRIVATE_ENDPOINTS=1 (see docs/DEPLOY.md).
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from webapp import db

logger = logging.getLogger("webapp.providers")

# Same shape as the settings cache: a couple of seconds so one turn does not read
# Postgres twice, bounded so per-account rows cannot grow it without limit.
TTL = float(os.environ.get("SYSML_PROVIDER_TTL") or 2.0)
MAX_ENTRIES = 256

# Anything else (loopback, RFC1918, link-local, CGNAT, multicast, reserved, ::1)
# is refused by default: the container can reach its own services and the cloud
# metadata endpoint at 169.254.169.254, and a saved base url is fetched with the
# app's full network position.
_PRIVATE_OK = ("1", "true", "yes", "on")

_COLUMNS = "provider, model, base_url, api_key_cipher, temperature, effort"
_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
_fernet_instance: Any = None
_fernet_lock = threading.Lock()


# --- Secrets ----------------------------------------------------------------


def secret() -> str:
    return (os.environ.get("SYSML_SECRET_KEY") or "").strip()


def enabled() -> bool:
    """True when the deployment has a secret, i.e. provider keys can be stored."""
    return bool(secret())


def _fernet() -> Any:
    """The Fernet built from SYSML_SECRET_KEY, derived once per process."""
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance
    with _fernet_lock:
        if _fernet_instance is None:
            from cryptography.fernet import Fernet

            digest = hashlib.sha256(secret().encode("utf-8")).digest()
            _fernet_instance = Fernet(base64.urlsafe_b64encode(digest))
    return _fernet_instance


def encrypt(key: Any) -> str:
    """Encrypt an api key for storage; "" stays "" and no secret raises."""
    text = str(key or "").strip()
    if not text:
        return ""
    if not enabled():
        raise ValueError(
            "the operator must set SYSML_SECRET_KEY to store provider keys; "
            "everything else can be saved without one"
        )
    return _fernet().encrypt(text.encode("utf-8")).decode("utf-8")


def decrypt(token: Any) -> str:
    """Decrypt a stored token; an unreadable one is "" (the row is dropped)."""
    if not token:
        return ""
    if not enabled():
        return ""
    try:
        return _fernet().decrypt(str(token).encode("utf-8")).decode("utf-8")
    except Exception:
        return ""


# --- Base url guard ---------------------------------------------------------


def _addresses(host: str, port: int) -> list:
    try:
        return [
            info[4][0]
            for info in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        ]
    except OSError as exc:
        raise ValueError(f"cannot resolve {host!r}: {exc}") from exc


def guard_base_url(url: Any) -> str:
    """Normalise an OpenAI-compatible base url, or raise ValueError (400).

    This is a write-time check: it stops the obvious SSRF (the metadata endpoint,
    a neighbour service on the container network) but cannot pin DNS, so a host
    that changes its answer between save and use is still reachable. The blast
    radius stays the owning account's own turns with its own key.
    """
    text = str(url or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https"):
        raise ValueError("base url must start with http:// or https://")
    if parts.username or parts.password:
        raise ValueError("base url must not embed credentials")
    host = parts.hostname or ""
    if not host:
        raise ValueError("base url must name a host")
    if (os.environ.get("SYSML_ALLOW_PRIVATE_ENDPOINTS") or "").strip().lower() in _PRIVATE_OK:
        return text.rstrip("/")
    for raw in _addresses(host, parts.port or (443 if parts.scheme == "https" else 80)):
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:  # pragma: no cover - getaddrinfo returned junk
            raise ValueError(f"base url host {host!r} resolved to {raw!r}")
        if (
            address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ValueError(
                f"base url host {host!r} resolves to {address}, which is not a "
                "public endpoint; set SYSML_ALLOW_PRIVATE_ENDPOINTS=1 to allow "
                "a local Ollama or a LAN gateway"
            )
    return text.rstrip("/")


# --- Rows -------------------------------------------------------------------


def load(user_id: Any, force: bool = False) -> Optional[Dict[str, Any]]:
    """This account's row, or None when it has none (deployment default)."""
    key = str(user_id or "")
    if not key:
        return None
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and not force and now - hit[0] < TTL:
            return hit[1]
    try:
        with db.connection() as conn:
            row = conn.execute(
                f"select {_COLUMNS} from user_provider_config where user_id = %s",
                (key,),
            ).fetchone()
    except Exception as exc:
        # A missing table (first boot) or a dead database must not break turns:
        # fall back to the deployment default and let /api/ready report it.
        logger.warning("provider config unavailable for %s (%s)", key, exc)
        return None
    with _cache_lock:
        if len(_cache) >= MAX_ENTRIES:
            _cache.clear()
        _cache[key] = (now, row)
    return row


def _forget(user_id: Any) -> None:
    with _cache_lock:
        _cache.pop(str(user_id or ""), None)


def save(user_id: Any, values: Dict[str, Any]) -> Dict[str, Any]:
    """Insert or replace this account's row; a blank api_key keeps the stored one.

    Raises ValueError for an api key when the deployment has no secret (the route
    turns it into a 400). The base url must already have been through
    guard_base_url.
    """
    key = str(user_id or "")
    cipher = encrypt(values.get("api_key"))
    if not cipher:
        existing = load(key, force=True) or {}
        cipher = str(existing.get("api_key_cipher") or "")
    temperature = values.get("temperature")
    temperature = None if temperature is None else float(temperature)
    with db.connection() as conn:
        conn.execute(
            "insert into user_provider_config "
            "(user_id, provider, model, base_url, api_key_cipher, temperature, effort) "
            "values (%s, %s, %s, %s, %s, %s, %s) "
            "on conflict (user_id) do update set "
            "provider = excluded.provider, model = excluded.model, "
            "base_url = excluded.base_url, api_key_cipher = excluded.api_key_cipher, "
            "temperature = excluded.temperature, effort = excluded.effort, "
            "updated_at = now()",
            (
                key,
                str(values.get("provider") or "").strip().lower(),
                str(values.get("model") or "").strip(),
                str(values.get("base_url") or "").strip(),
                cipher,
                temperature,
                str(values.get("effort") or "").strip(),
            ),
        )
    _forget(key)
    return load(key, force=True) or {}


def clear(user_id: Any) -> None:
    """Back to the deployment default for this account."""
    key = str(user_id or "")
    with db.connection() as conn:
        conn.execute("delete from user_provider_config where user_id = %s", (key,))
    _forget(key)


def _api_key(user_id: str, row: Dict[str, Any]) -> str:
    """Decrypt this row's key; an unreadable one is dropped, not fatal."""
    cipher = str(row.get("api_key_cipher") or "")
    if not cipher:
        return ""
    plain = decrypt(cipher)
    if plain:
        return plain
    logger.warning("dropping unreadable provider key for %s", user_id)
    try:
        with db.connection() as conn:
            conn.execute(
                "update user_provider_config set api_key_cipher = '' where user_id = %s",
                (user_id,),
            )
    except Exception:
        pass
    _forget(user_id)
    return ""


# --- Views ------------------------------------------------------------------


def overrides(user_id: Any) -> Dict[str, str]:
    """The SYSML_* mapping this account's row contributes ({} when it has none)."""
    key = str(user_id or "")
    row = load(key)
    if not row:
        return {}
    env: Dict[str, str] = {"SYSML_PROVIDER": str(row["provider"])}
    if row["model"]:
        env["SYSML_MODEL"] = str(row["model"])
    if row["base_url"]:
        env["SYSML_BASE_URL"] = str(row["base_url"])
    if row["effort"]:
        env["SYSML_EFFORT"] = str(row["effort"])
    if row["temperature"] is not None:
        env["SYSML_TEMPERATURE"] = str(row["temperature"])
    api_key = _api_key(key, row)
    if api_key:
        env["SYSML_API_KEY"] = api_key
    return env


def view(user_id: Any, deployment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The dialog's view of this account's backend; the key is never included.

    With no row the deployment's own values come back, marked source=deployment,
    so the form opens on what the account is actually using.
    """
    base: Dict[str, Any] = {
        "provider": "ollama",
        "model": "",
        "base_url": "",
        "effort": "",
        "temperature": None,
        "api_key_set": False,
    }
    base.update(deployment or {})
    base["source"] = "deployment"
    row = load(user_id)
    if not row:
        return base
    return {
        "provider": str(row["provider"]),
        "model": str(row["model"]),
        "base_url": str(row["base_url"]),
        "effort": str(row["effort"]),
        "temperature": row["temperature"],
        "api_key_set": bool(row["api_key_cipher"]),
        "source": "user",
    }


def signature(env: Dict[str, str]) -> tuple:
    """Hashable identity of the backend a mapping resolves to.

    The workflow cache keys on this: two accounts with the same provider, model,
    endpoint and key share one workflow (and one agent/graph), and any change
    builds a new one. The key itself is hashed so it never sits in a dict key
    that a log line might print.
    """
    api_key = str(env.get("SYSML_API_KEY") or "")
    return (
        str(env.get("SYSML_PROVIDER") or "ollama").strip().lower(),
        str(env.get("SYSML_MODEL") or "").strip(),
        str(env.get("SYSML_BASE_URL") or "").strip(),
        str(env.get("SYSML_EFFORT") or "").strip(),
        str(env.get("SYSML_TEMPERATURE") or "").strip(),
        hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16] if api_key else "",
        (env.get("SYSML_ENABLE_RAG") or "1").strip().lower(),
    )
