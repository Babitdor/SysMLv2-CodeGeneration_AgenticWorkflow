"""Per-key rate limits backed by Redis.

A fixed-window counter is enough for the limits here — chat turns per account,
login attempts per address, uploads per account — and costs one round trip. The
Lua script keeps INCR and PEXPIRE atomic: without it, a request that dies
between the two leaves a counter with no expiry, i.e. a permanent lockout.

Redis is required by the deployment, but the limiter fails *open*: if the script
errors, the request is allowed and the failure is logged. A limiter outage must
not take chat down with it.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

from webapp import db

logger = logging.getLogger(__name__)

# bucket -> (default limit, default window seconds). Override any of them with
# SYSML_RATE_<BUCKET>=<count>/<seconds>, e.g. SYSML_RATE_CHAT=20/300.
DEFAULTS: Dict[str, Tuple[int, int]] = {
    "chat": (20, 5 * 60),
    "auth": (10, 15 * 60),
    "upload": (30, 60 * 60),
    # Canvas writes: editable documents, so more permissive than a chat turn,
    # but still bounded — an editor that saves on every keystroke is not a plan.
    "canvas": (120, 60),
}

# INCR, and only the request that created the counter sets its expiry; PTTL
# comes back so Retry-After is exact rather than the whole window.
_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('PTTL', KEYS[1])
return {count, ttl}
"""

_scripts: Dict[int, Any] = {}


class RateLimited(Exception):
    """Mapped to 429 + Retry-After by the route layer."""

    def __init__(self, bucket: str, limit: int, window: int, retry_after: int) -> None:
        super().__init__(f"rate limit reached for {bucket}")
        self.bucket = bucket
        self.limit = limit
        self.window = window
        self.retry_after = max(1, int(retry_after))


def _configured(bucket: str) -> Tuple[int, int]:
    """(limit, window) for a bucket: env override, else the default."""
    default = DEFAULTS.get(bucket, (60, 60))
    raw = (os.environ.get(f"SYSML_RATE_{bucket.upper()}") or "").strip()
    if not raw:
        return default
    count, _, window = raw.partition("/")
    try:
        limit = int(count)
        seconds = int(window) if window else default[1]
    except ValueError:
        logger.warning("ignoring malformed SYSML_RATE_%s=%r", bucket.upper(), raw)
        return default
    if limit < 1 or seconds < 1:
        return default
    return limit, seconds


def _script(client: Any) -> Any:
    """The registered Lua script for this client (EVALSHA, with EVAL fallback)."""
    key = id(client)
    script = _scripts.get(key)
    if script is None:
        script = client.register_script(_SCRIPT)
        _scripts[key] = script
    return script


def enforce(
    bucket: str,
    key: str,
    limit: Optional[int] = None,
    window: Optional[int] = None,
) -> None:
    """Count one hit for (bucket, key); raise RateLimited past the limit."""
    default_limit, default_window = _configured(bucket)
    limit = default_limit if limit is None else limit
    window = default_window if window is None else window
    bucket_key = f"rate:{bucket}:{key}"
    try:
        client = db.redis_client()
        count, ttl = _script(client)(keys=[bucket_key], args=[window * 1000])
        count = int(count)
        ttl = int(ttl)
    except Exception as exc:  # Redis down, timeout, script error ...
        logger.warning("rate limiter unavailable (%s): allowing %s", exc, bucket)
        return
    if count > limit:
        # PTTL is -1 only if the key lost its expiry, which cannot happen here;
        # fall back to the window rather than sending a non-positive header.
        retry = math.ceil(ttl / 1000) if ttl > 0 else window
        raise RateLimited(bucket, limit, window, retry)


def exceeded(
    bucket: str,
    key: str,
    limit: Optional[int] = None,
    window: Optional[int] = None,
) -> Optional[int]:
    """Retry-After seconds when this key is already at its limit, else None.

    Reading the counter (without counting) is what lets a route refuse *before*
    doing the expensive thing — bcrypt on a wrong password — while still only
    ever counting failures.
    """
    default_limit, default_window = _configured(bucket)
    limit = default_limit if limit is None else limit
    window = default_window if window is None else window
    bucket_key = f"rate:{bucket}:{key}"
    try:
        client = db.redis_client()
        with client.pipeline() as pipe:
            pipe.get(bucket_key)
            pipe.pttl(bucket_key)
            count, ttl = pipe.execute()
    except Exception as exc:
        logger.warning("rate limiter unavailable (%s): allowing %s", exc, bucket)
        return None
    try:
        hits = int(count) if count is not None else 0
    except (TypeError, ValueError):
        return None
    if hits < limit:
        return None
    try:
        seconds = int(ttl)
    except (TypeError, ValueError):
        seconds = -1
    return max(1, math.ceil(seconds / 1000)) if seconds > 0 else window


def reset(bucket: str, key: str) -> None:
    """Clear a counter (tests, and after a successful login)."""
    try:
        db.redis_client().delete(f"rate:{bucket}:{key}")
    except Exception as exc:
        logger.warning("could not clear %s counter: %s", bucket, exc)
