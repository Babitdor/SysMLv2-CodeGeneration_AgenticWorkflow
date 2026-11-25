"""Email+password accounts and server-side sessions.

The browser holds an opaque session token; only its sha256 is stored, so a
database dump cannot be replayed as a login, and logout revokes the row instead
of relying on the client forgetting the cookie. Passwords are bcrypt at cost 12.

The first account to register becomes the administrator (it owns the provider
settings, the reset button and the approval store); set SYSML_ALLOW_SIGNUP=0 to
close registration once that account exists.

Every function takes the caller's connection, or opens one, so a route can run
registration and its first session in a single transaction.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from webapp import db

SESSION_COOKIE = "sysml_session"
SESSION_TTL_DAYS = 30
MIN_PASSWORD_CHARS = 10
# bcrypt hashes at most 72 bytes and raises past it: reject rather than 500.
MAX_PASSWORD_BYTES = 72
BCRYPT_ROUNDS = 12

# Serialises registration so two racing signups cannot both see "no users yet"
# and both claim the admin role. Transaction-scoped: released on commit.
REGISTRATION_LOCK_KEY = 8_274_119_005

_EMAIL = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")


class AuthError(Exception):
    """A rejected login/registration; the route turns it into JSON + status."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def normalize_email(raw: Any) -> str:
    """Trim, lowercase and validate an address; raises AuthError(400)."""
    email = str(raw or "").strip().lower()
    if not email or len(email) > 254 or not _EMAIL.match(email):
        raise AuthError("enter a valid email address")
    return email


def _check_password(password: Any) -> str:
    text = str(password or "")
    if len(text) < MIN_PASSWORD_CHARS:
        raise AuthError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
    if len(text.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise AuthError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
    return text


def hash_password(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(
        _check_password(password).encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    import bcrypt

    try:
        return bcrypt.checkpw(
            str(password or "").encode("utf-8")[:MAX_PASSWORD_BYTES],
            str(password_hash or "").encode("utf-8"),
        )
    except (ValueError, TypeError):
        # A malformed hash in the row is a failed login, never a 500.
        return False


def token_hash(raw: str) -> str:
    return hashlib.sha256(str(raw or "").encode("utf-8")).hexdigest()


def signup_open() -> bool:
    """Registration is allowed unless it was closed after the first admin."""
    return (os.environ.get("SYSML_ALLOW_SIGNUP") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _user_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "email": str(row["email"]),
        "role": str(row.get("role") or "user"),
    }


def register(
    email: Any, password: Any, conn: Optional[Any] = None
) -> Dict[str, Any]:
    """Create an account; the very first one is the administrator."""
    address = normalize_email(email)
    secret = _check_password(password)
    hashed = hash_password(secret)

    def run(connection: Any) -> Dict[str, Any]:
        with connection.transaction():
            connection.execute("select pg_advisory_xact_lock(%s)", (REGISTRATION_LOCK_KEY,))
            existing = connection.execute(
                "select 1 from users where lower(email) = %s", (address,)
            ).fetchone()
            if existing:
                raise AuthError("that email is already registered", status=409)
            first = connection.execute("select count(*) as n from users").fetchone()
            role = "admin" if int(first["n"] if first else 0) == 0 else "user"
            if role == "user" and not signup_open():
                raise AuthError("registration is closed on this deployment", status=403)
            row = connection.execute(
                "insert into users (email, password_hash, role) "
                "values (%s, %s, %s) returning id, email, role",
                (address, hashed, role),
            ).fetchone()
        return _user_row(row)

    if conn is not None:
        return run(conn)
    with db.connection() as connection:
        return run(connection)


def authenticate(email: Any, password: Any) -> Optional[Dict[str, Any]]:
    """The account for these credentials, or None (same answer for both halves)."""
    address = str(email or "").strip().lower()
    if not address:
        return None
    with db.connection() as connection:
        row = connection.execute(
            "select id, email, role, password_hash from users where lower(email) = %s",
            (address,),
        ).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return _user_row(row)


def create_session(user_id: str) -> str:
    """Mint a token for this user and store its hash; returns the raw token."""
    raw = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)
    with db.connection() as connection:
        connection.execute(
            "delete from sessions where user_id = %s and expires_at < now()",
            (user_id,),
        )
        connection.execute(
            "insert into sessions (token_hash, user_id, expires_at) values (%s, %s, %s)",
            (token_hash(raw), user_id, expires),
        )
    return raw


def user_from_token(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """The live account behind a cookie value; None when absent/expired/revoked."""
    if not raw:
        return None
    with db.connection() as connection:
        row = connection.execute(
            "select u.id, u.email, u.role from sessions s "
            "join users u on u.id = s.user_id "
            "where s.token_hash = %s and s.revoked_at is null and s.expires_at > now()",
            (token_hash(raw),),
        ).fetchone()
    return _user_row(row) if row else None


def revoke_session(raw: Optional[str]) -> bool:
    """Log out: mark the row revoked so the cookie stops working everywhere."""
    if not raw:
        return False
    with db.connection() as connection:
        row = connection.execute(
            "update sessions set revoked_at = now() "
            "where token_hash = %s and revoked_at is null returning 1",
            (token_hash(raw),),
        ).fetchone()
    return row is not None


def count_users() -> int:
    with db.connection() as connection:
        row = connection.execute("select count(*) as n from users").fetchone()
    return int(row["n"] if row else 0)
