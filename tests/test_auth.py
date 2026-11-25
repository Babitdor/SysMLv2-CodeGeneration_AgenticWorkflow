"""Accounts and sessions: the identity model every other test depends on.

Guarded here: passwords are bcrypt, the stored session is a hash rather than the
token, expired and revoked sessions stop working, and only the first account
(the administrator) may change deployment settings.
"""

from __future__ import annotations

import pytest

from webapp import auth, db

PASSWORD = "correct-horse-battery"


# --- hashing ---------------------------------------------------------------


def test_bcrypt_round_trip():
    digest = auth.hash_password(PASSWORD)
    assert digest != PASSWORD
    assert auth.verify_password(PASSWORD, digest) is True
    assert auth.verify_password("wrong", digest) is False


def test_a_password_longer_than_bcrypt_allows_is_rejected_not_crashed():
    with pytest.raises(auth.AuthError, match="at most"):
        auth.hash_password("x" * 100)


def test_a_short_password_is_rejected():
    with pytest.raises(auth.AuthError, match="at least"):
        auth.hash_password("short")


def test_a_corrupt_hash_is_a_failed_login():
    assert auth.verify_password(PASSWORD, "not-a-bcrypt-hash") is False


def test_the_session_row_never_holds_the_raw_token():
    user = auth.register("hash@example.test", PASSWORD)
    token = auth.create_session(user["id"])
    with db.connection() as conn:
        rows = conn.execute("select token_hash from sessions").fetchall()
    assert [row["token_hash"] for row in rows] == [auth.token_hash(token)]
    assert token not in [row["token_hash"] for row in rows]


def test_expired_and_revoked_sessions_stop_working():
    user = auth.register("expiry@example.test", PASSWORD)
    token = auth.create_session(user["id"])
    assert auth.user_from_token(token) is not None

    with db.connection() as conn:
        conn.execute(
            "update sessions set expires_at = now() - interval '1 second' "
            "where token_hash = %s",
            (auth.token_hash(token),),
        )
    assert auth.user_from_token(token) is None

    fresh = auth.create_session(user["id"])
    assert auth.revoke_session(fresh) is True
    assert auth.user_from_token(fresh) is None


# --- registration and login over HTTP ---------------------------------------


async def test_the_first_account_is_the_administrator(make_user):
    first = await make_user()
    second = await make_user()
    assert first.user["role"] == "admin"
    assert second.user["role"] == "user"


async def test_a_duplicate_email_is_a_conflict(make_user):
    user = await make_user(email="dup@example.test")
    response = await user.client.post(
        "/api/auth/register",
        json={"email": "DUP@example.test", "password": PASSWORD},
    )
    assert response.status_code == 409
    assert "already registered" in response.json()["detail"]


async def test_a_short_password_is_refused(make_user):
    user = await make_user()
    response = await user.client.post(
        "/api/auth/register", json={"email": "new@example.test", "password": "short"}
    )
    assert response.status_code == 400


async def test_signup_can_be_closed_after_the_first_admin(make_user, monkeypatch):
    await make_user()
    monkeypatch.setenv("SYSML_ALLOW_SIGNUP", "0")

    async with _client() as http:
        closed = await http.post(
            "/api/auth/register", json={"email": "late@example.test", "password": PASSWORD}
        )

    assert closed.status_code == 403
    assert "closed" in closed.json()["detail"]


async def test_wrong_password_and_unknown_email_look_the_same(make_user):
    user = await make_user()
    wrong = await user.client.post(
        "/api/auth/login", json={"email": user.email, "password": "not-the-password"}
    )
    unknown = await user.client.post(
        "/api/auth/login", json={"email": "nobody@example.test", "password": PASSWORD}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


async def test_login_issues_a_working_cookie(make_user):
    user = await make_user()
    assert (await user.client.get("/api/auth/me")).status_code == 200
    assert user.client.cookies.get(auth.SESSION_COOKIE)


async def test_logout_revokes_the_session_server_side(make_user):
    user = await make_user()
    token = user.client.cookies.get(auth.SESSION_COOKIE)
    assert (await user.client.post("/api/auth/logout")).status_code == 200
    # The cookie may still be replayed by an attacker who copied it: the row is
    # the thing that stops working.
    assert (await user.client.get("/api/auth/me")).status_code == 401
    assert auth.user_from_token(token) is None


async def test_every_data_route_requires_a_session(client):
    for path in (
        "/api/threads",
        "/api/documents",
        "/api/config",
        "/api/auth/me",
        "/api/render/state",
    ):
        response = await client.get(path)
        assert response.status_code == 401, path
        assert response.json()["detail"] == "authentication required"


async def test_deployment_settings_are_admin_only(user_a, user_b):
    assert (await user_a.client.get("/api/config")).status_code == 200
    forbidden = await user_b.client.get("/api/config")
    assert forbidden.status_code == 403
    assert "administrator" in forbidden.json()["detail"]
    reset = await user_b.client.post("/api/reset")
    assert reset.status_code == 403
    approve = await user_b.client.post("/api/approve", json={"code": "package A;"})
    assert approve.status_code == 403


def _client():
    """A throwaway client for requests that must not reuse another's cookies."""
    import httpx

    from webapp import server

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://testserver"
    )
