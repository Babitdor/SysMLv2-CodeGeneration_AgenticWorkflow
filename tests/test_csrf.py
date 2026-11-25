"""Cross-site request forgery: a cookie API needs the Origin checked.

SameSite=Lax already blocks most of it; these tests pin the belt-and-braces in
OriginGuardMiddleware, plus the JSON-only rule on mutating routes (a plain HTML
form cannot send application/json).
"""

from __future__ import annotations

CREDENTIALS = {"email": "csrf@example.test", "password": "correct-horse-battery"}


async def test_a_foreign_origin_cannot_write(client):
    response = await client.post(
        "/api/auth/login", json=CREDENTIALS, headers={"origin": "https://evil.test"}
    )
    assert response.status_code == 403
    assert "cross-origin" in response.json()["detail"]


async def test_the_same_origin_is_allowed(client):
    response = await client.post(
        "/api/auth/login", json=CREDENTIALS, headers={"origin": "http://testserver"}
    )
    # Not a 403: the request reached the route, which rejected the credentials.
    assert response.status_code == 401


async def test_an_explicitly_allowed_origin_is_allowed(client, monkeypatch):
    monkeypatch.setenv("SYSML_ALLOWED_ORIGINS", "https://app.example.test")
    response = await client.post(
        "/api/auth/login",
        json=CREDENTIALS,
        headers={"origin": "https://app.example.test"},
    )
    assert response.status_code == 401


async def test_a_form_encoded_write_is_refused(client):
    response = await client.post(
        "/api/auth/login",
        content=b"email=a@b.test&password=x",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 415
    assert "application/json" in response.json()["detail"]


async def test_safe_methods_are_not_origin_checked(client):
    response = await client.get(
        "/api/threads", headers={"origin": "https://evil.test"}
    )
    # Read-only and cookie-protected: the 401 is the answer, not a CSRF rejection.
    assert response.status_code == 401
