"""Rate limits and admission control.

The point is not the exact numbers but the shape: a per-account budget for turns
and uploads, a failing-sign-in lockout keyed by address+email, a 429 that tells
the client when to come back, and a limiter that fails open when Redis is gone.
"""

from __future__ import annotations

import pytest

from webapp import limits


async def _new_thread(http) -> str:
    response = await http.post("/api/threads", json={})
    assert response.status_code == 200, response.text
    return response.json()["thread"]["id"]


async def test_the_chat_budget_is_per_account(monkeypatch, user_a, user_b, stub, workspace):
    monkeypatch.setenv("SYSML_RATE_CHAT", "2/60")
    stub.route = "chat"
    stub.reply = "ok"
    thread_id = await _new_thread(user_a.client)

    first = await user_a.client.post(f"/api/threads/{thread_id}/chat", json={"message": "one"})
    second = await user_a.client.post(f"/api/threads/{thread_id}/chat", json={"message": "two"})
    third = await user_a.client.post(f"/api/threads/{thread_id}/chat", json={"message": "three"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "chat" in third.json()["detail"]
    retry_after = int(third.headers["Retry-After"])
    assert 1 <= retry_after <= 60

    # Another account has its own budget.
    theirs = await _new_thread(user_b.client)
    allowed = await user_b.client.post(f"/api/threads/{theirs}/chat", json={"message": "hi"})
    assert allowed.status_code == 200


async def test_the_legacy_chat_route_is_limited_too(monkeypatch, user_a, stub, workspace):
    monkeypatch.setenv("SYSML_RATE_CHAT", "1/60")
    stub.route = "chat"
    stub.reply = "ok"

    assert (await user_a.client.post("/api/chat", json={"message": "one"})).status_code == 200
    assert (await user_a.client.post("/api/chat", json={"message": "two"})).status_code == 429


async def test_uploads_are_limited(monkeypatch, user_a):
    monkeypatch.setenv("SYSML_RATE_UPLOAD", "1/60")

    first = await user_a.client.post(
        "/api/documents", files=[("files", ("a.txt", b"one", "text/plain"))]
    )
    second = await user_a.client.post(
        "/api/documents", files=[("files", ("b.txt", b"two", "text/plain"))]
    )

    assert first.status_code == 200
    assert second.status_code == 429


async def test_failed_sign_ins_lock_the_address_out(monkeypatch, make_user):
    monkeypatch.setenv("SYSML_RATE_AUTH", "3/60")
    user = await make_user()

    attempts = [
        await user.client.post(
            "/api/auth/login", json={"email": user.email, "password": "wrong-password"}
        )
        for _ in range(3)
    ]
    blocked = await user.client.post(
        "/api/auth/login", json={"email": user.email, "password": user.password}
    )

    assert [response.status_code for response in attempts] == [401, 401, 401]
    # Locked out even with the right password: otherwise the lockout is decorative.
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


async def test_a_successful_sign_in_clears_the_counter(monkeypatch, make_user):
    monkeypatch.setenv("SYSML_RATE_AUTH", "2/60")
    user = await make_user()

    async def wrong():
        return await user.client.post(
            "/api/auth/login", json={"email": user.email, "password": "wrong-password"}
        )

    assert (await wrong()).status_code == 401
    ok = await user.client.post(
        "/api/auth/login", json={"email": user.email, "password": user.password}
    )
    assert ok.status_code == 200

    # The budget restarted: two fresh failures are not a lockout. Without the
    # reset the second one would already be a 429.
    assert (await wrong()).status_code == 401
    assert (await wrong()).status_code == 401


def test_the_documented_defaults():
    assert limits.DEFAULTS["chat"] == (20, 300)
    assert limits.DEFAULTS["auth"] == (10, 900)
    assert limits.DEFAULTS["upload"] == (30, 3600)


def test_a_malformed_override_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("SYSML_RATE_CHAT", "not-a-number")
    assert limits._configured("chat") == (20, 300)


def test_the_limiter_fails_open_when_redis_is_gone(monkeypatch):
    class DeadRedis:
        def register_script(self, script):
            raise ConnectionError("redis is down")

    monkeypatch.setattr(limits.db, "redis_client", lambda: DeadRedis())
    monkeypatch.setattr(limits, "_scripts", {})

    limits.enforce("chat", "someone")  # must not raise
    assert limits.exceeded("auth", "someone") is None


async def test_a_turn_over_the_budget_is_refused_before_any_work(monkeypatch, user_a, stub, workspace):
    """The counter is checked before the provider is called: no work, no cost."""
    monkeypatch.setenv("SYSML_RATE_CHAT", "1/60")
    stub.route = "chat"
    stub.reply = "ok"
    thread_id = await _new_thread(user_a.client)

    await user_a.client.post(f"/api/threads/{thread_id}/chat", json={"message": "one"})
    calls = len(stub.prompts)
    refused = await user_a.client.post(
        f"/api/threads/{thread_id}/chat", json={"message": "two"}
    )

    assert refused.status_code == 429
    assert len(stub.prompts) == calls
    with pytest.raises(limits.RateLimited):
        limits.enforce("chat", user_a.id, limit=0, window=60)
