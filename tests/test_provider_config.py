"""Per-account provider configuration: an own endpoint, an own key, nothing shared.

The contract: a row is that account's own backend, no row falls back to the
deployment default, one account can never see or change another's row, and a
stored key is write-only — the dialog learns api_key_set, never the key.
"""

from __future__ import annotations

import pytest

from webapp import db, providers, server

# Loopback is refused by default; these tests want a local endpoint shape.
LOCAL = {"SYSML_ALLOW_PRIVATE_ENDPOINTS": "1"}


def payload(**overrides):
    body = {
        "provider": "openai",
        "model": "gpt-5.2-mini",
        "base_url": "http://127.0.0.1:11434/v1",
        "temperature": 0.4,
        "effort": "high",
    }
    body.update(overrides)
    return body


async def test_requires_a_session(client):
    assert (await client.get("/api/provider-config")).status_code == 401
    assert (await client.put("/api/provider-config", json=payload())).status_code == 401
    assert (await client.delete("/api/provider-config")).status_code == 401


async def test_no_row_follows_the_deployment(user_b):
    response = await user_b.client.get("/api/provider-config")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "deployment"
    # The form opens on what the account is actually using, and the key is
    # described, never echoed.
    assert body["model"] == server.active_backend(server.effective_env())[1]
    assert "api_key" not in body


async def test_saved_row_is_that_accounts_backend_only(user_a, user_b, monkeypatch):
    monkeypatch.setenv("SYSML_ALLOW_PRIVATE_ENDPOINTS", "1")
    saved = await user_b.client.put("/api/provider-config", json=payload())
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["source"] == "user"
    assert body["model"] == "gpt-5.2-mini"
    assert body["temperature"] == 0.4
    assert body["effort"] == "high"
    assert "api_key" not in body

    # The row reaches that account's turns...
    own = server.effective_env(user_b.id)
    assert server.active_backend(own) == ("openai", "gpt-5.2-mini")
    assert own["SYSML_BASE_URL"] == "http://127.0.0.1:11434/v1"
    # ...and is a different backend, so it gets its own workflow...
    assert providers.signature(own) != providers.signature(server.effective_env())
    # ...while everyone else keeps the deployment default.
    other = await user_a.client.get("/api/provider-config")
    assert other.json()["source"] == "deployment"
    assert other.json()["model"] != "gpt-5.2-mini"
    assert server.active_backend(server.effective_env(user_a.id))[1] != "gpt-5.2-mini"


async def test_clear_restores_the_deployment_default(user_b, monkeypatch):
    monkeypatch.setenv("SYSML_ALLOW_PRIVATE_ENDPOINTS", "1")
    assert (await user_b.client.put("/api/provider-config", json=payload())).status_code == 200
    cleared = await user_b.client.delete("/api/provider-config")
    assert cleared.status_code == 200
    assert cleared.json()["source"] == "deployment"
    assert providers.overrides(user_b.id) == {}
    with db.connection() as conn:
        row = conn.execute(
            "select count(*) as total from user_provider_config where user_id = %s",
            (user_b.id,),
        ).fetchone()
    assert row["total"] == 0


async def test_private_endpoint_needs_the_operator_opt_in(user_b):
    response = await user_b.client.put(
        "/api/provider-config", json=payload(base_url="http://127.0.0.1:8080/v1")
    )
    assert response.status_code == 400
    assert "SYSML_ALLOW_PRIVATE_ENDPOINTS" in response.json()["detail"]


@pytest.mark.parametrize(
    "bad, detail",
    [
        ({"provider": "not-a-provider"}, "unknown provider"),
        ({"model": "  "}, "model is required"),
        ({"temperature": 3}, "between 0 and 2"),
        ({"base_url": "ftp://example.test"}, "http://"),
        ({"base_url": "http://user:pw@example.test"}, "credentials"),
    ],
)
async def test_bad_input_is_a_400(user_b, monkeypatch, bad, detail):
    monkeypatch.setenv("SYSML_ALLOW_PRIVATE_ENDPOINTS", "1")
    response = await user_b.client.put("/api/provider-config", json=payload(**bad))
    assert response.status_code == 400, response.text
    assert detail in response.json()["detail"]


async def test_key_is_write_only_and_stored_encrypted(user_b, monkeypatch):
    monkeypatch.setenv("SYSML_ALLOW_PRIVATE_ENDPOINTS", "1")
    monkeypatch.setenv("SYSML_SECRET_KEY", "test-secret")
    monkeypatch.setattr(providers, "_fernet_instance", None)

    response = await user_b.client.put(
        "/api/provider-config", json=payload(api_key="sk-test-1234567890")
    )
    assert response.status_code == 200, response.text
    assert response.json()["api_key_set"] is True
    assert "sk-test-1234567890" not in response.text
    # A stored key is what the account's turns authenticate with.
    assert providers.overrides(user_b.id)["SYSML_API_KEY"] == "sk-test-1234567890"

    with db.connection() as conn:
        cipher = conn.execute(
            "select api_key_cipher from user_provider_config where user_id = %s",
            (user_b.id,),
        ).fetchone()["api_key_cipher"]
    assert cipher and "sk-test-1234567890" not in cipher

    # Blank keeps the stored key, which is what the form submits untouched.
    again = await user_b.client.put("/api/provider-config", json=payload(api_key=""))
    assert again.status_code == 200
    assert again.json()["api_key_set"] is True
    assert providers.overrides(user_b.id)["SYSML_API_KEY"] == "sk-test-1234567890"


async def test_key_without_a_secret_is_refused(user_b, monkeypatch):
    monkeypatch.setenv("SYSML_ALLOW_PRIVATE_ENDPOINTS", "1")
    monkeypatch.delenv("SYSML_SECRET_KEY", raising=False)
    response = await user_b.client.put(
        "/api/provider-config", json=payload(api_key="sk-test-1234567890")
    )
    assert response.status_code == 400, response.text
    assert "SYSML_SECRET_KEY" in response.json()["detail"]
