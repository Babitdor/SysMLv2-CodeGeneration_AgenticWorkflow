"""Saved backend configurations: what the settings dialog lists and switches.

The invariant worth guarding: switching configurations replaces the override set
wholesale, so an api key (or base url) saved in one configuration can never leak
into the next one. Settings now live in app_settings, so every worker sees the
same ones and a restart keeps them.
"""

from __future__ import annotations

import json

from webapp import server


def overrides() -> dict:
    return server.settings_snapshot(force=True)["overrides"]


async def test_a_saved_configuration_is_listed_and_active(user_a):
    saved = await user_a.client.post(
        "/api/config",
        json={
            "provider": "openai",
            "model": "stealth/glyph-cluster::free",
            "temperature": 0.4,
            "base_url": "https://api.example.test/v1",
            "api_key": "sk-secret",
            "name": "openai hosted",
        },
    )
    assert saved.status_code == 200

    config = (await user_a.client.get("/api/config")).json()

    assert config["active"] == "openai hosted"
    assert [profile["name"] for profile in config["profiles"]] == ["openai hosted"]
    profile = config["profiles"][0]
    assert profile["provider"] == "openai"
    assert profile["model"] == "stealth/glyph-cluster::free"
    assert profile["temperature"] == 0.4
    assert profile["base_url"] == "https://api.example.test/v1"
    assert profile["api_key_set"] is True
    # The key itself is never handed to the browser.
    assert "sk-secret" not in json.dumps(config)


async def test_saving_without_a_name_leaves_no_active_configuration(user_a):
    await user_a.client.post("/api/config", json={"provider": "openai", "name": "one"})
    await user_a.client.post("/api/config", json={"model": "gpt-5"})

    config = (await user_a.client.get("/api/config")).json()

    assert config["active"] is None
    assert [profile["name"] for profile in config["profiles"]] == ["one"]


async def test_switching_replaces_instead_of_merging(user_a):
    await user_a.client.post(
        "/api/config",
        json={
            "provider": "openai",
            "base_url": "https://a.test",
            "api_key": "sk-a",
            "name": "a",
        },
    )
    # Saved while a is still live: b must not inherit a's url or key.
    await user_a.client.post(
        "/api/config",
        json={"provider": "ollama", "model": "qwen3-coder:480b-cloud", "name": "b"},
    )
    profiles = server.settings_snapshot(force=True)["profiles"]["profiles"]
    assert profiles["b"] == {
        "SYSML_PROVIDER": "ollama",
        "SYSML_MODEL": "qwen3-coder:480b-cloud",
    }
    assert profiles["a"]["SYSML_API_KEY"] == "sk-a"

    assert (await user_a.client.post("/api/config/profiles/a/apply")).status_code == 200
    assert overrides()["SYSML_BASE_URL"] == "https://a.test"
    assert overrides()["SYSML_API_KEY"] == "sk-a"

    assert (await user_a.client.post("/api/config/profiles/b/apply")).status_code == 200

    # b never set a base url or a key: it must not inherit a's.
    applied = overrides()
    assert applied["SYSML_PROVIDER"] == "ollama"
    assert applied["SYSML_MODEL"] == "qwen3-coder:480b-cloud"
    assert "SYSML_BASE_URL" not in applied
    assert "SYSML_API_KEY" not in applied
    assert (await user_a.client.get("/api/config")).json()["active"] == "b"


async def test_resaving_the_active_configuration_keeps_its_key(user_a):
    await user_a.client.post(
        "/api/config", json={"provider": "openai", "api_key": "sk-a", "name": "a"}
    )
    # Re-save under the same name with the key field blank ("keep the stored one").
    await user_a.client.post(
        "/api/config", json={"provider": "openai", "model": "gpt-5", "name": "a"}
    )

    config = (await user_a.client.get("/api/config")).json()

    assert config["profiles"][0]["api_key_set"] is True
    assert "sk-a" not in json.dumps(config)


async def test_switching_to_an_unknown_name_is_404(user_a):
    response = await user_a.client.post("/api/config/profiles/nope/apply")

    assert response.status_code == 404
    assert "unknown configuration" in response.json()["detail"]


async def test_deleting_a_configuration_clears_it_as_active(user_a):
    await user_a.client.post("/api/config", json={"provider": "openai", "name": "a"})

    deleted = await user_a.client.delete("/api/config/profiles/a")

    assert deleted.status_code == 200
    assert deleted.json()["active"] is None
    assert (await user_a.client.get("/api/config")).json()["profiles"] == []


async def test_reset_clears_the_overrides_but_keeps_the_account(user_a):
    await user_a.client.post("/api/config", json={"provider": "openai", "api_key": "sk-a"})

    reset = await user_a.client.post("/api/reset")

    assert reset.status_code == 200
    assert overrides() == {}
    assert (await user_a.client.get("/api/auth/me")).status_code == 200


async def test_a_bad_provider_is_refused(user_a):
    response = await user_a.client.post("/api/config", json={"provider": "steam-engine"})

    assert response.status_code == 400
    assert "unknown provider" in response.json()["detail"]
