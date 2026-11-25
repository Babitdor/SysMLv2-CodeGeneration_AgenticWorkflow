"""Liveness, readiness and structured logs.

The old /api/health built the whole workflow to answer, which is exactly wrong
for a probe behind a load balancer: it is slow, it can fail for reasons that have
nothing to do with the process being alive, and every replica pays for it.
"""

from __future__ import annotations

import json
import logging

from webapp import db, logs, server


async def test_health_is_cheap_and_never_builds_the_workflow(client, monkeypatch):
    def refuse():
        raise AssertionError("/api/health must not build the workflow")

    monkeypatch.setattr(server, "get_workflow", refuse)

    response = await client.get("/api/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["ready"] is True
    assert payload["version"]
    assert payload["provider"] and payload["model"]
    # Unknown counters are omitted rather than reported as a lie.
    assert "rag_chunks" not in payload or payload["rag_chunks"] is not None


async def test_ready_is_200_when_every_dependency_answers(client, monkeypatch):
    monkeypatch.setattr(server, "_kernel_status", lambda: None)

    response = await client.get("/api/ready")
    payload = response.json()

    assert response.status_code == 200, payload
    assert payload["ready"] is True
    assert payload["components"]["postgres"] == "ok"
    assert payload["components"]["redis"] == "ok"


async def test_ready_names_the_component_that_is_down(client, monkeypatch):
    monkeypatch.setattr(server, "_kernel_status", lambda: None)
    monkeypatch.setattr(db, "postgres_status", lambda: "OperationalError: refused")

    response = await client.get("/api/ready")
    payload = response.json()

    assert response.status_code == 503
    assert payload["ready"] is False
    assert payload["failed"] == ["postgres"]
    assert "refused" in payload["components"]["postgres"]


async def test_ready_reports_the_kernel_separately(client, monkeypatch):
    monkeypatch.setattr(server, "_kernel_status", lambda: "ConnectionRefusedError: nope")

    response = await client.get("/api/ready")

    assert response.status_code == 503
    assert response.json()["failed"] == ["kernel"]


async def test_readiness_never_builds_the_workflow(client, monkeypatch):
    def refuse():
        raise AssertionError("/api/ready must not build the workflow")

    monkeypatch.setattr(server, "get_workflow", refuse)
    monkeypatch.setattr(server, "_kernel_status", lambda: None)

    assert (await client.get("/api/ready")).status_code == 200


async def test_the_request_id_is_propagated_and_minted(client, caplog):
    caplog.set_level(logging.INFO, logger="webapp.access")

    echoed = await client.get("/api/health", headers={"x-request-id": "abc-123"})
    assert echoed.headers["x-request-id"] == "abc-123"

    minted = await client.get("/api/health", headers={"x-request-id": "bad id, spaces"})
    assert minted.headers["x-request-id"] != "bad id, spaces"
    assert len(minted.headers["x-request-id"]) == 32

    # httpx's own logger can emit after the access line, so pick the access
    # record by name rather than trusting whatever landed last.
    access = next(r for r in reversed(caplog.records) if r.name == "webapp.access")
    payload = json.loads(logs.JsonFormatter().format(access))
    assert payload["request_id"] == minted.headers["x-request-id"]
    assert payload["path"] == "/api/health"
    assert payload["status"] == 200
    assert payload["duration_ms"] >= 0


def test_every_log_line_is_json_with_its_context():
    record = logging.LogRecord(
        "webapp.server", logging.INFO, __file__, 1, "hello %s", ("world",), None
    )
    token = logs.request_id_var.set("req-123")
    try:
        payload = json.loads(logs.JsonFormatter().format(record))
    finally:
        logs.request_id_var.reset(token)

    assert payload["msg"] == "hello world"
    assert payload["level"] == "info"
    assert payload["logger"] == "webapp.server"
    assert payload["request_id"] == "req-123"
    assert payload["ts"].endswith("Z")


async def test_an_unhandled_error_carries_the_request_id_but_not_the_traceback():
    import httpx

    @server.app.get("/api/_boom")
    async def boom():
        raise RuntimeError("kaboom")

    try:
        transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as http:
            response = await http.get("/api/_boom", headers={"x-request-id": "req-abc"})
    finally:
        server.app.router.routes = [
            route
            for route in server.app.router.routes
            if getattr(route, "path", "") != "/api/_boom"
        ]

    assert response.status_code == 500
    body = response.json()
    # Support gets the id that ties the answer to the traceback in the log.
    assert body["request_id"] == "req-abc"
    assert "kaboom" not in json.dumps(body)
    assert response.headers["x-request-id"] == "req-abc"


async def test_health_reports_the_accounts_own_backend(make_user):
    """The top bar must name the model this account's turns actually run on.

    /api/health used to answer from the deployment view only, so saving a model
    in the dialog left the strip showing the old one forever.
    """
    http = (await make_user()).client
    saved = await http.put(
        "/api/provider-config",
        json={"provider": "ollama", "model": "regression-sentinel", "temperature": 0.42},
    )
    assert saved.status_code == 200, saved.text

    payload = (await http.get("/api/health")).json()

    assert payload["provider"] == "ollama"
    assert payload["model"] == "regression-sentinel"
    assert payload["temperature"] == 0.42


async def test_health_without_a_session_reports_the_deployment(client):
    """The container healthcheck calls this anonymously and must keep answering."""
    payload = (await client.get("/api/health")).json()

    assert payload["ok"] is True
    assert payload["model"] == server._deployment_view()["model"]
