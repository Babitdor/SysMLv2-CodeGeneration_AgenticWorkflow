"""The only page the server serves is the built React app."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from support import sse_events
from webapp import server


def test_index_serves_the_built_app(monkeypatch, tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    built = ui / "index.html"
    built.write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    monkeypatch.setattr(server, "UI_DIR", ui)

    response = asyncio.run(server.index())

    assert Path(response.path) == built


def test_index_names_the_build_step_when_the_bundle_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "UI_DIR", tmp_path / "absent")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(server.index())

    assert excinfo.value.status_code == 503
    assert "npm run build" in excinfo.value.detail


def test_the_legacy_page_is_gone():
    # Retired: the vanilla page and its route are not part of the app any more.
    assert not hasattr(server, "legacy_index")
    assert not (server.STATIC_DIR / "index.html").exists()

async def test_an_attached_document_is_visible_in_the_trace(user_a, stub, workspace):
    """The chunks injected into the prompt come back as one retrieval row."""
    uploaded = await user_a.client.post(
        "/api/documents", files=[("files", ("spec.txt", b"Traglast 50 kg", "text/plain"))]
    )
    assert uploaded.status_code == 200, uploaded.text
    doc_id = uploaded.json()["documents"][0]["id"]

    stub.route = "chat"
    stub.reply = "the load is 50 kg"

    response = await user_a.client.post(
        "/api/chat", json={"message": "Traglast?", "doc_ids": [doc_id]}
    )
    assert response.status_code == 200, response.text

    events = sse_events(response.text)
    retrievals = [e for e in events if e.get("name") == "retrieve_documents"]
    assert len(retrievals) == 1
    retrieval = retrievals[0]
    assert retrieval["call_id"] == "retrieval"
    assert retrieval["status"] == "done"
    assert "Traglast 50 kg" in retrieval["output"]
    assert retrieval["chars"] >= len(retrieval["output"])
    # The read is traced before the model starts answering.
    first_token = next(i for i, e in enumerate(events) if e.get("type") == "token")
    assert events.index(retrieval) < first_token


async def test_the_generation_route_traces_the_retrieval_too(user_a, stub, workspace):
    uploaded = await user_a.client.post(
        "/api/documents", files=[("files", ("spec.txt", b"Traglast 50 kg", "text/plain"))]
    )
    assert uploaded.status_code == 200, uploaded.text
    doc_id = uploaded.json()["documents"][0]["id"]

    stub.route = "sysml"
    stub.reply = "here is the model"
    stub.code = "package Spec { part def Load; }"

    response = await user_a.client.post(
        "/api/chat", json={"message": "model it", "doc_ids": [doc_id]}
    )
    assert response.status_code == 200, response.text

    events = sse_events(response.text)
    names = [e.get("name") for e in events]
    assert names.count("retrieve_documents") == 1
    first_code = next(i for i, e in enumerate(events) if e.get("type") == "code")
    assert names.index("retrieve_documents") < first_code


async def test_a_turn_without_documents_traces_no_retrieval(user_a, stub, workspace):
    stub.route = "chat"

    response = await user_a.client.post("/api/chat", json={"message": "hello"})

    events = sse_events(response.text)
    assert not [e for e in events if e.get("name") == "retrieve_documents"]
    # The prompt is unchanged: no empty block for documents nobody attached.
    assert "SOURCE DOCUMENTS" not in str(stub.prompts[-1])

