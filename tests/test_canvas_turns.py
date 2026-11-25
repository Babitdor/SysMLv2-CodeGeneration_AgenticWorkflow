from types import SimpleNamespace

import httpx
import pytest
from support import sse_events

from webapp import router, server
from webapp.turn_tools import apply_replacement, web_results


def test_exact_replacement_rejects_overlapping_matches():
    with pytest.raises(ValueError, match="one occurrence"):
        apply_replacement("aaa", '{"old":"aa","new":"b"}')


def test_router_prefers_web_for_explicit_search():
    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="general_chat"))
    assert router.decide(llm, "Search the web for SysML release notes").route == router.WEB_SEARCH


@pytest.mark.parametrize(("prompt", "expected"), [
    ("How do I create a SysML package?", router.GENERAL_CHAT),
    ("Could you validate this SysML model?", router.SYSML_GENERATION),
    ("Draft an interface specification", router.DOCUMENT),
    ("Open Canvas artifact: code, Vehicle\nLatest request:\nRename the wheel", router.ARTIFACT_EDIT),
])
def test_router_fallback_distinguishes_questions_and_actions(prompt, expected):
    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="Unrecognized output"))
    assert router.decide(llm, prompt).route == expected


@pytest.mark.asyncio
async def test_canvas_edit_streams_exact_changes_without_writing_a_version(user_a, stub, monkeypatch):
    monkeypatch.setattr(
        router, "decide",
        lambda *_: router.RouteDecision(router.ARTIFACT_EDIT, "test", "llm"),
    )
    stub.reply = '{"old":"# Original","new":"# Revised"}'
    client = user_a.client
    thread = (await client.post("/api/threads", json={})).json()["thread"]
    thread_id = thread["id"]
    created = await client.post(
        f"/api/threads/{thread_id}/artifacts",
        json={"title": "Specification", "kind": "markdown", "content": "# Original\nKeep me."},
    )
    assert created.status_code == 200
    artifact_id = created.json()["artifact"]["id"]
    result = await client.post(
        f"/api/threads/{thread_id}/chat",
        json={"message": "Rename the heading", "canvas_kind": "markdown",
              "artifact": "# Original\nKeep me."},
    )
    events = sse_events(result.text)
    delta = next(event for event in events if event["type"] == "edit_delta")
    assert delta["content"] == "# Revised\nKeep me."
    assert delta["line"] == 1
    assert any(event["type"] == "edit_complete" and event["changed"] for event in events)
    assert not any(event["type"] == "artifact" for event in events)
    unchanged = (await client.get(f"/api/artifacts/{artifact_id}")).json()["artifact"]
    assert unchanged["current_version"] == 1
    assert unchanged["content"] == "# Original\nKeep me."
    saved = (await client.get(f"/api/threads/{thread_id}")).json()["thread"]
    assert saved["messages"][-1]["tool_trace"][-1]["name"] == "edit_artifact"


@pytest.mark.asyncio
async def test_document_turn_streams_and_persists_markdown(user_a, stub, monkeypatch):
    monkeypatch.setattr(
        router, "decide", lambda *_: router.RouteDecision(router.DOCUMENT, "test", "llm"),
    )
    stub.reply = "Specification body"
    client = user_a.client
    thread = (await client.post("/api/threads", json={})).json()["thread"]
    result = await client.post(
        f"/api/threads/{thread['id']}/chat", json={"message": "Draft a specification"},
    )
    events = sse_events(result.text)
    assert any(event["type"] == "document_delta" for event in events)
    artifact = next(event["artifact"] for event in events if event["type"] == "artifact")
    assert artifact["kind"] == "markdown"
    body = (await client.get(f"/api/artifacts/{artifact['id']}")).json()["artifact"]
    assert body["content"].strip() == "Specification body"


@pytest.mark.asyncio
async def test_workspace_upload_is_visible_only_in_its_account(user_a, user_b, workspace):
    result = await user_a.client.post(
        "/api/workspace/files", files=[("files", ("model.sysml", b"package A {}"))],
    )
    assert result.status_code == 200, result.text
    assert (await user_b.client.get("/api/workspace/files")).json()["files"] == []
    own = server._workspace_backend(user_a.id)
    assert own.download_files(["/uploads/model.sysml"])[0].content == b"package A {}"


async def test_chat_injects_retrieved_sysml_references(user_a, stub, monkeypatch):
    workflow = server.get_workflow(user_a.id)
    monkeypatch.setattr(workflow, "rag_pipeline", SimpleNamespace(
        get_context_for_prompt=lambda _: "Reference for part definitions",
    ), raising=False)
    response = await user_a.client.post("/api/chat", json={"message": "Explain SysML"})
    events = sse_events(response.text)
    retrieved = next(event for event in events if event.get("name") == "retrieve_sysml" and event["status"] == "done")
    assert "Reference for part definitions" in retrieved["output"]
    assert "Reference for part definitions" in str(stub.prompts[-1])


async def test_web_search_results_reach_the_reply_and_activity(user_a, stub, monkeypatch):
    monkeypatch.setattr(
        router, "decide", lambda *_: router.RouteDecision(router.WEB_SEARCH, "test", "llm"),
    )
    monkeypatch.setattr(server, "web_results", lambda *_: "SysML v2 release\nhttps://example.com/release")
    response = await user_a.client.post("/api/chat", json={"message": "Search the web for the SysML release"})
    events = sse_events(response.text)
    search = next(event for event in events if event.get("name") == "web_search" and event["status"] == "done")
    assert "https://example.com/release" in search["output"]
    assert "https://example.com/release" in str(stub.prompts[-1])


async def test_web_search_uses_only_the_signed_in_accounts_key(user_a, user_b, stub, monkeypatch):
    from webapp import providers

    monkeypatch.setenv("SYSML_SECRET_KEY", "test-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "deployment-key")
    monkeypatch.setattr(providers, "_fernet_instance", None)
    monkeypatch.setattr(
        router, "decide", lambda *_: router.RouteDecision(router.WEB_SEARCH, "test", "llm"),
    )
    keys = []

    def search(_, key):
        keys.append(key)
        return "Search result"

    monkeypatch.setattr(server, "web_results", search)
    assert (await user_a.client.put("/api/credentials/tavily", json={"api_key": "personal-search-key"})).status_code == 200
    await user_a.client.post("/api/chat", json={"message": "Search the web for SysML"})
    await user_b.client.post("/api/chat", json={"message": "Search the web for SysML"})
    assert keys == ["personal-search-key", ""]


def test_web_search_reads_tavily_sources(monkeypatch):
    def search(request):
        assert str(request.url) == "https://api.tavily.com/search"
        assert request.headers["authorization"] == "Bearer test-search-key"
        return httpx.Response(200, json={"results": [
            {"title": "SysML reference", "url": "https://example.com/reference", "content": "Example"},
        ]})

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(search), **kw))
    assert web_results("SysML query", "test-search-key") == "SysML reference\nhttps://example.com/reference\nExample"


async def test_approval_reports_storage_failure(user_a, stub, monkeypatch):
    from webapp import validation

    def fail(_, *, strict):
        assert strict
        raise OSError("Knowledge base is read-only")

    workflow = server.get_workflow(user_a.id)
    monkeypatch.setattr(workflow, "_update_knowledge_base", fail, raising=False)
    monkeypatch.setattr(validation, "validate_sysml_code", lambda _: {
        "status": "valid", "validator": "antlr",
    })
    response = await user_a.client.post("/api/approve", json={"code": "package Approved {}"})
    assert response.status_code == 503
    assert "read-only" in response.json()["detail"]
