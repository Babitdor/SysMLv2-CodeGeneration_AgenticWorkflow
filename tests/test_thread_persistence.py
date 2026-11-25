"""A generated model must survive a refresh, and the render endpoints must work.

The original bug: a turn whose content was empty (the pipeline emitted a code
frame with no prose) was dropped by the transcript reader, and the model only
existed as a fenced block inside that prose. The turn now carries model and
verdict as columns — in Postgres, so the next worker and the next restart see it.
"""

from __future__ import annotations

from webapp import server, threads

CODE = "package Test {\n    part def Wheel;\n}"
VERDICT = {
    "status": "valid",
    "ok": True,
    "attempt": 1,
    "final": True,
    "diagnostics": [],
    "summary": "no problems",
}


def turns(owner: str, thread_id: str):
    loaded = threads.load_thread(owner, thread_id)
    assert loaded is not None
    return loaded["messages"]


# --- storage ---------------------------------------------------------------


async def test_prose_and_model_are_stored_apart(user_a):
    thread = threads.create_thread(user_a.id, "chat")
    threads.append_message(user_a.id, thread["id"], "assistant", "Here it is.", model=CODE)

    (turn,) = turns(user_a.id, thread["id"])
    assert turn["content"] == "Here it is."
    assert turn["model"] == CODE


async def test_a_model_only_turn_is_kept(user_a):
    thread = threads.create_thread(user_a.id, "chat")
    threads.append_message(
        user_a.id, thread["id"], "assistant", "", model=CODE, validation=VERDICT
    )

    (turn,) = turns(user_a.id, thread["id"])
    assert turn["content"] == ""
    assert turn["model"] == CODE
    assert turn["validation"] == VERDICT


async def test_an_empty_turn_is_rejected(user_a):
    thread = threads.create_thread(user_a.id, "chat")

    assert (
        threads.append_message(user_a.id, thread["id"], "assistant", "  ", model=" ")
        is None
    )
    assert turns(user_a.id, thread["id"]) == []


async def test_a_model_only_turn_adds_nothing_to_the_prompt(user_a):
    thread = threads.create_thread(user_a.id, "chat")
    threads.append_message(user_a.id, thread["id"], "user", "model a wheel")
    threads.append_message(user_a.id, thread["id"], "assistant", "", model=CODE)

    history = threads.history_messages(threads.load_thread(user_a.id, thread["id"]))

    # The model-only turn has no prose to replay; an empty message would be
    # rejected by some providers, and the model travels in "artifact" instead.
    assert [turn["role"] for turn in history] == ["user"]


# --- the endpoint that dropped the turn ------------------------------------


async def test_a_code_turn_survives_the_stream(user_a, stub, workspace):
    stub.route = "generation"
    stub.reply = "Here you go."
    stub.code = CODE
    stub.validation = VERDICT

    created = await user_a.client.post("/api/threads", json={})
    thread_id = created.json()["thread"]["id"]

    streamed = await user_a.client.post(
        f"/api/threads/{thread_id}/chat", json={"message": "model a wheel"}
    )
    assert streamed.status_code == 200
    assert "event: code" in streamed.text
    assert "event: done" in streamed.text

    stored = await user_a.client.get(f"/api/threads/{thread_id}")
    last = stored.json()["thread"]["messages"][-1]
    assert last["role"] == "assistant"
    assert last["model"] == CODE
    assert last["validation"]["ok"] is True

    # The generated model is what the diagram panel then draws.
    state = (await user_a.client.get("/api/render/state")).json()
    assert state["has_model"] is True

    assert (await user_a.client.delete(f"/api/threads/{thread_id}")).status_code == 200
    assert (await user_a.client.get(f"/api/threads/{thread_id}")).status_code == 404


async def test_a_dropped_turn_keeps_the_question(user_a, stub, workspace):
    """The user turn is saved before the stream, so a failure is not silent."""
    stub.route = "generation"

    created = await user_a.client.post("/api/threads", json={})
    thread_id = created.json()["thread"]["id"]
    await user_a.client.post(f"/api/threads/{thread_id}/chat", json={"message": "hello"})

    roles = [turn["role"] for turn in turns(user_a.id, thread_id)]
    assert roles[:1] == ["user"]


# --- render ----------------------------------------------------------------


async def test_render_passes_through_and_caches(user_a, monkeypatch):
    data_url = "data:image/svg+xml;base64,PHN2Zy8+"
    calls = []

    def fake_kernel(code, element_name="", orientation="TB", compartments="default"):
        calls.append(code)
        return {"success": True, "data_url": data_url}

    monkeypatch.setattr(server, "_kernel_render", fake_kernel)
    monkeypatch.setattr(server, "_render_cache", {})

    body = {"code": CODE, "orientation": "LR", "compartments": "tree"}
    first = await user_a.client.post("/api/render", json=body)
    assert first.status_code == 200
    assert first.json()["data_url"] == data_url
    assert first.json()["cached"] is False

    second = await user_a.client.post("/api/render", json=body)
    assert second.json()["cached"] is True
    assert calls == [CODE]


async def test_the_render_cache_is_not_shared_between_accounts(user_a, user_b, monkeypatch):
    calls = []

    def fake_kernel(code, element_name="", orientation="TB", compartments="default"):
        calls.append(code)
        return {"success": True, "data_url": "data:image/svg+xml;base64,PHN2Zy8+"}

    monkeypatch.setattr(server, "_kernel_render", fake_kernel)
    monkeypatch.setattr(server, "_render_cache", {})

    body = {"code": CODE}
    assert (await user_a.client.post("/api/render", json=body)).json()["cached"] is False
    theirs = await user_b.client.post("/api/render", json=body)
    assert theirs.json()["cached"] is False
    assert calls == [CODE, CODE]


async def test_render_without_a_model_is_409(user_a, workspace):
    response = await user_a.client.post("/api/render", json={})

    assert response.status_code == 409
    assert "no model" in response.json()["detail"]
