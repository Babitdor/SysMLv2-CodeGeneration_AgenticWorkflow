"""Tenant isolation: the leaks the old cookie-as-identity model allowed.

Every case here was reachable before: read another visitor's thread by id, attach
another visitor's document, see their last generated model in the diagram panel,
or drive a turn inside their conversation.
"""

from __future__ import annotations

from webapp import documents, threads

CODE = "package Alpha {\n    part def Wheel;\n}"


async def _new_thread(http) -> str:
    response = await http.post("/api/threads", json={})
    assert response.status_code == 200, response.text
    return response.json()["thread"]["id"]


async def test_thread_lists_do_not_overlap(user_a, user_b):
    mine = await _new_thread(user_a.client)

    assert [row["id"] for row in (await user_a.client.get("/api/threads")).json()["threads"]] == [
        mine
    ]
    assert (await user_b.client.get("/api/threads")).json()["threads"] == []


async def test_a_foreign_thread_is_a_404_for_every_verb(user_a, user_b):
    mine = await _new_thread(user_a.client)

    assert (await user_b.client.get(f"/api/threads/{mine}")).status_code == 404
    renamed = await user_b.client.patch(f"/api/threads/{mine}", json={"title": "stolen"})
    assert renamed.status_code == 404
    assert (await user_b.client.delete(f"/api/threads/{mine}")).status_code == 404
    chat = await user_b.client.post(
        f"/api/threads/{mine}/chat", json={"message": "hello"}
    )
    assert chat.status_code == 404

    # And it is still there, untouched, for its owner.
    still_mine = await user_a.client.get(f"/api/threads/{mine}")
    assert still_mine.status_code == 200
    assert still_mine.json()["thread"]["title"] != "stolen"


async def test_a_foreign_thread_id_is_refused_by_the_legacy_chat_route(user_a, user_b):
    mine = await _new_thread(user_a.client)

    response = await user_b.client.post(
        "/api/chat", json={"message": "continue that", "thread_id": mine}
    )

    assert response.status_code == 400
    assert "thread_id" in response.json()["detail"]


async def test_a_junk_thread_id_is_a_404_not_a_crash(user_a):
    assert (await user_a.client.get("/api/threads/not-a-uuid")).status_code == 404


async def test_documents_are_private(user_a, user_b):
    uploaded = await user_a.client.post(
        "/api/documents",
        files=[("files", ("spec.txt", b"Traglast 50 kg", "text/plain"))],
    )
    assert uploaded.status_code == 200, uploaded.text
    document_id = uploaded.json()["documents"][0]["id"]

    assert [row["id"] for row in (await user_a.client.get("/api/documents")).json()["documents"]] == [
        document_id
    ]
    assert (await user_b.client.get("/api/documents")).json()["documents"] == []
    # Naming the id is not enough: retrieval filters by owner too.
    assert documents.search(user_b.id, [document_id], "Traglast") == []
    assert documents.search(user_a.id, [document_id], "Traglast")


async def test_the_last_model_is_per_account(user_a, user_b, stub, workspace):
    stub.route = "generation"
    stub.reply = "Here is a model."
    stub.code = CODE
    stub.validation = {"status": "valid", "ok": True}
    thread_id = await _new_thread(user_a.client)

    streamed = await user_a.client.post(
        f"/api/threads/{thread_id}/chat", json={"message": "model a wheel"}
    )
    assert streamed.status_code == 200
    assert "event: code" in streamed.text

    mine = await user_a.client.get("/api/render/latest")
    assert mine.status_code == 200

    # B has generated nothing: the panel must say so instead of drawing A's model.
    theirs = await user_b.client.get("/api/render/latest")
    assert theirs.status_code == 409
    state = (await user_b.client.get("/api/render/state")).json()
    assert state == {"revision": 0, "has_model": False}
    assert (await user_a.client.get("/api/render/state")).json()["has_model"] is True


async def test_a_conversation_stays_readable_by_its_owner_only(user_a, user_b, stub, workspace):
    stub.route = "chat"
    stub.reply = "hello there"
    thread_id = await _new_thread(user_a.client)

    await user_a.client.post(f"/api/threads/{thread_id}/chat", json={"message": "hi"})

    mine = (await user_a.client.get(f"/api/threads/{thread_id}")).json()["thread"]
    assert [turn["role"] for turn in mine["messages"]] == ["user", "assistant"]
    assert (await user_b.client.get(f"/api/threads/{thread_id}")).status_code == 404
    # A foreign id reads as absent at the store level as well.
    assert threads.load_thread(user_b.id, thread_id) is None
