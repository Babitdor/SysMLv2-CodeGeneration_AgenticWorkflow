"""Canvas artifacts: append-only versions, tenant isolation, capture on a turn.

The contract under test is the one the UI depends on: an edit never rewrites a
version, a restore never destroys one, another account cannot see any of it, and
a generation turn leaves a v1 behind that a reload can reopen.
"""

from __future__ import annotations

from support import event_types, sse_events

from webapp import artifacts

CODE = "package Test {\n    part def Wheel;\n}"
EDITED = "package Test {\n    part def Wheel;\n    part def Axle;\n}"
VERDICT = {
    "status": "valid",
    "ok": True,
    "attempt": 1,
    "final": True,
    "diagnostics": [],
    "summary": "no problems",
}


async def new_thread(http) -> str:
    response = await http.post("/api/threads", json={})
    assert response.status_code == 200, response.text
    return response.json()["thread"]["id"]


async def create(user, thread_id: str, **overrides):
    body = {"title": "wheel", "kind": "code", "content": CODE, **overrides}
    return await user.client.post(f"/api/threads/{thread_id}/artifacts", json=body)


async def generated_code(user, stub, workspace) -> tuple[str, str]:
    """One generation turn; returns (thread id, artifact id) as the UI reads them."""
    stub.route = "generation"
    stub.reply = "Here you go."
    stub.code = CODE
    stub.validation = VERDICT
    thread_id = await new_thread(user.client)

    streamed = await user.client.post(
        f"/api/threads/{thread_id}/chat", json={"message": "model a wheel"}
    )
    assert streamed.status_code == 200
    assert "artifact" in event_types(streamed.text)

    stored = (await user.client.get(f"/api/threads/{thread_id}")).json()["thread"]
    return thread_id, stored["messages"][-1]["artifact_id"]


# --- versioning ------------------------------------------------------------


async def test_a_new_artifact_starts_at_v1(user_a):
    thread_id = await new_thread(user_a.client)

    response = await create(user_a, thread_id)

    assert response.status_code == 200, response.text
    artifact = response.json()["artifact"]
    assert artifact["current_version"] == 1
    assert artifact["version"] == 1
    assert artifact["kind"] == "code"
    assert artifact["content"] == CODE
    assert [(v["version"], v["origin"]) for v in artifact["versions"]] == [(1, "generate")]


async def test_an_edit_appends_and_the_old_version_stays_readable(user_a):
    thread_id = await new_thread(user_a.client)
    created = (await create(user_a, thread_id)).json()["artifact"]

    saved = await user_a.client.post(
        f"/api/artifacts/{created['id']}/versions", json={"kind": "code", "content": EDITED}
    )

    assert saved.status_code == 200, saved.text
    edited = saved.json()["artifact"]
    assert edited["current_version"] == 2
    assert edited["content"] == EDITED

    older = await user_a.client.get(f"/api/artifacts/{created['id']}?version=1")
    assert older.status_code == 200
    # Time travel: v1 is unchanged and still what the pointer used to hold.
    assert older.json()["artifact"]["content"] == CODE
    assert older.json()["artifact"]["current_version"] == 2


async def test_a_restore_appends_instead_of_rewriting(user_a):
    thread_id = await new_thread(user_a.client)
    artifact_id = (await create(user_a, thread_id)).json()["artifact"]["id"]
    await user_a.client.post(
        f"/api/artifacts/{artifact_id}/versions", json={"kind": "code", "content": EDITED}
    )

    restored = await user_a.client.post(
        f"/api/artifacts/{artifact_id}/restore", json={"version": 1}
    )

    assert restored.status_code == 200, restored.text
    artifact = restored.json()["artifact"]
    assert artifact["current_version"] == 3
    assert artifact["content"] == CODE
    assert [(v["version"], v["origin"]) for v in artifact["versions"]] == [
        (3, "restore"),
        (2, "edit"),
        (1, "generate"),
    ]

    v2 = await user_a.client.get(f"/api/artifacts/{artifact_id}?version=2")
    assert v2.json()["artifact"]["content"] == EDITED


async def test_the_same_artifact_can_be_code_and_then_markdown(user_a):
    thread_id = await new_thread(user_a.client)
    artifact_id = (await create(user_a, thread_id)).json()["artifact"]["id"]

    saved = await user_a.client.post(
        f"/api/artifacts/{artifact_id}/versions",
        json={"kind": "markdown", "content": "# wheel\n\nA wheel."},
    )

    assert saved.json()["artifact"]["kind"] == "markdown"
    # kind is per version: the code version is still code when you go back to it.
    assert (await user_a.client.get(f"/api/artifacts/{artifact_id}?version=1")).json()[
        "artifact"
    ]["kind"] == "code"


async def test_a_bad_kind_is_a_bad_request(user_a):
    thread_id = await new_thread(user_a.client)

    response = await create(user_a, thread_id, kind="pdf")

    assert response.status_code == 400


# --- tenancy ---------------------------------------------------------------


async def test_another_account_cannot_reach_the_artifact(user_a, user_b):
    thread_id = await new_thread(user_a.client)
    artifact_id = (await create(user_a, thread_id)).json()["artifact"]["id"]
    path = f"/api/artifacts/{artifact_id}"

    assert (await user_b.client.get(path)).status_code == 404
    assert (
        await user_b.client.post(path + "/versions", json={"kind": "code", "content": "x"})
    ).status_code == 404
    assert (
        await user_b.client.post(path + "/restore", json={"version": 1})
    ).status_code == 404
    assert (await user_b.client.delete(path)).status_code == 404
    # Even the list of another account's thread: an empty list would read as
    # "this thread has no artifacts", which is a different, wrong answer.
    assert (await user_b.client.get(f"/api/threads/{thread_id}/artifacts")).status_code == 404

    assert (await user_a.client.get(path)).status_code == 200


async def test_the_tab_strip_lists_the_threads_artifacts(user_a):
    thread_id = await new_thread(user_a.client)
    created = (await create(user_a, thread_id)).json()["artifact"]
    await create(user_a, thread_id, title="axle")

    listed = await user_a.client.get(f"/api/threads/{thread_id}/artifacts")

    assert listed.status_code == 200
    rows = listed.json()["artifacts"]
    assert {row["title"] for row in rows} == {"wheel", "axle"}
    # Metadata only: the strip is not the document.
    assert "content" not in rows[0]
    assert {row["versions"] for row in rows} == {1}
    assert created["id"] in {row["id"] for row in rows}


# --- capture and lifecycle -------------------------------------------------


async def test_a_generation_turn_captures_its_model(user_a, stub, workspace):
    thread_id, artifact_id = await generated_code(user_a, stub, workspace)

    body = await user_a.client.get(f"/api/artifacts/{artifact_id}")
    artifact = body.json()["artifact"]
    assert artifact["content"] == CODE
    assert artifact["current_version"] == 1
    assert artifact["versions"][0]["origin"] == "generate"
    assert artifact["title"] == "model a wheel"


async def test_the_stream_announces_the_artifact_before_it_done(user_a, stub, workspace):
    await generated_code(user_a, stub, workspace)
    thread_id = (await user_a.client.get("/api/threads")).json()["threads"][0]["id"]

    streamed = await user_a.client.post(
        f"/api/threads/{thread_id}/chat", json={"message": "model an axle"}
    )

    kinds = event_types(streamed.text)
    assert kinds.index("artifact") < kinds.index("done")
    summary = [e for e in sse_events(streamed.text) if e["type"] == "artifact"][0]["artifact"]
    assert summary["current_version"] == 1
    assert summary["versions"] == 1


async def test_deleting_the_thread_removes_its_artifacts(user_a):
    thread_id = await new_thread(user_a.client)
    artifact_id = (await create(user_a, thread_id)).json()["artifact"]["id"]

    assert (await user_a.client.delete(f"/api/threads/{thread_id}")).status_code == 200

    assert (await user_a.client.get(f"/api/artifacts/{artifact_id}")).status_code == 404


async def test_deleting_the_artifact_leaves_the_turn(user_a, stub, workspace):
    thread_id, artifact_id = await generated_code(user_a, stub, workspace)

    assert (await user_a.client.delete(f"/api/artifacts/{artifact_id}")).status_code == 200

    turn = (await user_a.client.get(f"/api/threads/{thread_id}")).json()["thread"]["messages"][
        -1
    ]
    # set null, not cascade: the conversation is not the artifact's to delete.
    assert "artifact_id" not in turn
    assert turn["model"] == CODE


# --- caps ------------------------------------------------------------------


async def test_content_beyond_the_cap_is_truncated(user_a):
    thread_id = await new_thread(user_a.client)

    artifact = (
        await create(user_a, thread_id, content="x" * (artifacts.MAX_CONTENT_CHARS + 500))
    ).json()["artifact"]

    assert len(artifact["content"]) == artifacts.MAX_CONTENT_CHARS


async def test_the_oldest_version_is_pruned(user_a):
    thread_id = await new_thread(user_a.client)
    artifact_id = (await create(user_a, thread_id)).json()["artifact"]["id"]

    for index in range(artifacts.MAX_VERSIONS):
        saved = await user_a.client.post(
            f"/api/artifacts/{artifact_id}/versions",
            json={"kind": "code", "content": f"edit {index}"},
        )
        assert saved.status_code == 200, saved.text

    final = (await user_a.client.get(f"/api/artifacts/{artifact_id}")).json()["artifact"]
    assert final["current_version"] == artifacts.MAX_VERSIONS + 1
    assert len(final["versions"]) == artifacts.MAX_VERSIONS
    assert final["versions"][-1]["version"] == 2
    assert (await user_a.client.get(f"/api/artifacts/{artifact_id}?version=1")).status_code == 404
