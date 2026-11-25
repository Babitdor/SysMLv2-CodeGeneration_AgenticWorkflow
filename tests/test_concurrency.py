"""Bounded execution: the pool is a ceiling, and every turn still terminates.

The old shape spawned one unbounded thread per request, so a handful of clients
could saturate the box. These tests pin the replacement: an admission gate, a
fixed pool, and a deadline that always ends the stream with an explicit error.
"""

from __future__ import annotations

import asyncio
import time

from support import sse_events

from webapp import server


async def test_parallel_turns_never_exceed_the_pool(monkeypatch, user_a, stub, workspace):
    monkeypatch.setattr(server, "MAX_CONCURRENCY", 2)
    monkeypatch.setattr(server, "MAX_QUEUE", 8)
    stub.route = "chat"
    stub.reply = "one two three"
    stub.delay = 0.15

    responses = await asyncio.gather(
        *[
            user_a.client.post("/api/chat", json={"message": f"turn {index}"})
            for index in range(8)
        ]
    )

    assert [response.status_code for response in responses] == [200] * 8
    for response in responses:
        kinds = [event["type"] for event in sse_events(response.text)]
        assert kinds[-1] == "done"
        assert "token" in kinds
    # The measured high-water mark is the assertion that matters: never more than
    # the configured concurrency, and enough to prove the gate let real work run.
    assert stub.peak == 2


async def test_a_full_queue_answers_busy_instead_of_hanging(monkeypatch, user_a, stub, workspace):
    monkeypatch.setattr(server, "MAX_CONCURRENCY", 1)
    monkeypatch.setattr(server, "MAX_QUEUE", 0)
    monkeypatch.setattr(server, "QUEUE_WAIT", 0.2)
    stub.route = "chat"
    stub.reply = "slow"
    stub.delay = 0.6

    first = asyncio.create_task(user_a.client.post("/api/chat", json={"message": "first"}))
    await asyncio.sleep(0.05)  # let the first turn take the only slot
    second = await user_a.client.post("/api/chat", json={"message": "second"})
    await first

    events = sse_events(second.text)
    assert any(
        event["type"] == "error" and event.get("status") == 503 for event in events
    ), events
    assert events[-1]["type"] == "done"


async def test_a_turn_over_the_deadline_ends_with_an_error(monkeypatch, user_a, stub, workspace):
    monkeypatch.setattr(server, "TURN_TIMEOUT", 1.0)
    stub.route = "generation"
    stub.reply = ""
    stub.code = ""
    stub.hang = 3.0

    created = await user_a.client.post("/api/threads", json={})
    thread_id = created.json()["thread"]["id"]

    started = time.monotonic()
    response = await user_a.client.post(
        f"/api/threads/{thread_id}/chat", json={"message": "model something"}
    )
    elapsed = time.monotonic() - started

    events = sse_events(response.text)
    kinds = [event["type"] for event in events]
    assert any("timed out" in str(event.get("detail", "")) for event in events)
    assert kinds[-1] == "done"
    assert elapsed < 3.0, f"the deadline should fire at ~1s, took {elapsed:.1f}s"


async def test_a_provider_crash_ends_the_turn_and_releases_the_slot(
    monkeypatch, user_a, stub, workspace
):
    """A crashed turn must not leak its gate: the next one still gets through."""
    monkeypatch.setattr(server, "MAX_CONCURRENCY", 1)
    monkeypatch.setattr(server, "MAX_QUEUE", 0)
    stub.route = "generation"

    def explode(agent, prompt, thread_id):
        raise RuntimeError("provider fell over")
        yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(server, "run_sysml_pipeline", explode)
    created = await user_a.client.post("/api/threads", json={})
    thread_id = created.json()["thread"]["id"]

    failed = await user_a.client.post(
        f"/api/threads/{thread_id}/chat", json={"message": "boom"}
    )
    assert any("provider fell over" in str(e.get("detail")) for e in sse_events(failed.text))

    monkeypatch.undo()  # restore the stub pipeline
    monkeypatch.setattr(server, "MAX_CONCURRENCY", 1)
    monkeypatch.setattr(server, "MAX_QUEUE", 0)
    again = await user_a.client.post("/api/chat", json={"message": "again"})

    assert again.status_code == 200
    assert sse_events(again.text)[-1]["type"] == "done"
