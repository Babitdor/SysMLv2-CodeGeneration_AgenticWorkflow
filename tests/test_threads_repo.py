"""The thread repository: owner-scoped CRUD, caps and the derived title.

These call the store directly (that is the layer the routes depend on), with a
real account id so the foreign-key and ownership filters are exercised too.
"""

from __future__ import annotations

from webapp import threads


async def test_create_and_load_round_trip(user_a):
    thread = threads.create_thread(user_a.id, "a chat")

    assert thread["title"] == "a chat"
    assert thread["messages"] == []
    loaded = threads.load_thread(user_a.id, thread["id"])
    assert loaded["id"] == thread["id"]
    assert loaded["created_at"] > 0


async def test_the_first_user_turn_seeds_the_title(user_a):
    thread = threads.create_thread(user_a.id)

    threads.append_message(user_a.id, thread["id"], "user", "model a conveyor line")

    assert threads.load_thread(user_a.id, thread["id"])["title"] == "model a conveyor line"


async def test_a_rename_is_collapsed_to_one_line(user_a):
    thread = threads.create_thread(user_a.id)

    renamed = threads.rename_thread(user_a.id, thread["id"], "  **bold**  and\n noisy  ")

    assert renamed["title"] == "bold and noisy"


async def test_delete_removes_the_thread_and_its_turns(user_a):
    thread = threads.create_thread(user_a.id)
    threads.append_message(user_a.id, thread["id"], "user", "hello")

    assert threads.delete_thread(user_a.id, thread["id"]) is True
    assert threads.load_thread(user_a.id, thread["id"]) is None
    assert threads.list_threads(user_a.id) == []
    assert threads.delete_thread(user_a.id, thread["id"]) is False


async def test_junk_and_foreign_ids_are_never_found(user_a, user_b):
    thread = threads.create_thread(user_a.id)
    threads.append_message(user_a.id, thread["id"], "user", "hello")

    for owner in (user_a.id, user_b.id):
        assert threads.load_thread(owner, "not-a-uuid") is None
        assert threads.load_thread(owner, "") is None
    assert threads.load_thread(user_b.id, thread["id"]) is None
    assert threads.rename_thread(user_b.id, thread["id"], "hijacked") is None
    assert threads.delete_thread(user_b.id, thread["id"]) is False
    assert threads.append_message(user_b.id, thread["id"], "user", "injected") is None
    # The owner's copy is untouched.
    mine = threads.load_thread(user_a.id, thread["id"])
    assert [turn["content"] for turn in mine["messages"]] == ["hello"]


async def test_history_drops_turns_without_prose(user_a):
    thread = threads.create_thread(user_a.id)
    threads.append_message(user_a.id, thread["id"], "user", "model a wheel")
    threads.append_message(user_a.id, thread["id"], "assistant", "", model="package A;")

    history = threads.history_messages(threads.load_thread(user_a.id, thread["id"]))

    assert history == [{"role": "user", "content": "model a wheel"}]


async def test_the_turn_cap_keeps_the_newest(user_a, monkeypatch):
    monkeypatch.setattr(threads, "MAX_MESSAGES", 3)
    thread = threads.create_thread(user_a.id)

    for index in range(5):
        threads.append_message(user_a.id, thread["id"], "user", f"turn {index}")

    kept = [turn["content"] for turn in threads.load_thread(user_a.id, thread["id"])["messages"]]
    assert kept == ["turn 2", "turn 3", "turn 4"]


async def test_the_model_is_capped(user_a):
    thread = threads.create_thread(user_a.id)

    threads.append_message(user_a.id, thread["id"], "assistant", "big", model="x" * 50_000)

    stored = threads.load_thread(user_a.id, thread["id"])["messages"][0]
    assert len(stored["model"]) == threads.MAX_MODEL_CHARS


async def test_summaries_are_newest_first_with_message_counts(user_a):
    older = threads.create_thread(user_a.id, "older")
    newer = threads.create_thread(user_a.id, "newer")
    threads.append_message(user_a.id, newer["id"], "user", "one")
    threads.append_message(user_a.id, newer["id"], "assistant", "two")

    rows = threads.list_threads(user_a.id)

    assert [row["id"] for row in rows] == [newer["id"], older["id"]]
    assert [row["message_count"] for row in rows] == [2, 0]
