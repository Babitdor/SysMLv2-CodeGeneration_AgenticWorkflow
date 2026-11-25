"""Per-conversation transcripts, now in Postgres.

A thread belongs to one account: ``owner`` is the user id, and every statement
carries ``where user_id = %s``, so a foreign — or syntactically junk — id reads
as "not found" instead of opening someone else's conversation. Timestamps leave
this module as epoch seconds, which is what the sidebar already renders.

An assistant turn carries the generated model and its validation verdict as
columns, so a code-only turn (no prose) still survives a reload, plus the id of
the canvas artifact the turn produced (`artifact_id`, null once it is deleted).
Those values are capped, as is the number of turns: nothing prunes old
conversations yet, so a single thread cannot be allowed to grow without bound.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from psycopg.types.json import Jsonb

from webapp import db

MAX_MESSAGES = 200  # per-thread cap; nothing prunes old conversations yet
MAX_MESSAGE_CHARS = 40000  # one turn cannot grow the row without bound
MAX_MODEL_CHARS = 40000  # the generated model rides on the turn as its own
# column, so a code-only turn survives a reload instead of living in the prose.
TITLE_CHARS = 60
HISTORY_LIMIT = 16
_NOISE = re.compile(r"[`*_~>#\[\]()!]+")
_DEFAULT_TITLE = "New chat"
_ROLES = ("user", "assistant")


def _uuid(value: Any) -> Optional[str]:
    """Canonical uuid text, or None: ids arrive from the client and the URL."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


def _seconds(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def derive_title(text: str, limit: int = TITLE_CHARS) -> str:
    """One-line title from the first user turn: markdown noise and length gone."""
    flat = " ".join(_NOISE.sub(" ", text or "").split())
    if not flat:
        return _DEFAULT_TITLE
    if len(flat) <= limit:
        return flat
    cut = flat[: max(1, limit - 1)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return (cut.rstrip(" ,.;:-") or flat[: max(1, limit - 1)]) + "…"


def _message(row: Dict[str, Any]) -> Dict[str, Any]:
    turn: Dict[str, Any] = {
        "role": str(row["role"]),
        "content": str(row.get("content") or ""),
        "at": _seconds(row.get("at")),
    }
    model = str(row.get("model") or "")
    if model:
        turn["model"] = model
    validation = row.get("validation")
    if isinstance(validation, dict) and validation:
        turn["validation"] = validation
    # The artifact this turn produced, so the transcript can reopen it in the
    # canvas instead of only showing the model inline. Absent once it is deleted.
    artifact = str(row.get("artifact_id") or "")
    if artifact:
        turn["artifact_id"] = artifact
    return turn


def _thread(row: Dict[str, Any], messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    thread: Dict[str, Any] = {
        "id": str(row["id"]),
        "title": str(row.get("title") or "") or _DEFAULT_TITLE,
        "created_at": _seconds(row.get("created_at")),
        "updated_at": _seconds(row.get("updated_at")),
        "messages": messages or [],
    }
    return thread


def summary(thread: Dict[str, Any]) -> Dict[str, Any]:
    """Sidebar row: everything but the messages."""
    return {
        "id": thread.get("id", ""),
        "title": thread.get("title", _DEFAULT_TITLE),
        "created_at": thread.get("created_at", 0.0),
        "updated_at": thread.get("updated_at", 0.0),
        "message_count": len(thread.get("messages") or []),
    }


_SUMMARY_SQL = (
    "select t.id::text as id, t.title, "
    "extract(epoch from t.created_at) as created_at, "
    "extract(epoch from t.updated_at) as updated_at, "
    "count(m.id) as message_count "
    "from threads t left join thread_messages m on m.thread_id = t.id "
    "where t.user_id = %s group by t.id order by t.updated_at desc"
)


def list_threads(owner: Optional[str]) -> List[Dict[str, Any]]:
    """Newest first, messages omitted (the count is what the row shows)."""
    owner_id = _uuid(owner)
    if not owner_id:
        return []
    with db.connection() as conn:
        rows = conn.execute(_SUMMARY_SQL, (owner_id,)).fetchall()
    return [
        {
            "id": str(row["id"]),
            "title": str(row["title"]) or _DEFAULT_TITLE,
            "created_at": _seconds(row["created_at"]),
            "updated_at": _seconds(row["updated_at"]),
            "message_count": int(row["message_count"] or 0),
        }
        for row in rows
    ]


def create_thread(owner: Optional[str], title: str = "") -> Optional[Dict[str, Any]]:
    """A new empty thread, or None when there is no signed-in account."""
    owner_id = _uuid(owner)
    if not owner_id:
        return None
    heading = (title or "").strip()[:TITLE_CHARS] or _DEFAULT_TITLE
    with db.connection() as conn:
        row = conn.execute(
            "insert into threads (user_id, title) values (%s, %s) "
            "returning id::text as id, title, "
            "extract(epoch from created_at) as created_at, "
            "extract(epoch from updated_at) as updated_at",
            (owner_id, heading),
        ).fetchone()
    return _thread(row)


def load_thread(owner: Optional[str], thread_id: str) -> Optional[Dict[str, Any]]:
    """One thread with its turns, or None: the only way a route reaches a thread."""
    owner_id, wanted = _uuid(owner), _uuid(thread_id)
    if not owner_id or not wanted:
        return None
    with db.connection() as conn:
        row = conn.execute(
            "select id::text as id, title, "
            "extract(epoch from created_at) as created_at, "
            "extract(epoch from updated_at) as updated_at "
            "from threads where id = %s and user_id = %s",
            (wanted, owner_id),
        ).fetchone()
        if row is None:
            return None
        turns = conn.execute(
            "select role, content, model, validation, "
            "artifact_id::text as artifact_id, "
            "extract(epoch from created_at) as at "
            "from thread_messages where thread_id = %s order by id",
            (wanted,),
        ).fetchall()
    return _thread(row, [_message(turn) for turn in turns])


def rename_thread(
    owner: Optional[str], thread_id: str, title: str
) -> Optional[Dict[str, Any]]:
    owner_id, wanted = _uuid(owner), _uuid(thread_id)
    if not owner_id or not wanted:
        return None
    with db.connection() as conn:
        row = conn.execute(
            "update threads set title = %s, updated_at = now() "
            "where id = %s and user_id = %s returning 1",
            (derive_title(title, TITLE_CHARS), wanted, owner_id),
        ).fetchone()
    if row is None:
        return None
    return load_thread(owner, thread_id)


def delete_thread(owner: Optional[str], thread_id: str) -> bool:
    owner_id, wanted = _uuid(owner), _uuid(thread_id)
    if not owner_id or not wanted:
        return False
    with db.connection() as conn:
        row = conn.execute(
            "delete from threads where id = %s and user_id = %s returning 1",
            (wanted, owner_id),
        ).fetchone()
    return row is not None


def append_message(
    owner: Optional[str],
    thread_id: str,
    role: str,
    content: str,
    model: str = "",
    validation: Optional[Dict[str, Any]] = None,
    artifact_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Add one turn; the first user turn seeds the title.

    Returns the stored turn, or None when the thread is not this owner's or the
    turn carries nothing (the old file store's two silent-reject cases).
    """
    role = str(role or "").strip().lower()
    text = (content or "").strip()
    code = (model or "").strip()
    if role not in _ROLES or not (text or code):
        return None
    owner_id, wanted = _uuid(owner), _uuid(thread_id)
    if not owner_id or not wanted:
        return None
    verdict = validation if isinstance(validation, dict) and validation else None
    canvas = _uuid(artifact_id)

    with db.connection() as conn:
        with conn.transaction():
            # Owning the thread is the precondition for touching its messages, so
            # this update doubles as the authorisation check.
            row = conn.execute(
                "update threads set updated_at = now() "
                "where id = %s and user_id = %s returning title",
                (wanted, owner_id),
            ).fetchone()
            if row is None:
                return None
            current = str(row["title"] or "")
            if role == "user" and current in ("", _DEFAULT_TITLE):
                conn.execute(
                    "update threads set title = %s where id = %s",
                    (derive_title(text), wanted),
                )
            stored = conn.execute(
                "insert into thread_messages "
                "(thread_id, role, content, model, validation, artifact_id) "
                "values (%s, %s, %s, %s, %s, %s) "
                "returning role, content, model, validation, "
                "artifact_id::text as artifact_id, "
                "extract(epoch from created_at) as at",
                (
                    wanted,
                    role,
                    text[:MAX_MESSAGE_CHARS],
                    code[:MAX_MODEL_CHARS],
                    Jsonb(verdict) if verdict else None,
                    canvas,
                ),
            ).fetchone()
            # Keep the newest MAX_MESSAGES turns; the thread's history is bounded.
            conn.execute(
                "delete from thread_messages where thread_id = %s and id not in "
                "(select id from thread_messages where thread_id = %s "
                "order by id desc limit %s)",
                (wanted, wanted, MAX_MESSAGES),
            )
    return _message(stored)


def history_messages(
    thread: Dict[str, Any], limit: int = HISTORY_LIMIT
) -> List[Dict[str, str]]:
    """Last `limit` turns in the shape the providers expect."""
    turns = (thread.get("messages") or [])[-max(1, limit) :]
    # A code-only turn adds no prose, and an empty message can be rejected by
    # a provider: the model reaches the next turn through `artifact` instead.
    return [
        {"role": turn["role"], "content": turn["content"]}
        for turn in turns
        if str(turn.get("content", "")).strip()
    ]
