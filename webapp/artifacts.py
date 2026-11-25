"""Versioned artifacts: the right-hand canvas, now in Postgres.

An artifact is one named document owned by a thread. Its history lives in
`artifact_versions`, which is append-only: rows are never updated or rewritten,
an edit appends v(N+1), and a restore appends a *new* version whose content is
copied from the old one. Going back in time therefore cannot destroy a version,
and the same artifact can be code in one version and markdown in the next without
losing the record of which was which.

Ownership works the way it does for threads: every statement carries
`user_id = %s`, and creating an artifact proves ownership of the parent thread,
so a foreign or junk id reads as "not found" rather than opening someone else's
document. Growth is bounded per artifact — `MAX_CONTENT_CHARS` per version and
`MAX_VERSIONS` versions kept, pruned oldest-first — because nothing prunes idle
artifacts across threads yet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from webapp import db, threads

MAX_CONTENT_CHARS = 40000  # parity with threads.MAX_MODEL_CHARS: one version
MAX_VERSIONS = 50  # newest kept; 50 * 40k is the worst case per artifact
TITLE_CHARS = 60
KINDS = ("code", "markdown")
ORIGINS = ("generate", "edit", "restore")
_UNTITLED = "untitled"


def _uuid(value: Any) -> Optional[str]:
    """Canonical uuid text, or None: ids arrive from the client and the URL."""
    return threads._uuid(value)


def _seconds(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _version_no(value: Any) -> Optional[int]:
    """A positive version number, or None for anything that is not one."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _content(value: Any) -> str:
    return str(value or "")[:MAX_CONTENT_CHARS]


def _title(value: Any) -> str:
    """One-line title: the thread-title cleaning, with its own empty default."""
    text = " ".join(str(value or "").split())
    return threads.derive_title(text, TITLE_CHARS) if text else _UNTITLED


def _kind(value: Any) -> str:
    kind = str(value or "").strip().lower()
    return kind if kind in KINDS else KINDS[0]


def _origin(value: Any) -> str:
    origin = str(value or "").strip().lower()
    return origin if origin in ORIGINS else "edit"


def summary(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Tab-strip shape: metadata only, because the strip is not the document."""
    return {
        "id": artifact.get("id", ""),
        "title": artifact.get("title", "") or _UNTITLED,
        "kind": artifact.get("kind", KINDS[0]),
        "current_version": int(artifact.get("current_version") or 1),
        "versions": len(artifact.get("versions") or []),
        "updated_at": _seconds(artifact.get("updated_at")),
    }


def list_artifacts(owner: Optional[str], thread_id: str) -> List[Dict[str, Any]]:
    """The thread's artifacts, newest first; no content, the strip is cheap."""
    owner_id, parent = _uuid(owner), _uuid(thread_id)
    if not owner_id or not parent:
        return []
    with db.connection() as conn:
        rows = conn.execute(
            "select a.id::text as id, a.title, a.kind, a.current_version, "
            "count(v.id) as versions, "
            "extract(epoch from a.updated_at) as updated_at "
            "from artifacts a "
            "left join artifact_versions v on v.artifact_id = a.id "
            "where a.user_id = %s and a.thread_id = %s "
            "group by a.id order by a.updated_at desc",
            (owner_id, parent),
        ).fetchall()
    return [
        {
            "id": str(row["id"]),
            "title": str(row["title"] or "") or _UNTITLED,
            "kind": str(row["kind"] or KINDS[0]),
            "current_version": int(row["current_version"] or 1),
            "versions": int(row["versions"] or 0),
            "updated_at": _seconds(row["updated_at"]),
        }
        for row in rows
    ]


def load_artifact(
    owner: Optional[str], artifact_id: str, version: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """One artifact: its version rail plus the content of the version asked for.

    `version` defaults to the artifact's current version, so the caller that has
    no opinion gets the latest text. A pruned or never-existent version is None,
    the same answer as a foreign id.
    """
    owner_id, wanted = _uuid(owner), _uuid(artifact_id)
    if not owner_id or not wanted:
        return None
    with db.connection() as conn:
        row = conn.execute(
            "select id::text as id, thread_id::text as thread_id, title, "
            "current_version, extract(epoch from updated_at) as updated_at "
            "from artifacts where id = %s and user_id = %s",
            (wanted, owner_id),
        ).fetchone()
        if row is None:
            return None
        chosen = row["current_version"] if version is None else _version_no(version)
        if chosen is None:
            return None
        history = conn.execute(
            "select version, kind, origin, length(content) as chars, "
            "extract(epoch from created_at) as at "
            "from artifact_versions where artifact_id = %s order by version desc",
            (wanted,),
        ).fetchall()
        current = conn.execute(
            "select version, kind, content from artifact_versions "
            "where artifact_id = %s and version = %s",
            (wanted, chosen),
        ).fetchone()
    if current is None:
        return None
    return {
        "id": str(row["id"]),
        "thread_id": str(row["thread_id"]),
        "title": str(row["title"] or "") or _UNTITLED,
        "current_version": int(row["current_version"] or 1),
        "updated_at": _seconds(row["updated_at"]),
        "versions": [
            {
                "version": int(item["version"]),
                "kind": str(item["kind"]),
                "origin": str(item["origin"]),
                "chars": int(item["chars"] or 0),
                "at": _seconds(item["at"]),
            }
            for item in history
        ],
        # The version's own kind, not the artifact's: reading v1 as code while
        # v2 is markdown is the whole point of keeping kind per version.
        "version": int(current["version"]),
        "kind": str(current["kind"]),
        "content": str(current["content"] or ""),
    }


def create_artifact(
    owner: Optional[str],
    thread_id: str,
    title: str = "",
    kind: str = KINDS[0],
    content: str = "",
    origin: str = "generate",
) -> Optional[Dict[str, Any]]:
    """Insert an artifact and its v1 in one transaction, then return it loaded."""
    owner_id, parent = _uuid(owner), _uuid(thread_id)
    if not owner_id or not parent:
        return None
    kind = _kind(kind)
    with db.connection() as conn:
        with conn.transaction():
            # Owning the thread is the precondition for attaching an artifact to
            # it, so this select doubles as the authorisation check.
            owned = conn.execute(
                "select 1 from threads where id = %s and user_id = %s",
                (parent, owner_id),
            ).fetchone()
            if owned is None:
                return None
            row = conn.execute(
                "insert into artifacts (user_id, thread_id, title, kind) "
                "values (%s, %s, %s, %s) returning id::text as id",
                (owner_id, parent, _title(title), kind),
            ).fetchone()
            conn.execute(
                "insert into artifact_versions (artifact_id, version, kind, content, origin) "
                "values (%s, 1, %s, %s, %s)",
                (row["id"], kind, _content(content), _origin(origin)),
            )
    return load_artifact(owner, row["id"])


def add_version(
    owner: Optional[str],
    artifact_id: str,
    kind: str = KINDS[0],
    content: str = "",
    origin: str = "edit",
) -> Optional[Dict[str, Any]]:
    """Append v(N+1), move the pointer, prune the tail.

    A restore is the same append with `origin="restore"`: time travel never
    overwrites a version.
    """
    owner_id, wanted = _uuid(owner), _uuid(artifact_id)
    if not owner_id or not wanted:
        return None
    kind = _kind(kind)
    with db.connection() as conn:
        with conn.transaction():
            # The row lock is what keeps the next version unique: two saves would
            # otherwise read the same max(version) and collide. The unique
            # constraint behind it is the backstop, not the plan.
            row = conn.execute(
                "select current_version from artifacts "
                "where id = %s and user_id = %s for update",
                (wanted, owner_id),
            ).fetchone()
            if row is None:
                return None
            next_version = conn.execute(
                "select coalesce(max(version), 0) + 1 as next from artifact_versions "
                "where artifact_id = %s",
                (wanted,),
            ).fetchone()["next"]
            conn.execute(
                "insert into artifact_versions (artifact_id, version, kind, content, origin) "
                "values (%s, %s, %s, %s, %s)",
                (wanted, int(next_version), kind, _content(content), _origin(origin)),
            )
            conn.execute(
                "update artifacts set kind = %s, current_version = %s, updated_at = now() "
                "where id = %s",
                (kind, int(next_version), wanted),
            )
            # Keep the newest MAX_VERSIONS rows; the artifact's history is bounded.
            conn.execute(
                "delete from artifact_versions where artifact_id = %s and version not in "
                "(select version from artifact_versions where artifact_id = %s "
                "order by version desc limit %s)",
                (wanted, wanted, MAX_VERSIONS),
            )
    return load_artifact(owner, artifact_id)


def delete_artifact(owner: Optional[str], artifact_id: str) -> bool:
    """True when a row of this owner's was removed; the versions cascade."""
    owner_id, wanted = _uuid(owner), _uuid(artifact_id)
    if not owner_id or not wanted:
        return False
    with db.connection() as conn:
        row = conn.execute(
            "delete from artifacts where id = %s and user_id = %s returning 1",
            (wanted, owner_id),
        ).fetchone()
    return row is not None
