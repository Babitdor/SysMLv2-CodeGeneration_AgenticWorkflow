"""Import the pre-accounts state directory into Postgres.

Before accounts existed, threads were JSON files keyed by an anonymous cookie:

    <state>/threads/<cookie>/<thread-id>.json

Those files hold real conversations, and nothing in the app reads them any more.
This walks them and re-homes them under one account so an operator can keep them.

    python scripts/import_state.py --state-dir /workspace/state --user-email me@example.com --dry-run
    python scripts/import_state.py --state-dir /workspace/state --user-email me@example.com

Nothing is deleted: the files stay where they are, so a bad run can be undone by
deleting the imported threads (they are listed by id).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp import auth, db, threads  # noqa: E402

MAX_MESSAGES = threads.MAX_MESSAGES


class AccountMissing(Exception):
    """No such account: the importer never creates one implicitly."""


def account_id(email: str) -> str:
    with db.connection() as conn:
        row = conn.execute(
            "select id::text as id from users where lower(email) = %s",
            (auth.normalize_email(email),),
        ).fetchone()
    if row is None:
        raise AccountMissing(
            f"no account for {email!r}: register it in the UI first "
            "(the importer never creates accounts)"
        )
    return str(row["id"])


def legacy_files(state_dir: Path) -> List[Path]:
    root = state_dir / "threads"
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("*/*.json") if path.is_file())


def read_thread(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    return data


def import_thread(owner: str, data: Dict[str, Any], dry_run: bool) -> Optional[str]:
    """Create the thread and replay its turns; returns the new thread id."""
    title = str(data.get("title") or "").strip()
    if dry_run:
        return f"(dry run) {title!r}: {len(data.get('messages') or [])} turns"

    thread = threads.create_thread(owner, title)
    if thread is None:
        return None
    for turn in (data.get("messages") or [])[:MAX_MESSAGES]:
        if not isinstance(turn, dict):
            continue
        threads.append_message(
            owner,
            thread["id"],
            str(turn.get("role") or ""),
            str(turn.get("content") or ""),
            model=str(turn.get("model") or ""),
            validation=turn.get("validation") if isinstance(turn.get("validation"), dict) else None,
        )
    return thread["id"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        default="/workspace/state",
        help="the volume that holds the old threads/ directory",
    )
    parser.add_argument("--user-email", required=True, help="account to re-home them under")
    parser.add_argument("--dry-run", action="store_true", help="list what would be imported")
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N threads (0 = all)"
    )
    args = parser.parse_args(argv)

    try:
        owner = account_id(args.user_email)
    except AccountMissing as exc:
        db.close_clients()
        print(f"import: {exc}", file=sys.stderr)
        return 2

    files = legacy_files(Path(args.state_dir))
    if not files:
        db.close_clients()
        print(f"import: nothing to do ({Path(args.state_dir) / 'threads'} is empty)")
        return 0
    if args.limit:
        files = files[: args.limit]

    db.apply_migrations()

    try:
        return _import_all(files, Path(args.state_dir), owner, args.dry_run)
    finally:
        # The pool's worker threads are not daemons: leaving them to interpreter
        # shutdown makes the script hang ~5s and print a "couldn't stop thread" hint.
        db.close_clients()


def _import_all(files: List[Path], state_dir: Path, owner: str, dry_run: bool) -> int:
    imported = skipped = 0
    verb = "would import" if dry_run else "imported"
    for path in files:
        relative = path.relative_to(state_dir)
        data = read_thread(path)
        if data is None:
            skipped += 1
            print(f"  skip  {relative}: no readable turns")
            continue
        result = import_thread(owner, data, dry_run)
        if result is None:
            skipped += 1
            print(f"  skip  {relative}: could not store")
            continue
        imported += 1
        print(f"  {verb}  {result}  ({path.parent.name})")

    print(f"import: {verb} {imported} thread(s), skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
