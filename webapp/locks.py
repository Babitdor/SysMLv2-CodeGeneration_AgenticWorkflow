"""Cross-process advisory lock for the shared Chroma stores.

The RAG store and the approved-solutions store are SQLite databases on a shared
volume. Chroma is not safe for two writers, and with ``gunicorn -w N`` several
worker processes will try: an approval on worker A while worker B opens the same
store to answer a search. An ``flock()`` around open/search/approve serialises
them without a new service.

This is the documented scale-out boundary — it works on one host with one shared
volume, and a dedicated RAG service is the follow-up. ``fcntl`` is POSIX only, so
on Windows the lock degrades to a no-op rather than breaking local development.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator, Optional

STATE_DIR = Path(os.environ.get("SYSML_STATE_DIR", "/workspace/state"))
LOCK_PATH = STATE_DIR / "chroma.lock"


@contextlib.contextmanager
def flock(path: Optional[Path] = None) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``path`` for the block's duration.

    Yields immediately (unlocked) where fcntl is unavailable — Windows — so the
    module imports and the app runs there, with the caveat that concurrency is
    only serialised inside one process.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return

    target = Path(path) if path is not None else LOCK_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(str(target), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        # An unwritable state dir must not turn every search into a 500.
        yield
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)
