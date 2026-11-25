"""Cross-process serialisation for the shared Chroma stores.

Chroma's SQLite files are not safe for a second writer, and "gunicorn -w 2" means
two processes will try. flock is POSIX only, so this test runs in the Linux
container / CI and skips on a Windows dev box — where the lock is a documented
no-op and concurrency is only serialised inside one process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from webapp import locks

try:  # the module degrades to a no-op where fcntl is missing
    import fcntl  # noqa: F401

    HAS_FLOCK = True
except ImportError:  # pragma: no cover - Windows
    HAS_FLOCK = False

pytestmark = pytest.mark.skipif(not HAS_FLOCK, reason="flock needs POSIX")

ROOT = Path(__file__).resolve().parent.parent

HELPER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    sys.path.insert(0, {root!r})
    from webapp import locks

    name, log, lock = sys.argv[1:4]
    with locks.flock(Path(lock)):
        with open(log, "a", encoding="utf-8") as handle:
            handle.write("start " + name + chr(10))
            handle.flush()
            time.sleep(0.25)
            handle.write("end " + name + chr(10))
            handle.flush()
    """
)


def _spawn(name: str, log: Path, lock: Path) -> subprocess.Popen:
    script = HELPER.format(root=str(ROOT))
    return subprocess.Popen(
        [sys.executable, "-c", script, name, str(log), str(lock)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_two_processes_never_cross_the_critical_section(tmp_path):
    log = tmp_path / "order.log"
    lock = tmp_path / "chroma.lock"

    first = _spawn("a", log, lock)
    second = _spawn("b", log, lock)
    for process in (first, second):
        _, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr.decode("utf-8", "replace")

    # Four lines, two tokens each ("start a" / "end a" ...).
    lines = log.read_text(encoding="utf-8").split()
    assert len(lines) == 8, lines
    # Either order is fine; interleaving is not.
    assert lines in (
        ["start", "a", "end", "a", "start", "b", "end", "b"],
        ["start", "b", "end", "b", "start", "a", "end", "a"],
    )


def test_the_lock_is_released_when_the_block_exits(tmp_path):
    lock = tmp_path / "chroma.lock"

    with locks.flock(lock):
        assert lock.is_file()
    with locks.flock(lock):  # would deadlock if the first block had leaked it
        pass


def test_an_unwritable_location_does_not_break_the_turn(tmp_path):
    """A read-only state dir must degrade to "no lock", not to a 500."""
    with locks.flock(tmp_path / "missing-parent" / "chroma.lock"):
        pass
    assert os.path.isdir(tmp_path)
