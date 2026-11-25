"""Postgres and Redis access for the multi-tenant app.

Everything that used to be a process-global dict or a JSON file on the
workspace volume now lives in Postgres — users, sessions, threads, their
messages, uploaded documents and the deployment-wide backend settings — so
every worker sees the same state and a restart cannot lose it. Redis holds the
two things that must be shared but short-lived: rate-limit counters and each
user's last validated model.

Clients are built lazily and cached per process: importing this module never
opens a socket, which keeps the migration CLI and the unit tests cheap. A test
that needs a clean process calls :func:`close_clients` in its teardown, or the
pool's threads outlive the test that made them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

# Migrations are plain .sql files applied in name order; the recorded name is
# the file name, so renaming one re-runs it — treat these as immutable.
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# Session-level advisory lock: two workers starting at once (gunicorn -w N)
# must not try to create the same table twice. The value is arbitrary but has
# to stay stable across releases, or the lock stops being shared.
MIGRATION_LOCK_KEY = 8_274_119_004

DEFAULT_POOL_SIZE = 8

_pool: Any = None
_redis: Any = None


def database_url() -> str:
    """DATABASE_URL, or a message an operator can act on."""
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Accounts, threads and documents live in "
            "Postgres now; copy .env.example to .env and start the stack."
        )
    return url


def redis_url() -> str:
    url = (os.environ.get("REDIS_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "REDIS_URL is not set. Rate limits and the per-user model cache use "
            "Redis; copy .env.example to .env and start the stack."
        )
    return url


def pool() -> Any:
    """The process-wide connection pool, opened on first use."""
    global _pool
    if _pool is None:
        # Imported here so a deployment without Postgres (the migration CLI on a
        # bare checkout, say) still loads the module for its docstrings/constants.
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(
            database_url(),
            min_size=1,
            max_size=int(os.environ.get("SYSML_DB_POOL_SIZE") or DEFAULT_POOL_SIZE),
            open=True,
            # Autocommit keeps the everyday read/insert a single statement, so a
            # dropped request never leaves a transaction open. Anything that must
            # be atomic uses an explicit `with conn.transaction()` block.
            kwargs={"row_factory": dict_row, "autocommit": True},
        )
    return _pool


def connection() -> Any:
    """One pooled connection as a context manager (returns it on exit)."""
    return pool().connection()


def redis_client() -> Any:
    """The process-wide Redis client, lazily built with short timeouts.

    The timeouts are deliberately tight: a wedged Redis must degrade the rate
    limiter into "fail open", not hold every request for the default 60s.
    """
    global _redis
    if _redis is None:
        import redis

        _redis = redis.Redis.from_url(
            redis_url(),
            decode_responses=True,
            socket_timeout=float(os.environ.get("SYSML_REDIS_TIMEOUT") or 2.0),
            socket_connect_timeout=float(os.environ.get("SYSML_REDIS_TIMEOUT") or 2.0),
        )
    return _redis


def close_clients() -> None:
    """Drop the cached pool/client (tests, and a clean shutdown)."""
    global _pool, _redis
    pool_handles, _pool = (_pool, None)
    if pool_handles is not None:
        try:
            pool_handles.close()
        except Exception:
            pass
    client, _redis = (_redis, None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def _migration_files() -> List[Path]:
    if not MIGRATIONS_DIR.is_dir():
        return []
    return sorted(MIGRATIONS_DIR.glob("*.sql"), key=lambda path: path.name)


def applied_migrations() -> List[str]:
    """Names recorded in schema_migrations, oldest first."""
    with connection() as conn:
        rows = conn.execute(
            "select name from schema_migrations order by name"
        ).fetchall()
    return [str(row["name"]) for row in rows]


def apply_migrations() -> List[str]:
    """Apply every pending .sql file once, under an advisory lock.

    Returns the migrations this call applied (empty when the schema is already
    current). Each file runs in its own transaction together with its
    schema_migrations row, so a failed migration leaves neither half behind.
    """
    applied_now: List[str] = []
    with connection() as conn:
        # Session-level lock, taken on this pooled connection and released
        # explicitly: the migration itself needs its own transaction.
        conn.execute("select pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
        try:
            conn.execute(
                "create table if not exists schema_migrations ("
                "name text primary key, "
                "applied_at timestamptz not null default now())"
            )
            done = {
                str(row["name"])
                for row in conn.execute("select name from schema_migrations").fetchall()
            }
            for path in _migration_files():
                if path.name in done:
                    continue
                sql = path.read_text(encoding="utf-8")
                with conn.transaction():
                    # No parameters -> simple query protocol, so one file may
                    # hold several statements.
                    conn.execute(sql)
                    conn.execute(
                        "insert into schema_migrations (name) values (%s)",
                        (path.name,),
                    )
                applied_now.append(path.name)
        finally:
            conn.execute("select pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
    return applied_now


def error_status(check: Any) -> Optional[str]:
    """Run a readiness check; None when it is fine, else the reason."""
    try:
        check()
    except Exception as exc:  # connection refused, auth failure, timeout ...
        return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return None


def postgres_status() -> Optional[str]:
    """None when a trivial query answers, else why it did not."""

    def probe() -> None:
        with connection() as conn:
            conn.execute("select 1").fetchone()

    return error_status(probe)


def redis_status() -> Optional[str]:
    def probe() -> None:
        if redis_client().ping() is not True:
            raise RuntimeError("unexpected PING reply")

    return error_status(probe)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m webapp.db",
        description="Apply the Postgres migrations and report what is recorded.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="apply pending migrations, then print the applied names",
    )
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 2
    try:
        fresh = apply_migrations()
        if fresh:
            print(f"applied: {', '.join(fresh)}")
        for name in applied_migrations():
            print(name)
    finally:
        # The pool's worker threads are not daemons: leaving them to interpreter
        # shutdown makes the CLI hang ~5s and print a "couldn't stop thread" hint.
        close_clients()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
