"""Migrations: applied once, recorded once, and idempotent under a second run."""

from __future__ import annotations

import pytest

from webapp import db

EXPECTED = [
    "0001_init.sql",
    "0002_artifacts.sql",
    "0003_user_provider_config.sql",
    "0004_turn_tool_trace.sql",
    "0005_user_service_credentials.sql",
]


def test_first_migration_is_recorded(migrated):
    assert db.applied_migrations() == EXPECTED


def test_provider_credential_migration_keeps_existing_ciphertext():
    owner = "1f13fe73-6de0-4503-846f-e556aba247e0"
    with db.connection() as conn, conn.transaction(force_rollback=True):
        conn.execute("create schema credential_migration_test")
        conn.execute("set local search_path to credential_migration_test")
        conn.execute("create table users (id uuid primary key)")
        conn.execute((db.MIGRATIONS_DIR / EXPECTED[2]).read_text(encoding="utf-8"))
        conn.execute("insert into users (id) values (%s)", (owner,))
        conn.execute(
            "insert into user_provider_config (user_id, provider, api_key_cipher) values (%s, %s, %s)",
            (owner, "anthropic", "existing-ciphertext"),
        )
        conn.execute((db.MIGRATIONS_DIR / EXPECTED[4]).read_text(encoding="utf-8"))
        row = conn.execute(
            "select user_id::text, service, key_cipher from user_service_credentials",
        ).fetchone()
        assert row == {"user_id": owner, "service": "anthropic", "key_cipher": "existing-ciphertext"}


def test_a_second_run_applies_nothing(migrated):
    # gunicorn -w N boots several workers against the same database; the second
    # one must find the schema done and change nothing.
    assert db.apply_migrations() == []
    with db.connection() as conn:
        rows = conn.execute("select name from schema_migrations order by name").fetchall()
    assert [row["name"] for row in rows] == EXPECTED


def test_every_tenant_table_is_keyed_by_owner(migrated):
    with db.connection() as conn:
        rows = conn.execute(
            "select table_name from information_schema.columns "
            "where table_schema = 'public' and column_name = 'user_id'"
        ).fetchall()
    assert {row["table_name"] for row in rows} >= {
        "sessions",
        "threads",
        "documents",
    }


def test_clients_report_healthy(migrated):
    assert db.postgres_status() is None
    assert db.redis_status() is None


def test_a_missing_database_url_says_what_to_do(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db.database_url()


def test_a_broken_component_is_named(monkeypatch):
    def explode() -> None:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(db, "postgres_status", lambda: db.error_status(explode))
    assert "connection refused" in db.postgres_status()
