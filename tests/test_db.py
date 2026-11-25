"""Migrations: applied once, recorded once, and idempotent under a second run."""

from __future__ import annotations

import pytest

from webapp import db

EXPECTED = ["0001_init.sql", "0002_artifacts.sql", "0003_user_provider_config.sql"]


def test_first_migration_is_recorded(migrated):
    assert db.applied_migrations() == EXPECTED


def test_a_second_run_applies_nothing(migrated):
    # gunicorn -w N boots several workers against the same database; the second
    # one must find the schema done and change nothing.
    assert db.apply_migrations() == []
    with db.connection() as conn:
        rows = conn.execute("select name from schema_migrations").fetchall()
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
