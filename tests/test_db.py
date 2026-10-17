import sqlite3

import pytest

from resolve_agent.domain.db import atomic, init_schema, transaction
from resolve_agent.domain.schema import SCHEMA_VERSION


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def test_init_schema_sets_version_and_is_idempotent(conn: sqlite3.Connection) -> None:
    assert _user_version(conn) == SCHEMA_VERSION
    init_schema(conn)
    assert _user_version(conn) == SCHEMA_VERSION


def test_unsupported_schema_version_is_rejected(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="unsupported schema version"):
        init_schema(conn)


def test_foreign_keys_are_enforced(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO orders (id, customer_id, status, placed_on, total_cents)"
            " VALUES ('o1', 'nobody', 'placed', '2026-01-01', 100)"
        )


def test_delivered_order_requires_delivery_date(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO customers VALUES ('c1', 'c1@example.com', 'C One')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO orders (id, customer_id, status, placed_on, total_cents)"
            " VALUES ('o1', 'c1', 'delivered', '2026-01-01', 100)"
        )


def test_transaction_rolls_back_on_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(ZeroDivisionError), transaction(conn):
        conn.execute("INSERT INTO customers VALUES ('c1', 'c1@example.com', 'C One')")
        raise ZeroDivisionError
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0


def test_atomic_joins_an_open_transaction(conn: sqlite3.Connection) -> None:
    with pytest.raises(ZeroDivisionError), transaction(conn):
        with atomic(conn):
            conn.execute("INSERT INTO customers VALUES ('c1', 'c1@example.com', 'C One')")
        raise ZeroDivisionError
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0
