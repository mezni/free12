"""SQLite connection, schema initialisation and transaction helpers (stdlib only)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from resolve_agent.domain.schema import SCHEMA_SQL, SCHEMA_VERSION


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection in autocommit mode; transactions are explicit (see `transaction`)."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version == SCHEMA_VERSION:
        return
    if version != 0:
        raise RuntimeError(f"unsupported schema version {version}, expected {SCHEMA_VERSION}")
    with transaction(conn):
        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, rolling back on any exception. Not re-entrant."""
    if conn.in_transaction:
        raise RuntimeError("transaction already open; use atomic() to join it")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


@contextmanager
def atomic(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Join the caller's transaction if one is open, otherwise start a new one."""
    if conn.in_transaction:
        yield conn
    else:
        with transaction(conn):
            yield conn
