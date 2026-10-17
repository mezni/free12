import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest

from resolve_agent.config import Settings
from resolve_agent.domain.clock import Clock
from resolve_agent.domain.db import connect, init_schema
from resolve_agent.domain.seed import seed
from resolve_agent.obs import InMemoryExporter, Metrics, Tracer, set_metrics, set_tracer
from resolve_agent.tools import ToolContext

TODAY = date(2026, 10, 16)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:")
    init_schema(c)
    yield c
    c.close()


@pytest.fixture
def seeded(conn: sqlite3.Connection) -> sqlite3.Connection:
    seed(conn, TODAY)
    return conn


@pytest.fixture
def clock() -> Clock:
    return lambda: datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


@pytest.fixture
def ctx(seeded: sqlite3.Connection, clock: Clock) -> ToolContext:
    """Tool context for the authenticated customer Alice."""
    return ToolContext(
        customer_id="cus_alice",
        conn=seeded,
        settings=Settings.from_env({}),
        clock=clock,
        request_id="req_test",
    )


@pytest.fixture(autouse=True)
def _reset_observability() -> Iterator[None]:
    """Global tracer and metrics must never leak between tests."""
    yield
    set_tracer(Tracer())
    set_metrics(Metrics())


@pytest.fixture
def spans() -> InMemoryExporter:
    """Install a tracer that keeps finished spans in memory."""
    exporter = InMemoryExporter()
    set_tracer(Tracer([exporter]))
    return exporter
