"""Per-request context injected by the orchestrator. The model can never set these values."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from resolve_agent.config import Settings
from resolve_agent.domain.clock import Clock, system_clock


@dataclass(frozen=True, slots=True)
class ToolContext:
    customer_id: str  # established by the auth layer, never taken from model output
    conn: sqlite3.Connection
    settings: Settings
    clock: Clock = system_clock
    request_id: str = ""
    actor: str = "agent"

    @property
    def today(self) -> date:
        return self.clock().date()
