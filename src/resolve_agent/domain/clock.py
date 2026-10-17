"""Injectable clock so time-dependent code is testable."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

Clock = Callable[[], datetime]


def system_clock() -> datetime:
    return datetime.now(UTC)
