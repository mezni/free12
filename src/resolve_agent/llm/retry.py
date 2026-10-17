"""Retry policy: exponential backoff with jitter, bounded by a total time budget.

The budget matters as much as the attempt count: with a p95 latency target, a third
retry after two slow failures is worse than an honest error and a human handoff.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from resolve_agent.config import LLMSettings


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 4.0
    total_budget_s: float = 15.0

    @classmethod
    def from_settings(cls, llm: LLMSettings) -> RetryPolicy:
        return cls(max_attempts=llm.max_retries + 1, total_budget_s=llm.total_budget_s)

    def delay(self, attempt: int, retry_after: float | None, rng: random.Random) -> float:
        """Seconds to wait after the given (1-based) failed attempt."""
        ceiling = min(self.max_delay_s, self.base_delay_s * 2.0 ** (attempt - 1))
        delay = ceiling / 2 + rng.random() * ceiling / 2  # "equal jitter": never below half
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.max_delay_s))
        return delay
