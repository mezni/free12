"""Token cost estimation from configured prices (never hardcoded)."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from resolve_agent.config import ModelPrice
from resolve_agent.llm.types import Usage


def estimate_cost(model: str, usage: Usage, prices: Mapping[str, ModelPrice]) -> float | None:
    price = prices.get(model)
    if price is None:
        return None
    total = (
        Decimal(usage.input_tokens) * price.input_per_mtok
        + Decimal(usage.output_tokens) * price.output_per_mtok
    ) / Decimal(1_000_000)
    return float(total)
