"""Money helpers: Decimal in Python, integer cents in the database. Never float."""

from __future__ import annotations

from decimal import Decimal

_CENT = Decimal("0.01")


def to_cents(amount: Decimal) -> int:
    cents = amount * 100
    if cents != cents.to_integral_value():
        raise ValueError(f"amount {amount} has more than 2 decimal places")
    return int(cents)


def from_cents(cents: int) -> Decimal:
    return (Decimal(cents) / 100).quantize(_CENT)
