from decimal import Decimal

import pytest

from resolve_agent.domain.money import from_cents, to_cents


def test_round_trip() -> None:
    assert to_cents(Decimal("49.99")) == 4999
    assert from_cents(4999) == Decimal("49.99")
    assert from_cents(5000) == Decimal("50.00")


def test_no_float_drift() -> None:
    assert to_cents(Decimal("0.10") + Decimal("0.20")) == 30


def test_rejects_sub_cent_amounts() -> None:
    with pytest.raises(ValueError, match="decimal places"):
        to_cents(Decimal("49.999"))
