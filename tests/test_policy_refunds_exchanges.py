from decimal import Decimal

import pytest

from factories import TODAY, make_item, make_order
from resolve_agent.config import PolicySettings
from resolve_agent.domain.models import ItemCondition
from resolve_agent.policy import Reason, check_exchange_eligibility, check_refund

SETTINGS = PolicySettings()
ZERO = Decimal("0")


def _refund(amount: str, item_total: str = "500.00", already: Decimal = ZERO):  # type: ignore[no-untyped-def]
    return check_refund(
        Decimal(amount), item_total=Decimal(item_total), already_refunded=already, settings=SETTINGS
    )


@pytest.mark.parametrize(
    ("amount", "reason", "needs_approval"),
    [
        ("0.01", Reason.REFUND_AUTO_APPROVED, False),
        ("49.99", Reason.REFUND_AUTO_APPROVED, False),
        ("50.00", Reason.REFUND_NEEDS_APPROVAL, True),
        ("50.01", Reason.REFUND_NEEDS_APPROVAL, True),
        ("500.00", Reason.REFUND_NEEDS_APPROVAL, True),
    ],
)
def test_refund_threshold(amount: str, reason: Reason, needs_approval: bool) -> None:
    decision = _refund(amount)
    assert decision.allowed
    assert decision.reason is reason
    assert decision.requires_approval is needs_approval


@pytest.mark.parametrize("amount", ["0", "-5.00"])
def test_refund_must_be_positive(amount: str) -> None:
    decision = _refund(amount)
    assert (decision.allowed, decision.reason) == (False, Reason.REFUND_INVALID_AMOUNT)


def test_refund_cannot_exceed_item_value() -> None:
    decision = _refund("30.00", item_total="40.00", already=Decimal("20.00"))
    assert (decision.allowed, decision.reason) == (False, Reason.REFUND_EXCEEDS_ITEM_VALUE)


def test_refund_up_to_the_remaining_value_is_allowed() -> None:
    assert _refund("20.00", item_total="40.00", already=Decimal("20.00")).allowed


@pytest.mark.parametrize(
    ("category", "allowed"),
    [("apparel", True), ("footwear", True), ("accessories", True), ("electronics", False)],
)
def test_exchange_depends_on_category(category: str, allowed: bool) -> None:
    decision = check_exchange_eligibility(
        make_order(),
        make_item(category=category),
        ItemCondition.GOOD,
        [],
        today=TODAY,
        settings=SETTINGS,
    )
    assert decision.allowed is allowed
    if not allowed:
        assert decision.reason is Reason.CATEGORY_NOT_EXCHANGEABLE


def test_exchange_inherits_return_rules() -> None:
    decision = check_exchange_eligibility(
        make_order(delivered_days_ago=45),
        make_item(),
        ItemCondition.GOOD,
        [],
        today=TODAY,
        settings=SETTINGS,
    )
    assert decision.reason is Reason.RETURN_WINDOW_EXPIRED
