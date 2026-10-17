import sqlite3

import pytest

from factories import TODAY, make_item, make_order, make_return
from resolve_agent.config import PolicySettings
from resolve_agent.domain.models import ItemCondition, OrderStatus, ReturnStatus
from resolve_agent.domain.repositories import OrderRepository
from resolve_agent.policy import Reason, check_return_eligibility

SETTINGS = PolicySettings()
GOOD = ItemCondition.GOOD


def _check(order, item, condition=GOOD, existing=()):  # type: ignore[no-untyped-def]
    return check_return_eligibility(
        order, item, condition, existing, today=TODAY, settings=SETTINGS
    )


@pytest.mark.parametrize("days", range(0, 61))
def test_window_is_exclusive_at_30_days(days: int) -> None:
    decision = _check(make_order(delivered_days_ago=days), make_item())
    assert decision.allowed is (days < 30)
    expected = Reason.RETURN_ELIGIBLE if days < 30 else Reason.RETURN_WINDOW_EXPIRED
    assert decision.reason is expected


def test_window_facts_are_reported() -> None:
    decision = _check(make_order(delivered_days_ago=31), make_item())
    assert decision.facts == {"days_since_delivery": 31, "window_days": 30}


@pytest.mark.parametrize(
    ("condition", "allowed"),
    [(ItemCondition.NEW, True), (ItemCondition.GOOD, True), (ItemCondition.DAMAGED, False)],
)
def test_condition(condition: ItemCondition, allowed: bool) -> None:
    decision = _check(make_order(), make_item(), condition)
    assert decision.allowed is allowed
    if not allowed:
        assert decision.reason is Reason.CONDITION_NOT_ACCEPTABLE


@pytest.mark.parametrize(
    ("status", "days", "reason"),
    [
        (OrderStatus.PLACED, None, Reason.ORDER_NOT_DELIVERED),
        (OrderStatus.SHIPPED, None, Reason.ORDER_NOT_DELIVERED),
        (OrderStatus.CANCELLED, None, Reason.ORDER_CANCELLED),
    ],
)
def test_order_must_be_delivered(status: OrderStatus, days: int | None, reason: Reason) -> None:
    decision = _check(make_order(delivered_days_ago=days, status=status), make_item())
    assert (decision.allowed, decision.reason) == (False, reason)


def test_item_must_belong_to_the_order() -> None:
    decision = _check(make_order(), make_item(order_id="other"))
    assert decision.reason is Reason.ITEM_NOT_IN_ORDER


def test_active_return_blocks_a_second_return() -> None:
    item = make_item()
    for status in (ReturnStatus.REQUESTED, ReturnStatus.APPROVED, ReturnStatus.REFUNDED):
        decision = _check(make_order(), item, existing=[make_return(item, status)])
        assert decision.reason is Reason.ALREADY_RETURNED


def test_rejected_return_does_not_block_a_new_one() -> None:
    item = make_item()
    decision = _check(make_order(), item, existing=[make_return(item, ReturnStatus.REJECTED)])
    assert decision.allowed


def test_delivery_date_in_the_future_is_a_data_error() -> None:
    with pytest.raises(ValueError, match="future"):
        _check(make_order(delivered_days_ago=-1), make_item())


def test_seeded_orders_resolve_as_expected(seeded: sqlite3.Connection) -> None:
    orders = OrderRepository(seeded)
    expected = {
        "ord_1001": Reason.RETURN_ELIGIBLE,
        "ord_1002": Reason.RETURN_ELIGIBLE,
        "ord_1003": Reason.RETURN_WINDOW_EXPIRED,
        "ord_1004": Reason.RETURN_WINDOW_EXPIRED,
        "ord_1005": Reason.ORDER_NOT_DELIVERED,
        "ord_1006": Reason.ORDER_NOT_DELIVERED,
        "ord_1007": Reason.ORDER_CANCELLED,
    }
    for order_id, reason in expected.items():
        order = orders.get(order_id)
        assert order is not None
        item = orders.items(order_id)[0]
        assert _check(order, item).reason is reason, order_id
