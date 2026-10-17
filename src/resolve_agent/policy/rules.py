"""Business rules as pure functions: no I/O, no clock, no environment, no LLM.

Callers pass in everything the rule needs (including `today`), so every rule is
deterministic and can be tested at its boundaries.

Assumptions (change here, nowhere else):
- The return window starts at delivery and is exclusive: with a 30-day window,
  a return 29 days after delivery is accepted and 30 days after is not.
- Returns cover a whole order line (all units of an item).
- Refund approval applies to each refund individually, not cumulatively per order.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from resolve_agent.config import PolicySettings
from resolve_agent.domain.models import (
    ItemCondition,
    Order,
    OrderItem,
    OrderStatus,
    ReturnRequest,
    ReturnStatus,
)
from resolve_agent.policy.decisions import Decision, Reason

ACCEPTABLE_CONDITIONS = frozenset({ItemCondition.NEW, ItemCondition.GOOD})
ACTIVE_RETURN_STATUSES = frozenset(ReturnStatus) - {ReturnStatus.REJECTED}


def check_return_eligibility(
    order: Order,
    item: OrderItem,
    condition: ItemCondition,
    existing_returns: Sequence[ReturnRequest],
    *,
    today: date,
    settings: PolicySettings,
) -> Decision:
    if item.order_id != order.id:
        return Decision.deny(Reason.ITEM_NOT_IN_ORDER)
    if order.status is OrderStatus.CANCELLED:
        return Decision.deny(Reason.ORDER_CANCELLED)
    if order.status is not OrderStatus.DELIVERED or order.delivered_on is None:
        return Decision.deny(Reason.ORDER_NOT_DELIVERED, {"order_status": order.status.value})
    if any(
        r.order_item_id == item.id and r.status in ACTIVE_RETURN_STATUSES for r in existing_returns
    ):
        return Decision.deny(Reason.ALREADY_RETURNED)

    days = (today - order.delivered_on).days
    if days < 0:
        raise ValueError(f"order {order.id} has a delivery date in the future")
    facts = {"days_since_delivery": days, "window_days": settings.return_window_days}
    if days >= settings.return_window_days:
        return Decision.deny(Reason.RETURN_WINDOW_EXPIRED, facts)
    if condition not in ACCEPTABLE_CONDITIONS:
        return Decision.deny(Reason.CONDITION_NOT_ACCEPTABLE, {"condition": condition.value})
    return Decision.allow(Reason.RETURN_ELIGIBLE, facts)


def check_exchange_eligibility(
    order: Order,
    item: OrderItem,
    condition: ItemCondition,
    existing_returns: Sequence[ReturnRequest],
    *,
    today: date,
    settings: PolicySettings,
) -> Decision:
    base = check_return_eligibility(
        order, item, condition, existing_returns, today=today, settings=settings
    )
    if not base.allowed:
        return base
    if item.category not in settings.exchangeable_categories:
        return Decision.deny(Reason.CATEGORY_NOT_EXCHANGEABLE, {"category": item.category})
    return Decision.allow(Reason.EXCHANGE_ELIGIBLE, base.facts)


def check_refund(
    amount: Decimal,
    *,
    item_total: Decimal,
    already_refunded: Decimal,
    settings: PolicySettings,
) -> Decision:
    if amount <= 0:
        return Decision.deny(Reason.REFUND_INVALID_AMOUNT)
    if amount + already_refunded > item_total:
        return Decision.deny(
            Reason.REFUND_EXCEEDS_ITEM_VALUE,
            {"item_total": item_total, "already_refunded": already_refunded},
        )
    if amount >= settings.auto_refund_limit:
        return Decision.allow(
            Reason.REFUND_NEEDS_APPROVAL,
            {"amount": amount, "auto_refund_limit": settings.auto_refund_limit},
            requires_approval=True,
        )
    return Decision.allow(Reason.REFUND_AUTO_APPROVED, {"amount": amount})
