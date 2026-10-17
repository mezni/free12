"""Policy decisions: machine-readable reason codes plus the facts behind them.

The LLM never decides policy. It receives a Decision and phrases it for the customer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class Reason(StrEnum):
    RETURN_ELIGIBLE = "return_eligible"
    EXCHANGE_ELIGIBLE = "exchange_eligible"
    ITEM_NOT_IN_ORDER = "item_not_in_order"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_NOT_DELIVERED = "order_not_delivered"
    ALREADY_RETURNED = "already_returned"
    RETURN_WINDOW_EXPIRED = "return_window_expired"
    CONDITION_NOT_ACCEPTABLE = "condition_not_acceptable"
    CATEGORY_NOT_EXCHANGEABLE = "category_not_exchangeable"
    REFUND_AUTO_APPROVED = "refund_auto_approved"
    REFUND_NEEDS_APPROVAL = "refund_needs_approval"
    REFUND_INVALID_AMOUNT = "refund_invalid_amount"
    REFUND_EXCEEDS_ITEM_VALUE = "refund_exceeds_item_value"


@dataclass(frozen=True, slots=True)
class Decision:
    """`allowed`: policy permits proceeding. `requires_approval`: a human must sign off."""

    allowed: bool
    reason: Reason
    requires_approval: bool = False
    facts: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def allow(
        cls,
        reason: Reason,
        facts: Mapping[str, object] | None = None,
        *,
        requires_approval: bool = False,
    ) -> Decision:
        return cls(True, reason, requires_approval, dict(facts or {}))

    @classmethod
    def deny(cls, reason: Reason, facts: Mapping[str, object] | None = None) -> Decision:
        return cls(False, reason, False, dict(facts or {}))


REASON_TEMPLATES: dict[Reason, str] = {
    Reason.RETURN_ELIGIBLE: (
        "This item can be returned: it was delivered {days_since_delivery} days ago."
    ),
    Reason.EXCHANGE_ELIGIBLE: "This item can be exchanged.",
    Reason.ITEM_NOT_IN_ORDER: "That item is not part of this order.",
    Reason.ORDER_CANCELLED: "This order was cancelled, so there is nothing to return.",
    Reason.ORDER_NOT_DELIVERED: (
        "This order has not been delivered yet, so it can't be returned. "
        "Current status: {order_status}."
    ),
    Reason.ALREADY_RETURNED: "A return for this item is already in progress or completed.",
    Reason.RETURN_WINDOW_EXPIRED: (
        "Returns must be requested fewer than {window_days} days after delivery; "
        "this order was delivered {days_since_delivery} days ago."
    ),
    Reason.CONDITION_NOT_ACCEPTABLE: (
        "Items must be in new or good condition to be returned; this one was reported as "
        "{condition}."
    ),
    Reason.CATEGORY_NOT_EXCHANGEABLE: (
        "Items in the {category} category can't be exchanged, but they can be returned "
        "for a refund."
    ),
    Reason.REFUND_AUTO_APPROVED: "A refund of ${amount} can be issued automatically.",
    Reason.REFUND_NEEDS_APPROVAL: (
        "A refund of ${amount} needs manager approval because it is ${auto_refund_limit} or more."
    ),
    Reason.REFUND_INVALID_AMOUNT: "The refund amount must be greater than zero.",
    Reason.REFUND_EXCEEDS_ITEM_VALUE: (
        "That refund would exceed the item's value of ${item_total} "
        "(already refunded: ${already_refunded})."
    ),
}


def explain(decision: Decision) -> str:
    """Plain-language, customer-safe explanation built only from the decision's facts."""
    return REASON_TEMPLATES[decision.reason].format_map(decision.facts)
