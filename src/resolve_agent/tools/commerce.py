"""The tools the agent can call. Authorization and policy are enforced here, in code.

Rules every tool follows:
- the customer comes from `ctx`, never from arguments;
- lookups are customer-scoped, and "not yours" looks identical to "does not exist";
- money amounts are computed server-side, never accepted from the model;
- policy decisions come from `resolve_agent.policy`, not from this file.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from resolve_agent import policy
from resolve_agent.domain.models import (
    ItemCondition,
    Order,
    OrderItem,
    RefundStatus,
    ReturnStatus,
)
from resolve_agent.domain.repositories import (
    AuditLog,
    OrderRepository,
    RefundRepository,
    ReturnRepository,
    new_id,
)
from resolve_agent.tools.context import ToolContext
from resolve_agent.tools.registry import ToolRegistry
from resolve_agent.tools.result import ToolError, ToolResult, success

ORDER_NOT_FOUND = "No order with that ID was found for this customer."
MAX_LISTED_ORDERS = 10


class EscalationReason(StrEnum):
    CUSTOMER_REQUEST = "customer_request"
    FRUSTRATED_CUSTOMER = "frustrated_customer"
    OUT_OF_SCOPE = "out_of_scope"
    POLICY_EXCEPTION = "policy_exception"
    SUSPECTED_FRAUD = "suspected_fraud"
    OTHER = "other"


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _order_view(order: Order) -> dict[str, object]:
    return {
        "order_id": order.id,
        "status": order.status.value,
        "placed_on": order.placed_on.isoformat(),
        "delivered_on": _iso(order.delivered_on),
        "carrier": order.carrier,
        "tracking_number": order.tracking_number,
        "estimated_delivery": _iso(order.estimated_delivery),
        "total": str(order.total),
    }


def _item_view(item: OrderItem) -> dict[str, object]:
    return {
        "item_id": item.id,
        "name": item.name,
        "category": item.category,
        "unit_price": str(item.unit_price),
        "quantity": item.quantity,
    }


def _load_order_and_item(ctx: ToolContext, order_id: str, item_id: str) -> tuple[Order, OrderItem]:
    orders = OrderRepository(ctx.conn)
    order = orders.get_for_customer(order_id, ctx.customer_id)
    if order is None:
        raise ToolError("order_not_found", ORDER_NOT_FOUND)
    item = next((i for i in orders.items(order.id) if i.id == item_id), None)
    if item is None:
        raise ToolError("item_not_found", "That item is not part of this order.")
    return order, item


OrderId = Annotated[str, "The order ID, for example ord_1001"]
ItemId = Annotated[str, "The item ID within the order, for example ord_1001_i1"]
Condition = Annotated[ItemCondition, "Condition of the item as described by the customer"]


def get_order_status(ctx: ToolContext, order_id: OrderId) -> ToolResult:
    """Look up the status, tracking details and items of one of the customer's orders."""
    orders = OrderRepository(ctx.conn)
    order = orders.get_for_customer(order_id, ctx.customer_id)
    if order is None:
        raise ToolError("order_not_found", ORDER_NOT_FOUND)
    return success(order=_order_view(order), items=[_item_view(i) for i in orders.items(order.id)])


def list_my_orders(
    ctx: ToolContext, limit: Annotated[int, "How many recent orders to return (1-10)"] = 5
) -> ToolResult:
    """List the customer's most recent orders. Use when they don't know their order ID."""
    limit = max(1, min(limit, MAX_LISTED_ORDERS))
    orders = OrderRepository(ctx.conn).list_for_customer(ctx.customer_id)[:limit]
    return success(orders=[_order_view(o) for o in orders])


def check_return_eligibility(
    ctx: ToolContext, order_id: OrderId, item_id: ItemId, condition: Condition
) -> ToolResult:
    """Check whether an item can be returned under the return policy. Creates nothing."""
    order, item = _load_order_and_item(ctx, order_id, item_id)
    existing = ReturnRepository(ctx.conn).for_order(order.id)
    decision = policy.check_return_eligibility(
        order, item, condition, existing, today=ctx.today, settings=ctx.settings.policy
    )
    return success(
        eligible=decision.allowed,
        reason=decision.reason.value,
        explanation=policy.explain(decision),
    )


def check_exchange_eligibility(
    ctx: ToolContext, order_id: OrderId, item_id: ItemId, condition: Condition
) -> ToolResult:
    """Check whether an item can be exchanged under the exchange policy. Creates nothing."""
    order, item = _load_order_and_item(ctx, order_id, item_id)
    existing = ReturnRepository(ctx.conn).for_order(order.id)
    decision = policy.check_exchange_eligibility(
        order, item, condition, existing, today=ctx.today, settings=ctx.settings.policy
    )
    return success(
        eligible=decision.allowed,
        reason=decision.reason.value,
        explanation=policy.explain(decision),
    )


def create_return(
    ctx: ToolContext,
    order_id: OrderId,
    item_id: ItemId,
    condition: Condition,
    reason: Annotated[str, "Why the customer is returning the item, in their own words"],
) -> ToolResult:
    """Start a return for an item. Only succeeds if the return policy allows it."""
    order, item = _load_order_and_item(ctx, order_id, item_id)
    returns = ReturnRepository(ctx.conn, ctx.clock)
    decision = policy.check_return_eligibility(
        order,
        item,
        condition,
        returns.for_order(order.id),
        today=ctx.today,
        settings=ctx.settings.policy,
    )
    if not decision.allowed:
        raise ToolError("not_eligible", policy.explain(decision), reason=decision.reason.value)

    request = returns.create(
        order_id=order.id, order_item_id=item.id, reason=reason, item_condition=condition
    )
    returns.set_status(request.id, ReturnStatus.APPROVED)
    AuditLog(ctx.conn, ctx.clock).record(
        actor=ctx.actor,
        action="return.approved",
        entity="return",
        entity_id=request.id,
        details={"order_id": order.id, "item_id": item.id, "condition": condition.value},
    )
    return success(
        return_id=request.id,
        status=ReturnStatus.APPROVED.value,
        item=_item_view(item),
        explanation=policy.explain(decision),
    )


def issue_refund(
    ctx: ToolContext, return_id: Annotated[str, "The ID of an approved return, e.g. ret_ab12"]
) -> ToolResult:
    """Refund an approved return. The amount is computed by the system and cannot be chosen."""
    returns = ReturnRepository(ctx.conn, ctx.clock)
    refunds = RefundRepository(ctx.conn, ctx.clock)
    key = f"refund:{return_id}"

    existing = refunds.get_by_key(key)
    if existing is not None and returns.get_for_customer(return_id, ctx.customer_id) is not None:
        return success(
            already_processed=True,
            refund_id=existing.id,
            amount=str(existing.amount),
            status=existing.status.value,
        )

    request = returns.get_for_customer(return_id, ctx.customer_id)
    if request is None:
        raise ToolError("return_not_found", "No return with that ID was found for this customer.")
    if request.status not in (ReturnStatus.APPROVED, ReturnStatus.RECEIVED):
        raise ToolError(
            "refund_not_allowed",
            f"This return is '{request.status.value}' and cannot be refunded.",
        )
    item = OrderRepository(ctx.conn).get_item(request.order_item_id)
    if item is None:
        raise ToolError("internal_error", "The returned item could not be found.")

    already = refunds.total_active_for_return(return_id)
    amount = item.line_total - already
    decision = policy.check_refund(
        amount,
        item_total=item.line_total,
        already_refunded=already,
        settings=ctx.settings.policy,
    )
    if not decision.allowed:
        raise ToolError(
            "refund_not_allowed", policy.explain(decision), reason=decision.reason.value
        )

    status = RefundStatus.PENDING_APPROVAL if decision.requires_approval else RefundStatus.ISSUED
    refund = refunds.create(return_id=return_id, amount=amount, status=status, idempotency_key=key)
    if status is RefundStatus.ISSUED:
        returns.set_status(return_id, ReturnStatus.REFUNDED)
    AuditLog(ctx.conn, ctx.clock).record(
        actor=ctx.actor,
        action=f"refund.{status.value}",
        entity="refund",
        entity_id=refund.id,
        details={"return_id": return_id, "amount": amount},
    )
    return success(
        already_processed=False,
        refund_id=refund.id,
        amount=str(refund.amount),
        status=status.value,
        requires_approval=decision.requires_approval,
        explanation=policy.explain(decision),
    )


def escalate_to_human(
    ctx: ToolContext,
    reason: Annotated[EscalationReason, "Why a human should take over"],
    summary: Annotated[str, "Short summary of the conversation so far for the human agent"],
) -> ToolResult:
    """Hand the conversation to a human agent. Use when asked, or when you cannot resolve it."""
    reference = new_id("esc")
    AuditLog(ctx.conn, ctx.clock).record(
        actor=ctx.actor,
        action="escalation.requested",
        entity="customer",
        entity_id=ctx.customer_id,
        details={"reference": reference, "reason": reason.value, "summary": summary},
    )
    return success(escalated=True, reference=reference)


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(get_order_status)
    registry.add(list_my_orders)
    registry.add(check_return_eligibility)
    registry.add(check_exchange_eligibility)
    registry.add(create_return, writes=True)
    registry.add(issue_refund, writes=True)
    registry.add(escalate_to_human, writes=True)
    return registry
