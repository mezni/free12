import sqlite3
from decimal import Decimal

import pytest

from resolve_agent.domain.models import ItemCondition, ReturnStatus
from resolve_agent.domain.repositories import (
    AuditLog,
    RefundRepository,
    ReturnRepository,
)
from resolve_agent.tools import ToolContext, ToolRegistry, ToolResult, build_registry


@pytest.fixture
def registry() -> ToolRegistry:
    return build_registry()


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _return(registry: ToolRegistry, ctx: ToolContext, order: str = "ord_1002") -> ToolResult:
    return registry.call(
        "create_return",
        {"order_id": order, "item_id": f"{order}_i1", "condition": "good", "reason": "wrong size"},
        ctx,
    )


def first_amount(result: ToolResult) -> Decimal:
    return Decimal(str(result.data["amount"]))


# --- authorization -------------------------------------------------------------------------


def test_own_order_is_visible(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = registry.call("get_order_status", {"order_id": "ord_1005"}, ctx)
    assert result.ok
    assert result.data["order"]["tracking_number"] == "1Z999AA10123456784"  # type: ignore[index]


def test_other_customers_order_looks_like_a_missing_order(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    theirs = registry.call("get_order_status", {"order_id": "ord_2001"}, ctx)
    missing = registry.call("get_order_status", {"order_id": "ord_9999"}, ctx)
    assert (theirs.ok, theirs.error_code) == (False, "order_not_found")
    assert theirs == missing


def test_model_cannot_choose_the_customer(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = registry.call("list_my_orders", {"customer_id": "cus_bob"}, ctx)
    assert result.error_code == "invalid_arguments"


def test_list_my_orders_only_returns_own_orders_and_caps_limit(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    result = registry.call("list_my_orders", {"limit": 500}, ctx)
    ids = [o["order_id"] for o in result.data["orders"]]  # type: ignore[attr-defined]
    assert "ord_2001" not in ids
    assert len(ids) == 8


def test_cannot_use_an_item_from_someone_elses_order(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    result = registry.call(
        "create_return",
        {"order_id": "ord_1001", "item_id": "ord_2001_i1", "condition": "good", "reason": "x"},
        ctx,
    )
    assert result.error_code == "item_not_found"
    assert _count(ctx.conn, "returns") == 0


# --- eligibility checks --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "eligible", "reason"),
    [
        ("ord_1002", True, "return_eligible"),
        ("ord_1003", False, "return_window_expired"),
        ("ord_1005", False, "order_not_delivered"),
        ("ord_1007", False, "order_cancelled"),
    ],
)
def test_check_return_eligibility(
    registry: ToolRegistry, ctx: ToolContext, order: str, eligible: bool, reason: str
) -> None:
    result = registry.call(
        "check_return_eligibility",
        {"order_id": order, "item_id": f"{order}_i1", "condition": "good"},
        ctx,
    )
    assert result.ok
    assert result.data["eligible"] is eligible
    assert result.data["reason"] == reason
    assert result.data["explanation"]
    assert _count(ctx.conn, "returns") == 0


def test_check_exchange_eligibility_uses_category(registry: ToolRegistry, ctx: ToolContext) -> None:
    footwear = registry.call(
        "check_exchange_eligibility",
        {"order_id": "ord_1002", "item_id": "ord_1002_i1", "condition": "good"},
        ctx,
    )
    electronics = registry.call(
        "check_exchange_eligibility",
        {"order_id": "ord_1001", "item_id": "ord_1001_i1", "condition": "good"},
        ctx,
    )
    assert footwear.data["eligible"] is True
    assert electronics.data["reason"] == "category_not_exchangeable"


# --- returns -------------------------------------------------------------------------------


def test_create_return_approves_and_audits(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = _return(registry, ctx)
    assert result.ok and result.data["status"] == "approved"
    stored = ReturnRepository(ctx.conn).get(str(result.data["return_id"]))
    assert stored is not None and stored.status is ReturnStatus.APPROVED
    actions = [e.action for e in AuditLog(ctx.conn).entries("return", stored.id)]
    assert actions == ["return.approved"]


def test_second_return_for_the_same_item_is_refused(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    assert _return(registry, ctx).ok
    again = _return(registry, ctx)
    assert (again.ok, again.error_code, again.data["reason"]) == (
        False,
        "not_eligible",
        "already_returned",
    )
    assert _count(ctx.conn, "returns") == 1


def test_expired_window_creates_nothing(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = _return(registry, ctx, order="ord_1003")
    assert (result.ok, result.error_code) == (False, "not_eligible")
    assert result.data["reason"] == "return_window_expired"
    assert "30 days" in str(result.message)
    assert _count(ctx.conn, "returns") == 0


def test_damaged_items_are_refused(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = registry.call(
        "create_return",
        {"order_id": "ord_1002", "item_id": "ord_1002_i1", "condition": "damaged", "reason": "x"},
        ctx,
    )
    assert result.data["reason"] == "condition_not_acceptable"


# --- refunds -------------------------------------------------------------------------------


def test_small_refund_is_issued_automatically(registry: ToolRegistry, ctx: ToolContext) -> None:
    return_id = str(_return(registry, ctx, "ord_1002").data["return_id"])  # $49.99
    result = registry.call("issue_refund", {"return_id": return_id}, ctx)
    assert result.ok
    assert (result.data["status"], result.data["amount"]) == ("issued", "49.99")
    assert result.data["requires_approval"] is False
    stored = ReturnRepository(ctx.conn).get(return_id)
    assert stored is not None and stored.status is ReturnStatus.REFUNDED


def test_refund_at_the_threshold_needs_approval(registry: ToolRegistry, ctx: ToolContext) -> None:
    return_id = str(_return(registry, ctx, "ord_1001").data["return_id"])  # $120.00
    result = registry.call("issue_refund", {"return_id": return_id}, ctx)
    assert result.ok
    assert (result.data["status"], result.data["requires_approval"]) == ("pending_approval", True)
    stored = ReturnRepository(ctx.conn).get(return_id)
    assert stored is not None and stored.status is ReturnStatus.APPROVED


def test_refund_retry_is_idempotent(registry: ToolRegistry, ctx: ToolContext) -> None:
    return_id = str(_return(registry, ctx, "ord_1002").data["return_id"])
    first = registry.call("issue_refund", {"return_id": return_id}, ctx)
    retry = registry.call("issue_refund", {"return_id": return_id}, ctx)
    assert retry.ok and retry.data["already_processed"] is True
    assert retry.data["refund_id"] == first.data["refund_id"]
    assert _count(ctx.conn, "refunds") == 1
    assert RefundRepository(ctx.conn).total_active_for_return(return_id) == first_amount(first)


def test_model_cannot_choose_the_refund_amount(registry: ToolRegistry, ctx: ToolContext) -> None:
    return_id = str(_return(registry, ctx, "ord_1002").data["return_id"])
    result = registry.call("issue_refund", {"return_id": return_id, "amount": "500.00"}, ctx)
    assert result.error_code == "invalid_arguments"
    assert _count(ctx.conn, "refunds") == 0


def test_refund_for_another_customers_return_is_refused(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    bob = ToolContext("cus_bob", ctx.conn, ctx.settings, ctx.clock)
    bobs_return = str(_return(registry, bob, "ord_2001").data["return_id"])
    result = registry.call("issue_refund", {"return_id": bobs_return}, ctx)
    assert (result.ok, result.error_code) == (False, "return_not_found")
    assert _count(ctx.conn, "refunds") == 0
    bob_retry = registry.call("issue_refund", {"return_id": bobs_return}, bob)
    assert bob_retry.ok


def test_refund_requires_an_approved_return(registry: ToolRegistry, ctx: ToolContext) -> None:
    returns = ReturnRepository(ctx.conn, ctx.clock)
    request = returns.create(
        order_id="ord_1002",
        order_item_id="ord_1002_i1",
        reason="x",
        item_condition=ItemCondition.GOOD,
    )
    result = registry.call("issue_refund", {"return_id": request.id}, ctx)
    assert (result.ok, result.error_code) == (False, "refund_not_allowed")


# --- escalation and audit ------------------------------------------------------------------


def test_escalation_is_recorded(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = registry.call(
        "escalate_to_human",
        {"reason": "frustrated_customer", "summary": "Third contact about a late parcel."},
        ctx,
    )
    assert result.ok and result.data["escalated"] is True
    entries = AuditLog(ctx.conn).entries("customer", "cus_alice")
    escalation = [e for e in entries if e.action == "escalation.requested"]
    assert escalation[0].details["reference"] == result.data["reference"]


def test_escalation_reason_must_be_known(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = registry.call("escalate_to_human", {"reason": "because", "summary": "x"}, ctx)
    assert result.error_code == "invalid_arguments"


def test_every_call_leaves_an_audit_entry(registry: ToolRegistry, ctx: ToolContext) -> None:
    registry.call("get_order_status", {"order_id": "ord_1001"}, ctx)
    registry.call("get_order_status", {"order_id": "ord_2001"}, ctx)
    registry.call("get_order_status", {"nope": 1}, ctx)
    entries = [
        e
        for e in AuditLog(ctx.conn).entries("customer", "cus_alice")
        if e.action.startswith("tool.")
    ]
    assert [(e.action, e.details["ok"], e.details["error_code"]) for e in entries] == [
        ("tool.get_order_status", True, None),
        ("tool.get_order_status", False, "order_not_found"),
        ("tool.get_order_status", False, "invalid_arguments"),
    ]
    assert entries[0].details["request_id"] == "req_test"
