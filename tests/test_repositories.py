import sqlite3
from decimal import Decimal

import pytest

from resolve_agent.domain.clock import Clock
from resolve_agent.domain.models import ItemCondition, RefundStatus, ReturnStatus
from resolve_agent.domain.repositories import (
    AuditLog,
    CustomerRepository,
    OrderRepository,
    RefundRepository,
    ReturnRepository,
)


def test_customer_lookup(seeded: sqlite3.Connection) -> None:
    customer = CustomerRepository(seeded).get("cus_alice")
    assert customer is not None and customer.email == "alice@example.com"
    assert CustomerRepository(seeded).get("cus_unknown") is None


def test_orders_are_scoped_to_their_customer(seeded: sqlite3.Connection) -> None:
    orders = OrderRepository(seeded)
    assert orders.get_for_customer("ord_1001", "cus_alice") is not None
    assert orders.get_for_customer("ord_1001", "cus_bob") is None
    assert orders.get_for_customer("ord_2001", "cus_alice") is None


def test_list_for_customer_only_returns_own_orders(seeded: sqlite3.Connection) -> None:
    ids = {o.id for o in OrderRepository(seeded).list_for_customer("cus_bob")}
    assert ids == {"ord_2001"}


def test_order_items_sum_to_order_total(seeded: sqlite3.Connection) -> None:
    orders = OrderRepository(seeded)
    order = orders.get("ord_1008")
    assert order is not None
    items = orders.items("ord_1008")
    assert len(items) == 2
    assert sum((i.line_total for i in items), Decimal("0")) == order.total == Decimal("62.50")


def test_return_lifecycle(seeded: sqlite3.Connection, clock: Clock) -> None:
    returns = ReturnRepository(seeded, clock)
    created = returns.create(
        order_id="ord_1001",
        order_item_id="ord_1001_i1",
        reason="too small",
        item_condition=ItemCondition.GOOD,
    )
    assert created.status is ReturnStatus.REQUESTED
    returns.set_status(created.id, ReturnStatus.APPROVED)
    stored = returns.get(created.id)
    assert stored is not None and stored.status is ReturnStatus.APPROVED
    assert [r.id for r in returns.for_order("ord_1001")] == [created.id]


def test_return_for_unknown_item_is_rejected(seeded: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        ReturnRepository(seeded).create(
            order_id="ord_1001",
            order_item_id="nope",
            reason="x",
            item_condition=ItemCondition.NEW,
        )


def _return(seeded: sqlite3.Connection, clock: Clock) -> str:
    return (
        ReturnRepository(seeded, clock)
        .create(
            order_id="ord_1002",
            order_item_id="ord_1002_i1",
            reason="wrong size",
            item_condition=ItemCondition.GOOD,
        )
        .id
    )


def test_refund_is_idempotent(seeded: sqlite3.Connection, clock: Clock) -> None:
    return_id = _return(seeded, clock)
    refunds = RefundRepository(seeded, clock)
    kwargs = dict(
        return_id=return_id,
        amount=Decimal("49.99"),
        status=RefundStatus.APPROVED,
        idempotency_key="key-1",
    )
    first = refunds.create(**kwargs)  # type: ignore[arg-type]
    second = refunds.create(**kwargs)  # type: ignore[arg-type]
    assert first.id == second.id
    assert seeded.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 1
    assert refunds.total_active_for_return(return_id) == Decimal("49.99")


def test_refund_key_reuse_with_different_amount_is_an_error(
    seeded: sqlite3.Connection, clock: Clock
) -> None:
    return_id = _return(seeded, clock)
    refunds = RefundRepository(seeded, clock)
    refunds.create(
        return_id=return_id,
        amount=Decimal("49.99"),
        status=RefundStatus.APPROVED,
        idempotency_key="key-1",
    )
    with pytest.raises(ValueError, match="idempotency key"):
        refunds.create(
            return_id=return_id,
            amount=Decimal("10.00"),
            status=RefundStatus.APPROVED,
            idempotency_key="key-1",
        )


def test_refund_amount_must_be_positive(seeded: sqlite3.Connection, clock: Clock) -> None:
    return_id = _return(seeded, clock)
    with pytest.raises(sqlite3.IntegrityError):
        RefundRepository(seeded, clock).create(
            return_id=return_id,
            amount=Decimal("0"),
            status=RefundStatus.APPROVED,
            idempotency_key="zero",
        )
    assert not seeded.in_transaction


def test_rejected_refunds_do_not_count_towards_the_total(
    seeded: sqlite3.Connection, clock: Clock
) -> None:
    return_id = _return(seeded, clock)
    refunds = RefundRepository(seeded, clock)
    refund = refunds.create(
        return_id=return_id,
        amount=Decimal("49.99"),
        status=RefundStatus.PENDING_APPROVAL,
        idempotency_key="k",
    )
    refunds.set_status(refund.id, RefundStatus.REJECTED)
    assert refunds.total_active_for_return(return_id) == Decimal("0.00")


def test_audit_log_records_in_order(seeded: sqlite3.Connection, clock: Clock) -> None:
    audit = AuditLog(seeded, clock)
    audit.record(actor="agent", action="refund.requested", entity="refund", entity_id="r1")
    audit.record(
        actor="manager",
        action="refund.approved",
        entity="refund",
        entity_id="r1",
        details={"amount": Decimal("49.99")},
    )
    entries = audit.entries("refund", "r1")
    assert [e.action for e in entries] == ["refund.requested", "refund.approved"]
    assert entries[1].details == {"amount": "49.99"}
    assert audit.entries("refund", "other") == []
