"""Repositories: the only place that talks SQL. Every method returns domain objects.

Writes that touch several rows use `atomic()`, so they join the caller's transaction
or start their own. Reads for a customer are always scoped by customer_id.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal

from resolve_agent.domain.clock import Clock, system_clock
from resolve_agent.domain.db import atomic
from resolve_agent.domain.models import (
    AuditEntry,
    Customer,
    ItemCondition,
    Order,
    OrderItem,
    OrderStatus,
    Refund,
    RefundStatus,
    ReturnRequest,
    ReturnStatus,
)
from resolve_agent.domain.money import from_cents, to_cents


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _opt_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _opt_iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _order(r: sqlite3.Row) -> Order:
    return Order(
        id=r["id"],
        customer_id=r["customer_id"],
        status=OrderStatus(r["status"]),
        placed_on=date.fromisoformat(r["placed_on"]),
        total=from_cents(r["total_cents"]),
        delivered_on=_opt_date(r["delivered_on"]),
        carrier=r["carrier"],
        tracking_number=r["tracking_number"],
        estimated_delivery=_opt_date(r["estimated_delivery"]),
    )


def _item(r: sqlite3.Row) -> OrderItem:
    return OrderItem(
        id=r["id"],
        order_id=r["order_id"],
        sku=r["sku"],
        name=r["name"],
        category=r["category"],
        unit_price=from_cents(r["unit_price_cents"]),
        quantity=r["quantity"],
    )


def _return(r: sqlite3.Row) -> ReturnRequest:
    return ReturnRequest(
        id=r["id"],
        order_id=r["order_id"],
        order_item_id=r["order_item_id"],
        reason=r["reason"],
        item_condition=ItemCondition(r["item_condition"]),
        status=ReturnStatus(r["status"]),
        created_at=datetime.fromisoformat(r["created_at"]),
    )


def _refund(r: sqlite3.Row) -> Refund:
    return Refund(
        id=r["id"],
        return_id=r["return_id"],
        amount=from_cents(r["amount_cents"]),
        status=RefundStatus(r["status"]),
        idempotency_key=r["idempotency_key"],
        created_at=datetime.fromisoformat(r["created_at"]),
    )


class CustomerRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, customer: Customer) -> None:
        self._conn.execute(
            "INSERT INTO customers (id, email, name) VALUES (?, ?, ?)",
            (customer.id, customer.email, customer.name),
        )

    def get(self, customer_id: str) -> Customer | None:
        row = self._conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return Customer(row["id"], row["email"], row["name"]) if row else None


class OrderRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, order: Order, items: Sequence[OrderItem]) -> None:
        expected = sum((i.line_total for i in items), Decimal("0"))
        if expected != order.total:
            raise ValueError(f"order total {order.total} does not match items total {expected}")
        with atomic(self._conn):
            self._conn.execute(
                "INSERT INTO orders (id, customer_id, status, placed_on, delivered_on, carrier,"
                " tracking_number, estimated_delivery, total_cents)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order.id,
                    order.customer_id,
                    order.status.value,
                    order.placed_on.isoformat(),
                    _opt_iso(order.delivered_on),
                    order.carrier,
                    order.tracking_number,
                    _opt_iso(order.estimated_delivery),
                    to_cents(order.total),
                ),
            )
            self._conn.executemany(
                "INSERT INTO order_items (id, order_id, sku, name, category,"
                " unit_price_cents, quantity) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        i.id,
                        i.order_id,
                        i.sku,
                        i.name,
                        i.category,
                        to_cents(i.unit_price),
                        i.quantity,
                    )
                    for i in items
                ],
            )

    def get(self, order_id: str) -> Order | None:
        row = self._conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        return _order(row) if row else None

    def get_for_customer(self, order_id: str, customer_id: str) -> Order | None:
        """Customer-scoped lookup. Tools must use this, never `get`."""
        row = self._conn.execute(
            "SELECT * FROM orders WHERE id = ? AND customer_id = ?", (order_id, customer_id)
        ).fetchone()
        return _order(row) if row else None

    def list_for_customer(self, customer_id: str) -> list[Order]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE customer_id = ? ORDER BY placed_on DESC, id",
            (customer_id,),
        ).fetchall()
        return [_order(r) for r in rows]

    def items(self, order_id: str) -> list[OrderItem]:
        rows = self._conn.execute(
            "SELECT * FROM order_items WHERE order_id = ? ORDER BY id", (order_id,)
        ).fetchall()
        return [_item(r) for r in rows]

    def get_item(self, item_id: str) -> OrderItem | None:
        row = self._conn.execute("SELECT * FROM order_items WHERE id = ?", (item_id,)).fetchone()
        return _item(row) if row else None


class ReturnRepository:
    def __init__(self, conn: sqlite3.Connection, clock: Clock = system_clock) -> None:
        self._conn = conn
        self._clock = clock

    def create(
        self,
        *,
        order_id: str,
        order_item_id: str,
        reason: str,
        item_condition: ItemCondition,
        return_id: str | None = None,
    ) -> ReturnRequest:
        request = ReturnRequest(
            id=return_id or new_id("ret"),
            order_id=order_id,
            order_item_id=order_item_id,
            reason=reason,
            item_condition=item_condition,
            status=ReturnStatus.REQUESTED,
            created_at=self._clock(),
        )
        self._conn.execute(
            "INSERT INTO returns (id, order_id, order_item_id, reason, item_condition,"
            " status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                request.id,
                request.order_id,
                request.order_item_id,
                request.reason,
                request.item_condition.value,
                request.status.value,
                request.created_at.isoformat(),
            ),
        )
        return request

    def get(self, return_id: str) -> ReturnRequest | None:
        row = self._conn.execute("SELECT * FROM returns WHERE id = ?", (return_id,)).fetchone()
        return _return(row) if row else None

    def get_for_customer(self, return_id: str, customer_id: str) -> ReturnRequest | None:
        """Customer-scoped lookup (joins the order). Tools must use this, never `get`."""
        row = self._conn.execute(
            "SELECT r.* FROM returns r JOIN orders o ON o.id = r.order_id"
            " WHERE r.id = ? AND o.customer_id = ?",
            (return_id, customer_id),
        ).fetchone()
        return _return(row) if row else None

    def for_order(self, order_id: str) -> list[ReturnRequest]:
        rows = self._conn.execute(
            "SELECT * FROM returns WHERE order_id = ? ORDER BY created_at, id", (order_id,)
        ).fetchall()
        return [_return(r) for r in rows]

    def set_status(self, return_id: str, status: ReturnStatus) -> None:
        cursor = self._conn.execute(
            "UPDATE returns SET status = ? WHERE id = ?", (status.value, return_id)
        )
        if cursor.rowcount != 1:
            raise LookupError(f"return {return_id} not found")


class RefundRepository:
    def __init__(self, conn: sqlite3.Connection, clock: Clock = system_clock) -> None:
        self._conn = conn
        self._clock = clock

    def create(
        self,
        *,
        return_id: str,
        amount: Decimal,
        status: RefundStatus,
        idempotency_key: str,
        refund_id: str | None = None,
    ) -> Refund:
        """Idempotent: the same key and payload returns the existing refund, never a second one."""
        cents = to_cents(amount)
        with atomic(self._conn):
            existing = self.get_by_key(idempotency_key)
            if existing is not None:
                if existing.return_id != return_id or existing.amount != amount:
                    raise ValueError("idempotency key reused with a different request")
                return existing
            refund = Refund(
                id=refund_id or new_id("ref"),
                return_id=return_id,
                amount=from_cents(cents),
                status=status,
                idempotency_key=idempotency_key,
                created_at=self._clock(),
            )
            self._conn.execute(
                "INSERT INTO refunds (id, return_id, amount_cents, status, idempotency_key,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    refund.id,
                    refund.return_id,
                    cents,
                    refund.status.value,
                    refund.idempotency_key,
                    refund.created_at.isoformat(),
                ),
            )
        return refund

    def get(self, refund_id: str) -> Refund | None:
        row = self._conn.execute("SELECT * FROM refunds WHERE id = ?", (refund_id,)).fetchone()
        return _refund(row) if row else None

    def get_by_key(self, idempotency_key: str) -> Refund | None:
        row = self._conn.execute(
            "SELECT * FROM refunds WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return _refund(row) if row else None

    def set_status(self, refund_id: str, status: RefundStatus) -> None:
        cursor = self._conn.execute(
            "UPDATE refunds SET status = ? WHERE id = ?", (status.value, refund_id)
        )
        if cursor.rowcount != 1:
            raise LookupError(f"refund {refund_id} not found")

    def total_active_for_return(self, return_id: str) -> Decimal:
        """Sum of refunds for a return that are not rejected (guards against double refunds)."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) FROM refunds"
            " WHERE return_id = ? AND status != 'rejected'",
            (return_id,),
        ).fetchone()
        return from_cents(int(row[0]))


class AuditLog:
    def __init__(self, conn: sqlite3.Connection, clock: Clock = system_clock) -> None:
        self._conn = conn
        self._clock = clock

    def record(
        self,
        *,
        actor: str,
        action: str,
        entity: str,
        entity_id: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (ts, actor, action, entity, entity_id, details)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._clock().isoformat(),
                actor,
                action,
                entity,
                entity_id,
                json.dumps(details or {}, sort_keys=True, default=str),
            ),
        )

    def entries(self, entity: str, entity_id: str) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE entity = ? AND entity_id = ? ORDER BY id",
            (entity, entity_id),
        ).fetchall()
        return [
            AuditEntry(
                id=r["id"],
                ts=datetime.fromisoformat(r["ts"]),
                actor=r["actor"],
                action=r["action"],
                entity=r["entity"],
                entity_id=r["entity_id"],
                details=json.loads(r["details"]),
            )
            for r in rows
        ]
