"""Domain entities. Money is Decimal in Python and integer cents in SQLite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum


class OrderStatus(StrEnum):
    PLACED = "placed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class ItemCondition(StrEnum):
    NEW = "new"
    GOOD = "good"
    DAMAGED = "damaged"


class ReturnStatus(StrEnum):
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    RECEIVED = "received"
    REFUNDED = "refunded"


class RefundStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ISSUED = "issued"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Customer:
    id: str
    email: str
    name: str


@dataclass(frozen=True, slots=True)
class Order:
    id: str
    customer_id: str
    status: OrderStatus
    placed_on: date
    total: Decimal
    delivered_on: date | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: date | None = None


@dataclass(frozen=True, slots=True)
class OrderItem:
    id: str
    order_id: str
    sku: str
    name: str
    category: str
    unit_price: Decimal
    quantity: int

    @property
    def line_total(self) -> Decimal:
        return self.unit_price * self.quantity


@dataclass(frozen=True, slots=True)
class ReturnRequest:
    id: str
    order_id: str
    order_item_id: str
    reason: str
    item_condition: ItemCondition
    status: ReturnStatus
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Refund:
    id: str
    return_id: str
    amount: Decimal
    status: RefundStatus
    idempotency_key: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AuditEntry:
    id: int
    ts: datetime
    actor: str
    action: str
    entity: str
    entity_id: str
    details: dict[str, object]
