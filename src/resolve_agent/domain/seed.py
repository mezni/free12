"""Deterministic dev data with the edge cases the policy engine will be tested against.

Dates are relative to `today`, so "delivered 30 days ago" stays true whenever you seed.
Run: uv run python -m resolve_agent.domain.seed
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from resolve_agent.config import Settings
from resolve_agent.domain.db import atomic, connect, init_schema
from resolve_agent.domain.models import Customer, Order, OrderItem, OrderStatus
from resolve_agent.domain.repositories import CustomerRepository, OrderRepository

ALICE = Customer("cus_alice", "alice@example.com", "Alice Martin")
BOB = Customer("cus_bob", "bob@example.com", "Bob Chen")


def _delivered(
    order_id: str,
    customer_id: str,
    today: date,
    days_ago: int,
    items: list[tuple[str, str, str, str, int]],
) -> tuple[Order, list[OrderItem]]:
    delivered_on = today - timedelta(days=days_ago)
    return _build(
        order_id,
        customer_id,
        OrderStatus.DELIVERED,
        delivered_on - timedelta(days=4),
        items,
        delivered_on=delivered_on,
    )


def _build(
    order_id: str,
    customer_id: str,
    status: OrderStatus,
    placed_on: date,
    items: list[tuple[str, str, str, str, int]],
    *,
    delivered_on: date | None = None,
    carrier: str | None = None,
    tracking_number: str | None = None,
    estimated_delivery: date | None = None,
) -> tuple[Order, list[OrderItem]]:
    order_items = [
        OrderItem(f"{order_id}_i{n}", order_id, sku, name, category, Decimal(price), qty)
        for n, (sku, name, category, price, qty) in enumerate(items, start=1)
    ]
    total = sum((i.line_total for i in order_items), Decimal("0"))
    order = Order(
        order_id,
        customer_id,
        status,
        placed_on,
        total,
        delivered_on=delivered_on,
        carrier=carrier,
        tracking_number=tracking_number,
        estimated_delivery=estimated_delivery,
    )
    return order, order_items


def build_orders(today: date) -> list[tuple[Order, list[OrderItem]]]:
    return [
        _delivered(
            "ord_1001", ALICE.id, today, 5, [("HP-100", "Headphones", "electronics", "120.00", 1)]
        ),
        _delivered(
            "ord_1002", ALICE.id, today, 29, [("RS-210", "Running shoes", "footwear", "49.99", 1)]
        ),
        _delivered("ord_1003", ALICE.id, today, 30, [("JK-310", "Jacket", "apparel", "50.00", 1)]),
        _delivered(
            "ord_1004", ALICE.id, today, 31, [("BP-410", "Backpack", "accessories", "50.01", 1)]
        ),
        _delivered(
            "ord_1008",
            ALICE.id,
            today,
            12,
            [("TS-510", "T-shirt", "apparel", "25.00", 2), ("MG-610", "Mug", "home", "12.50", 1)],
        ),
        _build(
            "ord_1005",
            ALICE.id,
            OrderStatus.SHIPPED,
            today - timedelta(days=3),
            [("WB-710", "Water bottle", "home", "18.00", 1)],
            carrier="UPS",
            tracking_number="1Z999AA10123456784",
            estimated_delivery=today + timedelta(days=2),
        ),
        _build(
            "ord_1006",
            ALICE.id,
            OrderStatus.PLACED,
            today - timedelta(days=1),
            [("DL-810", "Desk lamp", "home", "35.00", 1)],
        ),
        _build(
            "ord_1007",
            ALICE.id,
            OrderStatus.CANCELLED,
            today - timedelta(days=9),
            [("KB-910", "Keyboard", "electronics", "80.00", 1)],
        ),
        _delivered(
            "ord_2001", BOB.id, today, 10, [("MN-120", "Monitor", "electronics", "75.00", 1)]
        ),
    ]


def seed(conn: sqlite3.Connection, today: date) -> None:
    customers = CustomerRepository(conn)
    orders = OrderRepository(conn)
    with atomic(conn):
        customers.add(ALICE)
        customers.add(BOB)
        for order, items in build_orders(today):
            orders.add(order, items)


def main() -> int:
    settings = Settings.from_env()
    if settings.env != "dev":
        print(f"refusing to seed: RESOLVE_ENV={settings.env}", file=sys.stderr)
        return 1
    db_path = Path(settings.db_path)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    conn = connect(db_path)
    init_schema(conn)
    seed(conn, date.today())
    print(f"seeded {db_path}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
