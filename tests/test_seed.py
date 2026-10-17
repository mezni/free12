import sqlite3
from datetime import timedelta
from decimal import Decimal

from conftest import TODAY
from resolve_agent.domain.models import OrderStatus
from resolve_agent.domain.repositories import OrderRepository


def test_boundary_orders_for_the_return_window(seeded: sqlite3.Connection) -> None:
    orders = OrderRepository(seeded)
    for order_id, days_ago in (("ord_1002", 29), ("ord_1003", 30), ("ord_1004", 31)):
        order = orders.get(order_id)
        assert order is not None
        assert order.delivered_on == TODAY - timedelta(days=days_ago)


def test_boundary_orders_for_the_refund_threshold(seeded: sqlite3.Connection) -> None:
    orders = OrderRepository(seeded)
    totals = {oid: orders.get(oid) for oid in ("ord_1002", "ord_1003", "ord_1004")}
    assert [o.total for o in totals.values() if o] == [
        Decimal("49.99"),
        Decimal("50.00"),
        Decimal("50.01"),
    ]


def test_every_order_status_is_represented(seeded: sqlite3.Connection) -> None:
    orders = OrderRepository(seeded).list_for_customer("cus_alice")
    assert {o.status for o in orders} == set(OrderStatus)


def test_shipped_order_has_tracking(seeded: sqlite3.Connection) -> None:
    order = OrderRepository(seeded).get("ord_1005")
    assert order is not None
    assert order.tracking_number and order.estimated_delivery == TODAY + timedelta(days=2)
