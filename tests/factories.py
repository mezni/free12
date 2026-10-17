"""Small builders so policy tests state only what matters."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from resolve_agent.domain.models import (
    ItemCondition,
    Order,
    OrderItem,
    OrderStatus,
    ReturnRequest,
    ReturnStatus,
)

TODAY = date(2026, 10, 16)


def make_order(
    *,
    delivered_days_ago: int | None = 5,
    status: OrderStatus = OrderStatus.DELIVERED,
    order_id: str = "ord_t",
) -> Order:
    delivered_on = (
        None if delivered_days_ago is None else TODAY - timedelta(days=delivered_days_ago)
    )
    return Order(
        id=order_id,
        customer_id="cus_t",
        status=status,
        placed_on=TODAY - timedelta(days=40),
        total=Decimal("100.00"),
        delivered_on=delivered_on,
    )


def make_item(
    *, category: str = "apparel", price: str = "40.00", order_id: str = "ord_t"
) -> OrderItem:
    return OrderItem(
        id=f"{order_id}_i1",
        order_id=order_id,
        sku="SKU-1",
        name="Thing",
        category=category,
        unit_price=Decimal(price),
        quantity=1,
    )


def make_return(item: OrderItem, status: ReturnStatus = ReturnStatus.REQUESTED) -> ReturnRequest:
    return ReturnRequest(
        id="ret_t",
        order_id=item.order_id,
        order_item_id=item.id,
        reason="test",
        item_condition=ItemCondition.GOOD,
        status=status,
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
