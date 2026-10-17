"""SQLite schema, version 1. Constraints live in the database, not just in Python."""

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE customers (
    id    TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name  TEXT NOT NULL
);

CREATE TABLE orders (
    id                 TEXT PRIMARY KEY,
    customer_id        TEXT NOT NULL REFERENCES customers(id),
    status             TEXT NOT NULL
        CHECK (status IN ('placed', 'shipped', 'delivered', 'cancelled')),
    placed_on          TEXT NOT NULL,
    delivered_on       TEXT,
    carrier            TEXT,
    tracking_number    TEXT,
    estimated_delivery TEXT,
    total_cents        INTEGER NOT NULL CHECK (total_cents >= 0),
    CHECK (status != 'delivered' OR delivered_on IS NOT NULL)
);
CREATE INDEX idx_orders_customer ON orders(customer_id);

CREATE TABLE order_items (
    id               TEXT PRIMARY KEY,
    order_id         TEXT NOT NULL REFERENCES orders(id),
    sku              TEXT NOT NULL,
    name             TEXT NOT NULL,
    category         TEXT NOT NULL,
    unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents >= 0),
    quantity         INTEGER NOT NULL CHECK (quantity > 0)
);
CREATE INDEX idx_items_order ON order_items(order_id);

CREATE TABLE returns (
    id             TEXT PRIMARY KEY,
    order_id       TEXT NOT NULL REFERENCES orders(id),
    order_item_id  TEXT NOT NULL REFERENCES order_items(id),
    reason         TEXT NOT NULL,
    item_condition TEXT NOT NULL CHECK (item_condition IN ('new', 'good', 'damaged')),
    status         TEXT NOT NULL
        CHECK (status IN ('requested', 'approved', 'rejected', 'received', 'refunded')),
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_returns_order ON returns(order_id);

CREATE TABLE refunds (
    id              TEXT PRIMARY KEY,
    return_id       TEXT NOT NULL REFERENCES returns(id),
    amount_cents    INTEGER NOT NULL CHECK (amount_cents > 0),
    status          TEXT NOT NULL
        CHECK (status IN ('pending_approval', 'approved', 'issued', 'rejected')),
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_refunds_return ON refunds(return_id);

CREATE TABLE audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    entity    TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_audit_entity ON audit_log(entity, entity_id);
"""
