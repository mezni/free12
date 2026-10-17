"""Every tool that accepts an identifier must refuse another customer's data. No exceptions."""

import sqlite3
from decimal import Decimal

import pytest

from resolve_agent.config import Settings
from resolve_agent.tools import ToolContext, ToolRegistry, build_registry

NOT_FOUND = {"order_not_found", "item_not_found", "return_not_found"}
BOB_SECRETS = ("ord_2001", "Monitor", "75.00", "cus_bob", "MN-120")


@pytest.fixture
def bob(ctx: ToolContext) -> ToolContext:
    return ToolContext("cus_bob", ctx.conn, Settings.from_env({}), ctx.clock, request_id="req_bob")


@pytest.fixture
def foreign_ids(bob: ToolContext) -> dict[str, object]:
    """Identifiers that belong to Bob, including a return he created."""
    created = build_registry().call(
        "create_return",
        {"order_id": "ord_2001", "item_id": "ord_2001_i1", "condition": "good", "reason": "size"},
        bob,
    )
    assert created.ok
    return {
        "order_id": "ord_2001",
        "item_id": "ord_2001_i1",
        "return_id": created.data["return_id"],
        "condition": "good",
        "reason": "trying to use someone else's order",
    }


def _id_tools(registry: ToolRegistry) -> list[tuple[str, dict[str, object]]]:
    tools = []
    for schema in registry.schemas():
        properties = schema["input_schema"]["properties"]  # type: ignore[index]
        if any(name.endswith("_id") for name in properties):
            tools.append((str(schema["name"]), properties))
    return tools


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_the_matrix_covers_the_expected_tools() -> None:
    names = [name for name, _ in _id_tools(build_registry())]
    assert names == [
        "get_order_status",
        "check_return_eligibility",
        "check_exchange_eligibility",
        "create_return",
        "issue_refund",
    ]


def test_no_tool_exposes_identity_or_money_parameters() -> None:
    banned = {"customer_id", "customer", "email", "amount", "refund_amount", "price", "total"}
    for schema in build_registry().schemas():
        properties = set(schema["input_schema"]["properties"])  # type: ignore[index]
        assert not properties & banned, schema["name"]


def test_every_id_tool_refuses_another_customers_identifiers(
    ctx: ToolContext, foreign_ids: dict[str, object]
) -> None:
    registry = build_registry()
    before = {t: _count(ctx.conn, t) for t in ("returns", "refunds")}
    for name, properties in _id_tools(registry):
        args = {key: value for key, value in foreign_ids.items() if key in properties}
        result = registry.call(name, args, ctx)  # Alice's context, Bob's identifiers
        assert not result.ok and result.error_code in NOT_FOUND, name
        blob = result.to_llm()
        assert not any(secret in blob for secret in BOB_SECRETS), name
    assert {t: _count(ctx.conn, t) for t in ("returns", "refunds")} == before


def test_not_found_is_identical_for_foreign_and_nonexistent_identifiers(
    ctx: ToolContext, foreign_ids: dict[str, object]
) -> None:
    registry = build_registry()
    nonexistent = {
        "order_id": "ord_9999",
        "item_id": "ord_9999_i1",
        "return_id": "ret_nope",
        "condition": "good",
        "reason": "x",
    }
    for name, properties in _id_tools(registry):
        foreign = registry.call(
            name, {k: v for k, v in foreign_ids.items() if k in properties}, ctx
        )
        missing = registry.call(
            name, {k: v for k, v in nonexistent.items() if k in properties}, ctx
        )
        assert (foreign.error_code, foreign.message) == (missing.error_code, missing.message), name


def test_list_tools_never_include_other_customers_data(ctx: ToolContext, bob: ToolContext) -> None:
    registry = build_registry()
    alice = registry.call("list_my_orders", {}, ctx).to_llm()
    assert not any(secret in alice for secret in BOB_SECRETS)
    bobs = registry.call("list_my_orders", {}, bob).to_llm()
    assert "ord_2001" in bobs and "ord_1001" not in bobs


def test_bob_can_still_use_his_own_identifiers(
    bob: ToolContext, foreign_ids: dict[str, object]
) -> None:
    result = build_registry().call("issue_refund", {"return_id": foreign_ids["return_id"]}, bob)
    assert result.ok and result.data["amount"] == str(Decimal("75.00"))
