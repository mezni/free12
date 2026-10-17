import json
from typing import Annotated, Literal

import pytest

from resolve_agent.domain.models import ItemCondition
from resolve_agent.tools import ToolContext, ToolRegistry, ToolResult, build_registry
from resolve_agent.tools.result import success


def sample(
    ctx: ToolContext,
    order_id: Annotated[str, "The order ID"],
    condition: ItemCondition,
    limit: int = 5,
    note: str | None = None,
    channel: Literal["email", "chat"] = "chat",
    urgent: bool = False,
) -> ToolResult:
    """Sample tool.

    Longer text that must not appear in the description.
    """
    return success(order_id=order_id, condition=condition, limit=limit, note=note, channel=channel)


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.add(sample)
    return r


def test_schema_is_generated_from_type_hints(registry: ToolRegistry) -> None:
    [schema] = registry.schemas()
    assert schema["name"] == "sample"
    assert schema["description"] == "Sample tool."
    input_schema = schema["input_schema"]
    assert isinstance(input_schema, dict)
    assert input_schema["additionalProperties"] is False
    assert input_schema["required"] == ["order_id", "condition"]
    props = input_schema["properties"]
    assert props["order_id"] == {"type": "string", "maxLength": 500, "description": "The order ID"}
    assert props["condition"] == {"type": "string", "enum": ["new", "good", "damaged"]}
    assert props["limit"] == {"type": "integer"}
    assert props["channel"] == {"type": "string", "enum": ["email", "chat"]}
    assert props["urgent"] == {"type": "boolean"}
    assert "ctx" not in props


def test_schemas_are_json_serialisable() -> None:
    json.dumps(build_registry().schemas())


def test_every_real_tool_is_documented_and_hides_customer_identity() -> None:
    for schema in build_registry().schemas():
        assert schema["description"], schema["name"]
        props = schema["input_schema"]["properties"]  # type: ignore[index]
        assert "customer_id" not in props, schema["name"]
        assert "amount" not in props, schema["name"]


def test_expected_tools_are_registered() -> None:
    assert build_registry().names() == [
        "check_exchange_eligibility",
        "check_return_eligibility",
        "create_return",
        "escalate_to_human",
        "get_order_status",
        "issue_refund",
        "list_my_orders",
    ]


def test_valid_call_coerces_types(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = registry.call("sample", {"order_id": " ord_1 ", "condition": "good"}, ctx)
    assert result.ok
    assert result.data["order_id"] == "ord_1"
    assert result.data["condition"] is ItemCondition.GOOD
    assert result.data["limit"] == 5


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"condition": "good"}, "missing required argument 'order_id'"),
        ({"order_id": "o", "condition": "good", "customer_id": "cus_bob"}, "unexpected argument"),
        ({"order_id": 5, "condition": "good"}, "'order_id' must be a non-empty string"),
        ({"order_id": "  ", "condition": "good"}, "non-empty string"),
        ({"order_id": "x" * 501, "condition": "good"}, "at most 500"),
        ({"order_id": "o", "condition": "broken"}, "'condition' must be one of"),
        ({"order_id": "o", "condition": "good", "limit": "3"}, "'limit' must be an integer"),
        ({"order_id": "o", "condition": "good", "limit": True}, "'limit' must be an integer"),
        ({"order_id": "o", "condition": "good", "urgent": "yes"}, "'urgent' must be a boolean"),
        ({"order_id": "o", "condition": "good", "channel": "fax"}, "'channel' must be one of"),
        ({"order_id": None, "condition": "good"}, "missing required argument"),
    ],
)
def test_invalid_arguments_are_rejected(
    registry: ToolRegistry, ctx: ToolContext, args: dict[str, object], fragment: str
) -> None:
    result = registry.call("sample", args, ctx)
    assert not result.ok
    assert result.error_code == "invalid_arguments"
    assert result.message is not None and fragment in result.message


def test_non_object_arguments_are_rejected(registry: ToolRegistry, ctx: ToolContext) -> None:
    assert registry.call("sample", ["x"], ctx).error_code == "invalid_arguments"


def test_unknown_tool(registry: ToolRegistry, ctx: ToolContext) -> None:
    result = registry.call("drop_tables", {}, ctx)
    assert (result.ok, result.error_code) == (False, "unknown_tool")


def test_unexpected_exception_becomes_internal_error_and_rolls_back(ctx: ToolContext) -> None:
    def boom(ctx: ToolContext) -> ToolResult:
        """Insert then crash."""
        ctx.conn.execute("INSERT INTO customers VALUES ('c9', 'c9@example.com', 'C Nine')")
        raise RuntimeError("secret internal detail")

    registry = ToolRegistry()
    registry.add(boom, writes=True)
    result = registry.call("boom", {}, ctx)
    assert (result.ok, result.error_code) == (False, "internal_error")
    assert "secret" not in result.to_llm()
    assert ctx.conn.execute("SELECT COUNT(*) FROM customers WHERE id = 'c9'").fetchone()[0] == 0
    assert not ctx.conn.in_transaction


def test_registration_rules() -> None:
    def no_ctx(order_id: str) -> ToolResult:
        """Doc."""
        return success()

    def no_doc(ctx: ToolContext) -> ToolResult:
        return success()

    def bad_type(ctx: ToolContext, values: list[str]) -> ToolResult:
        """Doc."""
        return success()

    r = ToolRegistry()
    for fn in (no_ctx, no_doc, bad_type):
        with pytest.raises(TypeError):
            r.add(fn)
    r.add(sample)
    with pytest.raises(ValueError, match="already registered"):
        r.add(sample)


def test_result_serialisation() -> None:
    from decimal import Decimal

    assert success(amount=Decimal("49.99")).to_llm() == '{"amount": "49.99", "ok": true}'
    failed = ToolResult(False, {"reason": "x"}, "not_eligible", "Nope")
    assert json.loads(failed.to_llm()) == {
        "ok": False,
        "error": "not_eligible",
        "message": "Nope",
        "reason": "x",
    }
