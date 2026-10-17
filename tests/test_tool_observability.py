import logging

import pytest

from resolve_agent.domain.repositories import AuditLog
from resolve_agent.obs import (
    InMemoryExporter,
    Metrics,
    Tracer,
    get_metrics,
    set_metrics,
    set_tracer,
)
from resolve_agent.obs.tracing import MetricsSpanExporter
from resolve_agent.tools import ToolContext, ToolRegistry, ToolResult, build_registry
from resolve_agent.tools.result import success


def test_a_tool_call_produces_one_tool_span(spans: InMemoryExporter, ctx: ToolContext) -> None:
    build_registry().call("get_order_status", {"order_id": "ord_1001"}, ctx)
    [span] = spans.spans
    assert (span.name, span.kind, span.status) == ("tool.get_order_status", "tool", "ok")
    assert span.attributes["tool.ok"] is True
    assert span.attributes["tool.writes"] is False
    assert span.attributes["customer_id"] == "cus_alice"
    assert span.attributes["request_id"] == "req_test"


def test_business_refusals_are_not_span_errors(spans: InMemoryExporter, ctx: ToolContext) -> None:
    build_registry().call("get_order_status", {"order_id": "ord_2001"}, ctx)
    [span] = spans.spans
    assert span.status == "ok"
    assert (span.attributes["tool.ok"], span.attributes["tool.error_code"]) == (
        False,
        "order_not_found",
    )


def test_audit_entry_carries_the_trace_id(spans: InMemoryExporter, ctx: ToolContext) -> None:
    build_registry().call("get_order_status", {"order_id": "ord_1001"}, ctx)
    [entry] = [e for e in AuditLog(ctx.conn).entries("customer", "cus_alice")]
    assert entry.details["trace_id"] == spans.spans[0].trace_id


def test_tool_spans_nest_under_the_callers_span(spans: InMemoryExporter, ctx: ToolContext) -> None:
    from resolve_agent.obs import get_tracer

    with get_tracer().span("agent.turn"):
        build_registry().call("list_my_orders", {}, ctx)
    tool_span, turn_span = spans.spans
    assert tool_span.parent_id == turn_span.span_id
    assert tool_span.trace_id == turn_span.trace_id


def test_unknown_tool_names_are_not_used_as_labels(
    spans: InMemoryExporter, ctx: ToolContext
) -> None:
    set_tracer(Tracer([spans]))
    metrics = Metrics()
    set_metrics(metrics)
    build_registry().call("rm_-rf_everything", {}, ctx)
    assert spans.spans[0].name == "tool.unknown"
    assert metrics.counter("tool.calls", tool="unknown", outcome="unknown_tool") == 1


def test_call_counters_and_latency_are_recorded(ctx: ToolContext) -> None:
    metrics = Metrics()
    set_metrics(metrics)
    set_tracer(Tracer([MetricsSpanExporter(metrics)]))
    registry = build_registry()
    registry.call("get_order_status", {"order_id": "ord_1001"}, ctx)
    registry.call("get_order_status", {"order_id": "ord_2001"}, ctx)
    registry.call("get_order_status", {"bad": 1}, ctx)
    assert get_metrics().counter("tool.calls", tool="get_order_status", outcome="ok") == 1
    assert (
        get_metrics().counter("tool.calls", tool="get_order_status", outcome="order_not_found") == 1
    )
    assert (
        get_metrics().counter("tool.calls", tool="get_order_status", outcome="invalid_arguments")
        == 1
    )
    summary = metrics.summary("span.duration_ms", kind="tool", name="tool.get_order_status")
    assert summary is not None and summary.count == 3


def test_a_crashing_tool_marks_the_span_and_logs_without_leaking(
    spans: InMemoryExporter, ctx: ToolContext, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(ctx: ToolContext) -> ToolResult:
        """Crash."""
        raise RuntimeError("secret internal detail")

    registry = ToolRegistry()
    registry.add(boom)
    with caplog.at_level(logging.ERROR, logger="resolve_agent.tools"):
        result = registry.call("boom", {}, ctx)
    assert result.error_code == "internal_error"
    assert spans.spans[0].status == "error"
    assert spans.spans[0].error == "internal_error"
    assert "secret" not in str(spans.spans[0].to_dict())
    record = caplog.records[0]
    assert record.getMessage() == "tool.crashed"
    assert record.fields == {"tool": "boom"}  # type: ignore[attr-defined]


def test_successful_writes_are_spanned_as_writes(spans: InMemoryExporter, ctx: ToolContext) -> None:
    def noop(ctx: ToolContext) -> ToolResult:
        """Write nothing."""
        return success()

    registry = ToolRegistry()
    registry.add(noop, writes=True)
    registry.call("noop", {}, ctx)
    assert spans.spans[0].attributes["tool.writes"] is True
