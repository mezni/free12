import contextvars
import json
import threading
from pathlib import Path

import pytest

from resolve_agent.obs import (
    InMemoryExporter,
    JsonlExporter,
    Metrics,
    MetricsSpanExporter,
    Tracer,
    current_span,
)


class FakeMonotonic:
    """Each call advances 5 ms, so durations are deterministic."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        self.now += 5_000_000
        return self.now


def test_no_current_span_outside_a_span() -> None:
    assert current_span() is None


def test_child_spans_share_the_trace_and_point_to_their_parent() -> None:
    exporter = InMemoryExporter()
    tracer = Tracer([exporter])
    with tracer.span("root") as root:
        assert current_span() is root
        with tracer.span("child") as child:
            assert current_span() is child
        assert current_span() is root
    assert current_span() is None
    child_span, root_span = exporter.spans  # children finish first
    assert child_span.trace_id == root_span.trace_id
    assert child_span.parent_id == root_span.span_id
    assert root_span.parent_id is None


def test_sibling_roots_get_different_traces() -> None:
    exporter = InMemoryExporter()
    tracer = Tracer([exporter])
    with tracer.span("a"):
        pass
    with tracer.span("b"):
        pass
    assert exporter.spans[0].trace_id != exporter.spans[1].trace_id


def test_duration_uses_the_monotonic_clock() -> None:
    exporter = InMemoryExporter()
    tracer = Tracer([exporter], monotonic_ns=FakeMonotonic())
    with tracer.span("timed"):
        pass
    assert exporter.spans[0].duration_ms == pytest.approx(5.0)


def test_exception_marks_the_span_and_is_reraised() -> None:
    exporter = InMemoryExporter()
    tracer = Tracer([exporter])
    with pytest.raises(ValueError, match="boom"), tracer.span("failing"):
        raise ValueError("boom")
    span = exporter.spans[0]
    assert span.status == "error"
    assert span.error == "ValueError: boom"
    assert current_span() is None


def test_attributes_are_clipped() -> None:
    exporter = InMemoryExporter()
    tracer = Tracer([exporter])
    with tracer.span("s", attributes={"big": "x" * 5000}) as span:
        span.set_attribute("also_big", "y" * 5000)
    attributes = exporter.spans[0].attributes
    assert len(str(attributes["big"])) == 1001
    assert len(str(attributes["also_big"])) == 1001


def test_a_failing_exporter_never_breaks_the_application() -> None:
    class Broken:
        def export(self, span: object) -> None:
            raise RuntimeError("disk full")

    good = InMemoryExporter()
    tracer = Tracer([Broken(), good])
    with tracer.span("s"):
        pass
    assert len(good.spans) == 1


def test_jsonl_export_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "spans.jsonl"
    tracer = Tracer([JsonlExporter(path)])
    with tracer.span("root", kind="agent", attributes={"intent": "return"}):
        with tracer.span("child", kind="tool"):
            pass
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["name"] for r in rows] == ["child", "root"]
    assert rows[1]["kind"] == "agent"
    assert rows[1]["attributes"] == {"intent": "return"}
    assert rows[0]["parent_id"] == rows[1]["span_id"]
    assert set(rows[0]) == {
        "trace_id", "span_id", "parent_id", "name", "kind", "start",
        "duration_ms", "status", "error", "attributes",
    }  # fmt: skip


def test_metrics_exporter_records_latency_and_errors() -> None:
    metrics = Metrics()
    tracer = Tracer([MetricsSpanExporter(metrics)], monotonic_ns=FakeMonotonic())
    with tracer.span("tool.x", kind="tool"):
        pass
    with pytest.raises(RuntimeError), tracer.span("tool.x", kind="tool"):
        raise RuntimeError
    summary = metrics.summary("span.duration_ms", kind="tool", name="tool.x")
    assert summary is not None and summary.count == 2
    assert metrics.counter("span.errors", kind="tool", name="tool.x") == 1


def test_threads_start_new_traces_unless_the_context_is_copied() -> None:
    exporter = InMemoryExporter()
    tracer = Tracer([exporter])

    def work(name: str) -> None:
        with tracer.span(name):
            pass

    with tracer.span("request") as root:
        plain = threading.Thread(target=work, args=("plain",))
        plain.start()
        plain.join()
        ctx = contextvars.copy_context()
        copied = threading.Thread(target=ctx.run, args=(work, "copied"))
        copied.start()
        copied.join()
    by_name = {s.name: s for s in exporter.spans}
    assert by_name["plain"].trace_id != root.trace_id
    assert by_name["copied"].trace_id == root.trace_id
    assert by_name["copied"].parent_id == root.span_id


def test_closing_a_generator_inside_a_span_is_cancellation_not_error() -> None:
    exporter = InMemoryExporter()
    tracer = Tracer([exporter])

    def numbers():  # type: ignore[no-untyped-def]
        with tracer.span("stream"):
            yield 1
            yield 2

    gen = numbers()
    next(gen)
    gen.close()
    span = exporter.spans[0]
    assert span.status == "ok" and span.attributes["cancelled"] is True


def test_a_sanitised_error_recorded_by_the_caller_is_kept() -> None:
    exporter = InMemoryExporter()
    tracer = Tracer([exporter])
    with pytest.raises(ValueError), tracer.span("s") as span:
        span.record_error("ValueError")
        raise ValueError("contains customer text")
    assert exporter.spans[0].error == "ValueError"
