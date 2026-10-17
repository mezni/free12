from pathlib import Path

from resolve_agent.obs import JsonlExporter, Tracer
from resolve_agent.obs.render import main, read_spans, render_trace


def _span(span_id: str, parent: str | None, name: str, start: str, **extra: object) -> dict:  # type: ignore[type-arg]
    return {
        "trace_id": "t1",
        "span_id": span_id,
        "parent_id": parent,
        "name": name,
        "kind": "internal",
        "start": start,
        "duration_ms": 12.5,
        "status": "ok",
        "error": None,
        "attributes": extra,
    }


def test_tree_is_indented_by_parentage() -> None:
    spans = [
        _span("c2", "root", "tool.issue_refund", "2026-10-16T12:00:00.300", **{"tool.ok": True}),
        _span("root", None, "agent.turn", "2026-10-16T12:00:00.000"),
        _span("c1", "root", "llm.call", "2026-10-16T12:00:00.100", **{"llm.model": "m"}),
    ]
    assert render_trace(spans).splitlines() == [
        "agent.turn  12.5ms",
        "  └─ llm.call  12.5ms  llm.model=m",
        "  └─ tool.issue_refund  12.5ms  tool.ok=True",
    ]


def test_errors_are_flagged() -> None:
    span = _span("a", None, "tool.x", "2026-10-16T12:00:00")
    span.update(status="error", error="internal_error")
    assert "[error: internal_error]" in render_trace([span])


def test_cli_renders_the_latest_trace(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "spans.jsonl"
    tracer = Tracer([JsonlExporter(path)])
    with tracer.span("first"):
        pass
    with tracer.span("second"):
        with tracer.span("inner"):
            pass
    assert len(read_spans(path)) == 3
    assert main([str(path)]) == 0
    output = capsys.readouterr().out
    assert "second" in output and "inner" in output and "first" not in output
    assert main([]) == 2
