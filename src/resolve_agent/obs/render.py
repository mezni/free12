"""Print a trace as an indented tree. Dev tool:

uv run python -m resolve_agent.obs.render traces/spans.jsonl [trace_id]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SHOWN_ATTRIBUTES = (
    "tool.ok",
    "tool.error_code",
    "llm.model",
    "llm.input_tokens",
    "llm.output_tokens",
)


def read_spans(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_trace(spans: Sequence[Mapping[str, Any]]) -> str:
    ids = {s["span_id"] for s in spans}
    children: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
    for span in sorted(spans, key=lambda s: s["start"]):
        parent = span["parent_id"] if span["parent_id"] in ids else None
        children[parent].append(span)

    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for span in children.get(parent, []):
            prefix = "  " * depth + ("└─ " if depth else "")
            flag = "" if span["status"] == "ok" else f"  [{span['status']}: {span['error']}]"
            attrs = span.get("attributes", {})
            shown = "".join(f"  {k}={attrs[k]}" for k in SHOWN_ATTRIBUTES if k in attrs)
            lines.append(f"{prefix}{span['name']}  {span['duration_ms']:.1f}ms{flag}{shown}")
            walk(span["span_id"], depth + 1)

    walk(None, 0)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m resolve_agent.obs.render <spans.jsonl> [trace_id]", file=sys.stderr)
        return 2
    spans = read_spans(args[0])
    if not spans:
        print("no spans found", file=sys.stderr)
        return 1
    trace_id = args[1] if len(args) > 1 else spans[-1]["trace_id"]
    print(render_trace([s for s in spans if s["trace_id"] == trace_id]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
