"""Tracing with spans propagated through `contextvars`. Stdlib only.

    with get_tracer().span("agent.turn", kind="agent") as span:
        span.set_attribute("intent", "return")
        with get_tracer().span("tool.get_order_status", kind="tool"):
            ...

Nested spans automatically get the current span as parent and share its trace id.
Context variables are per thread and per asyncio task; to keep a parent when handing work
to a worker thread, run it with `contextvars.copy_context().run(...)`.

Never put secrets or raw customer messages in attributes: they are written to disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from resolve_agent.domain.clock import Clock, system_clock
from resolve_agent.obs.metrics import Metrics

MAX_ATTRIBUTE_LENGTH = 1000
logger = logging.getLogger(__name__)


def _clip(value: object) -> object:
    if isinstance(value, str) and len(value) > MAX_ATTRIBUTE_LENGTH:
        return value[:MAX_ATTRIBUTE_LENGTH] + "…"
    return value


@dataclass(slots=True)
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    started_at: datetime
    start_ns: int
    end_ns: int | None = None
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, object] = field(default_factory=dict)

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = _clip(value)

    def record_error(self, message: str) -> None:
        self.status = "error"
        self.error = str(_clip(message))

    @property
    def duration_ms(self) -> float:
        end = self.end_ns if self.end_ns is not None else self.start_ns
        return (end - self.start_ns) / 1_000_000

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class SpanExporter(Protocol):
    def export(self, span: Span) -> None: ...


class InMemoryExporter:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)


class JsonlExporter:
    """Appends one JSON object per finished span."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        line = json.dumps(span.to_dict(), default=str, sort_keys=True)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class MetricsSpanExporter:
    """Turns every finished span into a latency observation and an error count."""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics

    def export(self, span: Span) -> None:
        self._metrics.observe("span.duration_ms", span.duration_ms, kind=span.kind, name=span.name)
        if span.status == "error":
            self._metrics.inc("span.errors", 1.0, kind=span.kind, name=span.name)


_current: ContextVar[Span | None] = ContextVar("resolve_agent_current_span", default=None)


def current_span() -> Span | None:
    return _current.get()


class Tracer:
    def __init__(
        self,
        exporters: Sequence[SpanExporter] = (),
        *,
        clock: Clock = system_clock,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._exporters = tuple(exporters)
        self._clock = clock
        self._monotonic_ns = monotonic_ns

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "internal",
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[Span]:
        parent = _current.get()
        span = Span(
            trace_id=parent.trace_id if parent else secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            parent_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            started_at=self._clock(),
            start_ns=self._monotonic_ns(),
            attributes={k: _clip(v) for k, v in (attributes or {}).items()},
        )
        token = _current.set(span)
        try:
            yield span
        except (GeneratorExit, asyncio.CancelledError):
            span.set_attribute("cancelled", True)  # abandoned or cancelled, not a failure
            raise
        except BaseException as exc:
            if span.status != "error":  # code may have recorded a sanitised error already
                span.record_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            span.end_ns = self._monotonic_ns()
            _current.reset(token)
            self._export(span)

    def _export(self, span: Span) -> None:
        for exporter in self._exporters:
            try:
                exporter.export(span)
            except Exception:  # observability must never break the application
                logger.exception("span exporter failed", extra={"fields": {"span": span.name}})


_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    global _tracer
    _tracer = tracer
