"""Tracing, structured logging and metrics (step 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import IO

from resolve_agent.config import Settings
from resolve_agent.obs.conventions import llm_attributes
from resolve_agent.obs.logs import configure_logging
from resolve_agent.obs.metrics import Metrics, get_metrics, set_metrics
from resolve_agent.obs.tracing import (
    InMemoryExporter,
    JsonlExporter,
    MetricsSpanExporter,
    Span,
    SpanExporter,
    Tracer,
    current_span,
    get_tracer,
    set_tracer,
)


@dataclass(frozen=True, slots=True)
class Observability:
    tracer: Tracer
    metrics: Metrics


def setup(settings: Settings, *, stream: IO[str] | None = None) -> Observability:
    """Configure logging, tracing and metrics once at process start."""
    configure_logging(settings.log_level, json_format=settings.log_json, stream=stream)
    metrics = Metrics()
    exporters: list[SpanExporter] = [MetricsSpanExporter(metrics)]
    if settings.trace_path:
        exporters.append(JsonlExporter(settings.trace_path))
    tracer = Tracer(exporters)
    set_tracer(tracer)
    set_metrics(metrics)
    return Observability(tracer, metrics)


__all__ = [
    "InMemoryExporter",
    "JsonlExporter",
    "Metrics",
    "MetricsSpanExporter",
    "Observability",
    "Span",
    "Tracer",
    "configure_logging",
    "current_span",
    "get_metrics",
    "get_tracer",
    "llm_attributes",
    "set_metrics",
    "set_tracer",
    "setup",
]
