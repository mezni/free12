"""Structured JSON logging that carries the current trace and span ids.

    logger.info("refund.issued", extra={"fields": {"amount": "49.99"}})

becomes one JSON line with ts, level, logger, event, trace_id, span_id and fields.
Values under known-secret keys are redacted.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import IO

from resolve_agent.obs.tracing import current_span

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "password", "secret", "token", "access_token"}
)
_HANDLER_MARKER = "_resolve_agent_handler"


def redact(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            k: REDACTED if str(k).lower() in SENSITIVE_KEYS else redact(v) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


class TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        span = current_span()
        record.trace_id = span.trace_id if span else None
        record.span_id = span.span_id if span else None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for attr in ("trace_id", "span_id"):
            value = getattr(record, attr, None)
            if value:
                payload[attr] = value
        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            payload["fields"] = redact(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(
    level: str = "INFO", *, json_format: bool = True, stream: IO[str] | None = None
) -> None:
    """Idempotent: replaces any handler installed by a previous call."""
    root = logging.getLogger("resolve_agent")
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    setattr(handler, _HANDLER_MARKER, True)
    handler.addFilter(TraceContextFilter())
    handler.setFormatter(
        JsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s trace=%(trace_id)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
