import io
import json
import logging
from collections.abc import Iterator

import pytest

from resolve_agent.obs import Tracer, configure_logging

LOGGER = logging.getLogger("resolve_agent.test")


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    yield buffer
    root = logging.getLogger("resolve_agent")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.propagate = True
    root.setLevel(logging.NOTSET)


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_json_line_has_the_standard_fields(stream: io.StringIO) -> None:
    configure_logging("INFO", stream=stream)
    LOGGER.info("refund.issued", extra={"fields": {"amount": "49.99"}})
    [line] = _lines(stream)
    assert line["event"] == "refund.issued"
    assert line["level"] == "INFO"
    assert line["logger"] == "resolve_agent.test"
    assert line["fields"] == {"amount": "49.99"}
    assert "ts" in line and "trace_id" not in line


def test_trace_and_span_ids_are_attached_inside_a_span(stream: io.StringIO) -> None:
    configure_logging("INFO", stream=stream)
    with Tracer().span("work") as span:
        LOGGER.info("inside")
    LOGGER.info("outside")
    inside, outside = _lines(stream)
    assert inside["trace_id"] == span.trace_id and inside["span_id"] == span.span_id
    assert "trace_id" not in outside


def test_secrets_are_redacted_even_when_nested(stream: io.StringIO) -> None:
    configure_logging("INFO", stream=stream)
    fields = {"api_key": "sk-1", "headers": {"Authorization": "Bearer x"}, "input_tokens": 12}
    LOGGER.info("call", extra={"fields": fields})
    [line] = _lines(stream)
    assert line["fields"] == {
        "api_key": "[REDACTED]",
        "headers": {"Authorization": "[REDACTED]"},
        "input_tokens": 12,
    }
    assert "sk-1" not in stream.getvalue() and "Bearer" not in stream.getvalue()


def test_exceptions_are_included(stream: io.StringIO) -> None:
    configure_logging("INFO", stream=stream)
    try:
        raise RuntimeError("kaput")
    except RuntimeError:
        LOGGER.exception("failed")
    [line] = _lines(stream)
    assert "RuntimeError: kaput" in str(line["exception"])


def test_level_filters_and_configuration_is_idempotent(stream: io.StringIO) -> None:
    configure_logging("WARNING", stream=stream)
    configure_logging("WARNING", stream=stream)
    LOGGER.info("hidden")
    LOGGER.warning("shown")
    assert [line["event"] for line in _lines(stream)] == ["shown"]  # one handler, no duplicates


def test_plain_format_is_available(stream: io.StringIO) -> None:
    configure_logging("INFO", json_format=False, stream=stream)
    LOGGER.info("hello")
    assert "INFO resolve_agent.test hello trace=None" in stream.getvalue()
