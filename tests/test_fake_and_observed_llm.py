from decimal import Decimal

import pytest

from llm_helpers import Harness, error_response, ok_response, text_stream
from resolve_agent.config import ModelPrice
from resolve_agent.llm import (
    FakeLLM,
    LLMOverloadedError,
    LLMRequest,
    Message,
    ObservedLLMClient,
    StreamDone,
    TextDelta,
    ToolUseStarted,
    text_response,
    tool_call_response,
)
from resolve_agent.obs import InMemoryExporter, Metrics, set_metrics

REQUEST = LLMRequest(
    model="claude-test",
    messages=[Message.user("hi")],
    purpose="router",
    prompt_version="router@v3",
)
PRICES = {"claude-test": ModelPrice(Decimal("3"), Decimal("15"))}


class FakeClock:
    def __init__(self, *times: float) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        return self._times.pop(0)


# --- FakeLLM ---------------------------------------------------------------------------------


def test_fake_plays_back_in_order_and_records_requests() -> None:
    fake = FakeLLM([text_response("one"), tool_call_response("t", {"a": 1})])
    assert fake.complete(REQUEST).text == "one"
    assert fake.complete(REQUEST).tool_calls[0].name == "t"
    assert fake.requests == [REQUEST, REQUEST]


def test_fake_raises_scripted_errors_and_callables_see_the_request() -> None:
    fake = FakeLLM([LLMOverloadedError("busy"), lambda r: text_response(f"model={r.model}")])
    with pytest.raises(LLMOverloadedError):
        fake.complete(REQUEST)
    assert fake.complete(REQUEST).text == "model=claude-test"


def test_fake_fails_loudly_when_the_script_runs_out() -> None:
    with pytest.raises(AssertionError, match="exhausted"):
        FakeLLM().complete(REQUEST)


def test_fake_streams_words_then_done() -> None:
    fake = FakeLLM([tool_call_response("t", {}, text="Checking now")])
    events = list(fake.stream(REQUEST))
    assert events[:2] == [TextDelta("Checking "), TextDelta("now")]
    assert isinstance(events[2], ToolUseStarted) and isinstance(events[3], StreamDone)


# --- ObservedLLMClient ------------------------------------------------------------------------


@pytest.fixture
def metrics() -> Metrics:
    m = Metrics()
    set_metrics(m)
    return m


def test_complete_records_a_span_with_tokens_and_cost(
    spans: InMemoryExporter, metrics: Metrics
) -> None:
    client = ObservedLLMClient(
        FakeLLM([text_response("hi", input_tokens=1000, output_tokens=200)]), prices=PRICES
    )
    client.complete(REQUEST)
    [span] = spans.spans
    assert (span.name, span.kind, span.status) == ("llm.call", "llm", "ok")
    attrs = span.attributes
    assert attrs["llm.model"] == "fake-model"
    assert attrs["llm.purpose"] == "router" and attrs["llm.prompt_version"] == "router@v3"
    assert (attrs["llm.input_tokens"], attrs["llm.output_tokens"]) == (1000, 200)
    assert attrs["llm.cost_usd"] == pytest.approx(0.006)  # (1000*3 + 200*15) / 1e6
    assert attrs["llm.stop_reason"] == "end_turn" and attrs["llm.streaming"] is False
    assert metrics.counter("llm.calls", model="claude-test", purpose="router", outcome="ok") == 1
    assert metrics.counter("llm.tokens", model="claude-test", direction="output") == 200
    assert metrics.counter("llm.cost_usd", model="claude-test") == pytest.approx(0.006)


def test_cost_is_omitted_without_a_configured_price(spans: InMemoryExporter) -> None:
    ObservedLLMClient(FakeLLM([text_response("hi")])).complete(REQUEST)
    assert "llm.cost_usd" not in spans.spans[0].attributes


def test_errors_mark_the_span_and_count_the_outcome(
    spans: InMemoryExporter, metrics: Metrics
) -> None:
    client = ObservedLLMClient(
        FakeLLM([LLMOverloadedError("busy", status=529, request_id="req_9")])
    )
    with pytest.raises(LLMOverloadedError):
        client.complete(REQUEST)
    span = spans.spans[0]
    assert span.status == "error" and span.error == "LLMOverloadedError"
    assert span.attributes["llm.error_type"] == "LLMOverloadedError"
    assert span.attributes["llm.status"] == 529 and span.attributes["llm.request_id"] == "req_9"
    assert (
        metrics.counter(
            "llm.calls", model="claude-test", purpose="router", outcome="LLMOverloadedError"
        )
        == 1
    )
    assert "busy" not in str(span.attributes) and "busy" not in str(span.error)


def test_stream_records_time_to_first_token(spans: InMemoryExporter) -> None:
    clock = FakeClock(0.0, 0.250)  # start, first token
    client = ObservedLLMClient(FakeLLM([text_response("hello world")]), monotonic=clock)
    list(client.stream(REQUEST))
    attrs = spans.spans[0].attributes
    assert attrs["llm.first_token_ms"] == pytest.approx(250.0)
    assert attrs["llm.streaming"] is True and attrs["llm.output_tokens"] == 5


def test_an_abandoned_stream_is_cancelled_not_failed(spans: InMemoryExporter) -> None:
    client = ObservedLLMClient(FakeLLM([text_response("one two three")]))
    stream = client.stream(REQUEST)
    next(stream)
    stream.close()
    span = spans.spans[0]
    assert span.status == "ok" and span.attributes["cancelled"] is True


def test_retries_surface_as_attempts_and_a_counter(
    spans: InMemoryExporter, metrics: Metrics
) -> None:
    h = Harness([error_response(529), ok_response()])
    client = ObservedLLMClient(h.client)
    response = client.complete(REQUEST)
    assert response.attempts == 2
    assert spans.spans[0].attributes["llm.attempts"] == 2
    assert spans.spans[0].attributes["llm.request_id"] == "req_abc"
    assert metrics.counter("llm.retries", reason="LLMOverloadedError") == 1


def test_streaming_through_the_real_client_reports_usage(spans: InMemoryExporter) -> None:
    h = Harness([text_stream("a ", "b")])
    list(ObservedLLMClient(h.client).stream(REQUEST))
    attrs = spans.spans[0].attributes
    assert (attrs["llm.input_tokens"], attrs["llm.output_tokens"]) == (12, 4)
