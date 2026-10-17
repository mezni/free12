import json

import httpx
import pytest

from llm_helpers import Harness, error_response, ok_response, stream_response, text_stream
from resolve_agent.config import LLMSettings
from resolve_agent.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMOverloadedError,
    LLMRateLimitError,
    LLMRequest,
    LLMServerError,
    LLMStreamInterruptedError,
    LLMTimeoutError,
    Message,
    RetryPolicy,
    StreamDone,
    TextDelta,
    ToolUseStarted,
)

REQUEST = LLMRequest(model="claude-test", messages=[Message.user("where is my order?")])


# --- request shape -------------------------------------------------------------------------


def test_request_shape() -> None:
    h = Harness([ok_response()])
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    h.client.complete(
        LLMRequest(
            model="claude-test",
            messages=[Message.user("hi")],
            system="You are support.",
            tools=tools,
            max_tokens=300,
            temperature=0.0,
        )
    )
    [request] = h.requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = h.body()
    assert body == {
        "model": "claude-test",
        "max_tokens": 300,
        "system": "You are support.",
        "temperature": 0.0,
        "tools": tools,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    assert "sk-test" not in json.dumps(body)


def test_optional_fields_are_omitted_and_base_url_is_configurable() -> None:
    h = Harness([ok_response()], settings=LLMSettings(api_key="k", base_url="http://proxy.local"))
    h.client.complete(REQUEST)
    assert str(h.requests[0].url) == "http://proxy.local/v1/messages"
    assert set(h.body()) == {"model", "max_tokens", "messages"}


# --- non-streaming responses -------------------------------------------------------------


def test_parses_text_usage_and_request_id() -> None:
    response = Harness([ok_response()]).client.complete(REQUEST)
    assert response.text == "Hello there"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 4)
    assert (response.stop_reason, response.model, response.request_id) == (
        "end_turn",
        "claude-test",
        "req_abc",
    )
    assert response.attempts == 1


def test_parses_tool_use_and_ignores_unknown_blocks() -> None:
    body = {
        "model": "claude-test",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Checking."},
            {
                "type": "tool_use",
                "id": "toolu_9",
                "name": "get_order_status",
                "input": {"order_id": "o1"},
            },
        ],
    }
    response = Harness([ok_response(body)]).client.complete(REQUEST)
    assert response.text == "Checking."
    [call] = response.tool_calls
    assert (call.id, call.name, dict(call.input)) == (
        "toolu_9",
        "get_order_status",
        {"order_id": "o1"},
    )
    assert response.stop_reason == "tool_use"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>gateway</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"content": [{"type": "tool_use", "id": "x"}]}),
        httpx.Response(200, json={"content": "text", "usage": {}}),
    ],
)
def test_malformed_responses_are_invalid_and_not_retried(response: httpx.Response) -> None:
    h = Harness([response])
    with pytest.raises(LLMInvalidResponseError):
        h.client.complete(REQUEST)
    assert len(h.requests) == 1


# --- retries -----------------------------------------------------------------------------


def test_overloaded_is_retried_then_succeeds() -> None:
    h = Harness([error_response(529), ok_response()])
    response = h.client.complete(REQUEST)
    assert response.text == "Hello there"
    assert response.attempts == 2
    assert len(h.requests) == 2 and len(h.sleeps) == 1


def test_rate_limit_honours_retry_after() -> None:
    h = Harness([error_response(429, **{"retry-after": "2"}), ok_response()])
    h.client.complete(REQUEST)
    assert h.sleeps[0] >= 2.0


@pytest.mark.parametrize("status", [500, 502, 503, 504, 408])
def test_server_errors_are_retried(status: int) -> None:
    h = Harness([error_response(status), ok_response()])
    assert h.client.complete(REQUEST).attempts == 2


def test_timeouts_and_connection_errors_are_retried() -> None:
    h = Harness([httpx.ReadTimeout("slow"), httpx.ConnectError("refused"), ok_response()])
    assert h.client.complete(REQUEST).attempts == 3


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, LLMBadRequestError),
        (404, LLMBadRequestError),
        (401, LLMAuthError),
        (403, LLMAuthError),
    ],
)
def test_client_errors_are_not_retried(status: int, error: type[Exception]) -> None:
    h = Harness([error_response(status, "bad thing")])
    with pytest.raises(error, match="bad thing"):
        h.client.complete(REQUEST)
    assert len(h.requests) == 1 and h.sleeps == []


def test_attempts_are_bounded_and_the_last_error_is_raised() -> None:
    h = Harness([error_response(529), error_response(529), error_response(529), ok_response()])
    with pytest.raises(LLMOverloadedError) as excinfo:
        h.client.complete(REQUEST)
    assert len(h.requests) == 3  # max_attempts=3
    assert excinfo.value.status == 529


def test_the_time_budget_stops_retries_early() -> None:
    h = Harness(
        [error_response(529), error_response(529), ok_response()],
        retry=RetryPolicy(max_attempts=5, base_delay_s=1.0, total_budget_s=1.2),
    )
    with pytest.raises(LLMOverloadedError):
        h.client.complete(REQUEST)
    assert len(h.requests) == 2  # the second backoff would exceed the budget


def test_exhausted_timeouts_raise_a_timeout_error() -> None:
    h = Harness([httpx.ReadTimeout("slow")] * 3)
    with pytest.raises(LLMTimeoutError):
        h.client.complete(REQUEST)


def test_connection_failure_error_does_not_leak_the_url_or_body() -> None:
    h = Harness([httpx.ConnectError("could not reach where is my order?")] * 3)
    with pytest.raises(LLMConnectionError) as excinfo:
        h.client.complete(REQUEST)
    assert "where is my order" not in str(excinfo.value)


def test_missing_api_key_fails_fast_without_a_request() -> None:
    h = Harness([ok_response()], settings=LLMSettings(api_key=""))
    with pytest.raises(LLMAuthError, match="ANTHROPIC_API_KEY"):
        h.client.complete(REQUEST)
    assert h.requests == []


def test_error_messages_are_truncated() -> None:
    h = Harness([error_response(400, "x" * 5000)])
    with pytest.raises(LLMBadRequestError) as excinfo:
        h.client.complete(REQUEST)
    assert len(str(excinfo.value)) == 300


def test_per_request_timeout_overrides_the_default() -> None:
    h = Harness([ok_response()])
    h.client.complete(LLMRequest(model="m", messages=[Message.user("x")], timeout_s=1.5))
    assert h.requests[0].extensions["timeout"]["read"] == 1.5


# --- streaming ---------------------------------------------------------------------------


def test_stream_yields_deltas_then_a_complete_response() -> None:
    h = Harness([text_stream("Your ", "order ", "shipped.")])
    events = list(h.client.stream(REQUEST))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Your ", "order ", "shipped."]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.response.text == "Your order shipped."
    assert (done.response.usage.input_tokens, done.response.usage.output_tokens) == (12, 4)
    assert (done.response.stop_reason, done.response.model) == ("end_turn", "claude-test")
    assert done.response.request_id == "req_stream" and done.response.attempts == 1
    assert h.body()["stream"] is True


def test_stream_assembles_tool_input_from_json_fragments() -> None:
    h = Harness(
        [
            stream_response(
                {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 5}}},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Looking."},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_order_status",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"order_id": '},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '"ord_1001"}'},
                },
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 9},
                },
                {"type": "message_stop"},
            )
        ]
    )
    events = list(h.client.stream(REQUEST))
    assert [type(e).__name__ for e in events] == ["TextDelta", "ToolUseStarted", "StreamDone"]
    assert events[1] == ToolUseStarted("toolu_1", "get_order_status")
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.response.stop_reason == "tool_use"
    assert [dict(c.input) for c in done.response.tool_calls] == [{"order_id": "ord_1001"}]
    assert done.response.text == "Looking."


def test_stream_retries_on_open_failure() -> None:
    h = Harness([error_response(529), text_stream("ok")])
    events = list(h.client.stream(REQUEST))
    assert isinstance(events[-1], StreamDone) and events[-1].response.attempts == 2
    assert len(h.sleeps) == 1


def test_stream_retries_an_error_event_that_arrives_before_any_content() -> None:
    early_error = stream_response(
        {"type": "message_start", "message": {"model": "m", "usage": {}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    h = Harness([early_error, text_stream("fine")])
    events = list(h.client.stream(REQUEST))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["fine"]


def test_stream_does_not_retry_after_output_was_delivered() -> None:
    mid_error = stream_response(
        {"type": "message_start", "message": {"model": "m", "usage": {}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Part"},
        },
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    h = Harness([mid_error, text_stream("never used")])
    received: list[object] = []
    with pytest.raises(LLMOverloadedError):
        for event in h.client.stream(REQUEST):
            received.append(event)
    assert received == [TextDelta("Part")]
    assert len(h.requests) == 1  # no duplicate request


def test_stream_that_ends_early_is_an_interruption() -> None:
    truncated = stream_response(
        {"type": "message_start", "message": {"model": "m", "usage": {}}},
    )
    h = Harness([truncated] * 3)
    with pytest.raises(LLMStreamInterruptedError):
        list(h.client.stream(REQUEST))
    assert len(h.requests) == 3  # nothing was delivered, so it is retried up to the limit


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        ("rate_limit_error", LLMRateLimitError),
        ("api_error", LLMServerError),
        ("invalid_request_error", LLMBadRequestError),
        ("authentication_error", LLMAuthError),
    ],
)
def test_stream_error_types_map_to_exceptions(error_type: str, expected: type[Exception]) -> None:
    body = stream_response({"type": "error", "error": {"type": error_type, "message": "m"}})
    h = Harness([body] * 3)
    with pytest.raises(expected):
        list(h.client.stream(REQUEST))


def test_stream_with_invalid_json_is_rejected() -> None:
    bad = httpx.Response(200, content=b"data: {not json}\n\n")
    with pytest.raises(LLMInvalidResponseError):
        list(Harness([bad]).client.stream(REQUEST))
