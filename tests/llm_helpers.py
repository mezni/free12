"""Shared builders for LLM client tests: SSE bodies, mock transports, scripted sequences."""

import json
from collections.abc import Callable, Sequence

import httpx

from resolve_agent.config import LLMSettings
from resolve_agent.llm import AnthropicClient, RetryPolicy

OK_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-test",
    "content": [{"type": "text", "text": "Hello there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 4},
}


def ok_response(body: dict | None = None, **headers: str) -> httpx.Response:  # type: ignore[type-arg]
    return httpx.Response(200, json=body or OK_BODY, headers={"request-id": "req_abc", **headers})


def error_response(status: int, message: str = "nope", **headers: str) -> httpx.Response:
    body = {"type": "error", "error": {"type": "x", "message": message}}
    return httpx.Response(status, json=body, headers=headers)


def sse(*events: dict) -> bytes:  # type: ignore[type-arg]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def stream_response(*events: dict) -> httpx.Response:  # type: ignore[type-arg]
    return httpx.Response(
        200,
        content=sse(*events),
        headers={"content-type": "text/event-stream", "request-id": "req_stream"},
    )


def text_stream(*chunks: str, input_tokens: int = 12, output_tokens: int = 4) -> httpx.Response:
    events: list[dict] = [  # type: ignore[type-arg]
        {
            "type": "message_start",
            "message": {"model": "claude-test", "usage": {"input_tokens": input_tokens}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
    ]
    events += [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        }
        for chunk in chunks
    ]
    events += [
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]
    return stream_response(*events)


class Harness:
    """An AnthropicClient on a mock transport, with recorded requests and sleeps."""

    def __init__(
        self,
        responses: Sequence[httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]],
        *,
        settings: LLMSettings | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self.now = 0.0
        self._responses = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item(request) if callable(item) else item

        def sleep(seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds

        import random

        self.client = AnthropicClient(
            settings or LLMSettings(api_key="sk-test"),
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            retry=retry or RetryPolicy(),
            sleep=sleep,
            monotonic=lambda: self.now,
            rng=random.Random(0),
        )

    def body(self, index: int = 0) -> dict:  # type: ignore[type-arg]
        return json.loads(self.requests[index].content)
