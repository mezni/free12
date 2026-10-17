"""Anthropic Messages API client on plain httpx: retries, timeouts, streaming, error mapping.

Request bodies contain customer text, so they are never logged or put in exceptions.
Only metadata (status, request id, attempt, delay) leaves this module.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

import httpx

from resolve_agent.config import LLMSettings
from resolve_agent.llm.errors import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMStreamInterruptedError,
    LLMTimeoutError,
    error_from_status,
    error_from_type,
)
from resolve_agent.llm.retry import RetryPolicy
from resolve_agent.llm.types import (
    LLMRequest,
    LLMResponse,
    StreamDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    ToolUseStarted,
    Usage,
)
from resolve_agent.obs import get_metrics

API_VERSION = "2023-06-01"
ERROR_SNIPPET_CHARS = 300
logger = logging.getLogger(__name__)


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


def parse_message(data: object, *, request_id: str | None) -> LLMResponse:
    """Parse a non-streaming Messages API response body."""
    try:
        if not isinstance(data, Mapping):
            raise TypeError("response is not an object")
        blocks: list[TextBlock | ToolUseBlock] = []
        for raw in data.get("content") or []:
            kind = raw.get("type")
            if kind == "text":
                blocks.append(TextBlock(str(raw.get("text", ""))))
            elif kind == "tool_use":
                blocks.append(ToolUseBlock(str(raw["id"]), str(raw["name"]), dict(raw["input"])))
            # other block types (e.g. thinking) are ignored
        usage = data.get("usage") or {}
        return LLMResponse(
            content=tuple(blocks),
            stop_reason=str(data.get("stop_reason") or ""),
            usage=Usage(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))),
            model=str(data.get("model", "")),
            request_id=request_id,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise LLMInvalidResponseError(
            f"unexpected response shape: {type(exc).__name__}", request_id=request_id
        ) from exc


def _iter_sse_data(lines: Iterable[str], request_id: str | None) -> Iterator[Mapping[str, Any]]:
    """Yield the JSON payload of each server-sent event (event names are not needed)."""
    data_lines: list[str] = []

    def flush() -> Mapping[str, Any] | None:
        if not data_lines:
            return None
        text = "\n".join(data_lines)
        data_lines.clear()
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise LLMInvalidResponseError("invalid JSON in stream", request_id=request_id) from exc
        if not isinstance(parsed, Mapping):
            raise LLMInvalidResponseError("stream event is not an object", request_id=request_id)
        return parsed

    for line in lines:
        if line == "":
            event = flush()
            if event is not None:
                yield event
        elif line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))
    event = flush()
    if event is not None:
        yield event


class _TextBuilder:
    def __init__(self) -> None:
        self.text = ""

    def build(self) -> TextBlock:
        return TextBlock(self.text)


class _ToolBuilder:
    def __init__(self, tool_id: str, name: str) -> None:
        self.tool_id, self.name = tool_id, name
        self.parts: list[str] = []

    def build(self) -> ToolUseBlock:
        raw = "".join(self.parts)
        try:
            tool_input = json.loads(raw) if raw else {}
        except ValueError as exc:
            raise LLMInvalidResponseError("tool input was not valid JSON") from exc
        if not isinstance(tool_input, dict):
            raise LLMInvalidResponseError("tool input was not an object")
        return ToolUseBlock(self.tool_id, self.name, tool_input)


class AnthropicClient:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        http_client: httpx.Client | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings
        self._http = http_client or httpx.Client()
        self._owns_http = http_client is None
        self._retry = retry or RetryPolicy.from_settings(settings)
        self._sleep = sleep
        self._monotonic = monotonic
        self._rng = rng or random.Random()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    # -- public API ---------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = self._monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._complete_once(request)
            except LLMError as err:
                self._wait_or_raise(err, attempt, started)
            else:
                elapsed_ms = (self._monotonic() - started) * 1000
                return replace(response, attempts=attempt, latency_ms=elapsed_ms)

    def stream(self, request: LLMRequest) -> Iterator[StreamEvent]:
        """Stream events. Retries transparently only until the first event reaches the caller."""
        started = self._monotonic()
        attempt = 0
        while True:
            attempt += 1
            delivered = False
            try:
                for event in self._stream_once(request):
                    delivered = True
                    if isinstance(event, StreamDone):
                        elapsed_ms = (self._monotonic() - started) * 1000
                        event = StreamDone(
                            replace(event.response, attempts=attempt, latency_ms=elapsed_ms)
                        )
                    yield event
                return
            except LLMError as err:
                if delivered:
                    raise  # the caller already saw partial output; a retry would duplicate it
                self._wait_or_raise(err, attempt, started)

    # -- retry helpers ------------------------------------------------------------------

    def _wait_or_raise(self, err: LLMError, attempt: int, started: float) -> None:
        """Sleep and return if the error should be retried, otherwise re-raise it."""
        if not err.retryable or attempt >= self._retry.max_attempts:
            raise err
        delay = self._retry.delay(attempt, err.retry_after, self._rng)
        if (self._monotonic() - started) + delay > self._retry.total_budget_s:
            raise err
        get_metrics().inc("llm.retries", 1.0, reason=type(err).__name__)
        logger.warning(
            "llm.retry",
            extra={
                "fields": {
                    "attempt": attempt,
                    "delay_s": round(delay, 3),
                    "error": type(err).__name__,
                    "status": err.status,
                    "request_id": err.request_id,
                }
            },
        )
        self._sleep(delay)

    # -- HTTP ---------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self._settings.api_key:
            raise LLMAuthError("ANTHROPIC_API_KEY is not set")
        return {
            "x-api-key": self._settings.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def _timeout(self, request: LLMRequest) -> httpx.Timeout:
        seconds = request.timeout_s or self._settings.timeout_s
        return httpx.Timeout(seconds, connect=min(5.0, seconds))

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [m.to_api() for m in request.messages],
        }
        if request.system:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = [dict(t) for t in request.tools]
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream"] = True
        return payload

    @property
    def _url(self) -> str:
        return f"{self._settings.base_url}/v1/messages"

    @staticmethod
    def _error_from_http(response: httpx.Response) -> LLMError:
        request_id = response.headers.get("request-id")
        message = response.text[:ERROR_SNIPPET_CHARS]
        try:
            body = response.json()
            message = str(body["error"]["message"])[:ERROR_SNIPPET_CHARS]
        except (ValueError, KeyError, TypeError):
            pass
        return error_from_status(
            response.status_code,
            message or f"HTTP {response.status_code}",
            retry_after=_parse_retry_after(response.headers.get("retry-after")),
            request_id=request_id,
        )

    def _complete_once(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self._http.post(
                self._url,
                json=self._payload(request, stream=False),
                headers=self._headers(),
                timeout=self._timeout(request),
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("request timed out") from exc
        except httpx.TransportError as exc:
            raise LLMConnectionError(f"connection failed: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise self._error_from_http(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMInvalidResponseError(
                "response was not JSON", request_id=response.headers.get("request-id")
            ) from exc
        return parse_message(data, request_id=response.headers.get("request-id"))

    def _stream_once(self, request: LLMRequest) -> Iterator[StreamEvent]:
        try:
            with self._http.stream(
                "POST",
                self._url,
                json=self._payload(request, stream=True),
                headers=self._headers(),
                timeout=self._timeout(request),
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise self._error_from_http(response)
                request_id = response.headers.get("request-id")
                yield from self._parse_stream(response.iter_lines(), request_id)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("stream timed out") from exc
        except httpx.TransportError as exc:
            raise LLMConnectionError(f"connection failed: {type(exc).__name__}") from exc

    @staticmethod
    def _parse_stream(lines: Iterable[str], request_id: str | None) -> Iterator[StreamEvent]:
        model, stop_reason = "", ""
        input_tokens = output_tokens = 0
        builders: dict[int, _TextBuilder | _ToolBuilder | None] = {}
        finished: dict[int, TextBlock | ToolUseBlock] = {}
        try:
            for data in _iter_sse_data(lines, request_id):
                kind = data.get("type")
                if kind == "message_start":
                    message = data["message"]
                    model = str(message.get("model", ""))
                    input_tokens = int(message.get("usage", {}).get("input_tokens", 0))
                elif kind == "content_block_start":
                    block = data["content_block"]
                    index = int(data["index"])
                    if block["type"] == "text":
                        builders[index] = _TextBuilder()
                    elif block["type"] == "tool_use":
                        builders[index] = _ToolBuilder(str(block["id"]), str(block["name"]))
                        yield ToolUseStarted(str(block["id"]), str(block["name"]))
                    else:
                        builders[index] = None  # unsupported block type: skip it
                elif kind == "content_block_delta":
                    builder = builders.get(int(data["index"]))
                    delta = data["delta"]
                    if isinstance(builder, _TextBuilder) and delta.get("type") == "text_delta":
                        builder.text += delta["text"]
                        yield TextDelta(delta["text"])
                    elif (
                        isinstance(builder, _ToolBuilder)
                        and delta.get("type") == "input_json_delta"
                    ):
                        builder.parts.append(delta["partial_json"])
                elif kind == "content_block_stop":
                    index = int(data["index"])
                    builder = builders.get(index)
                    if builder is not None:
                        finished[index] = builder.build()
                elif kind == "message_delta":
                    stop_reason = str(data.get("delta", {}).get("stop_reason") or stop_reason)
                    usage = data.get("usage", {})
                    output_tokens = int(usage.get("output_tokens", output_tokens))
                    input_tokens = int(usage.get("input_tokens") or input_tokens)
                elif kind == "message_stop":
                    yield StreamDone(
                        LLMResponse(
                            content=tuple(finished[i] for i in sorted(finished)),
                            stop_reason=stop_reason,
                            usage=Usage(input_tokens, output_tokens),
                            model=model,
                            request_id=request_id,
                        )
                    )
                    return
                elif kind == "error":
                    error = data.get("error", {})
                    raise error_from_type(
                        str(error.get("type", "")),
                        str(error.get("message", "stream error"))[:ERROR_SNIPPET_CHARS],
                        request_id=request_id,
                    )
                # "ping" and unknown event types are ignored on purpose
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise LLMInvalidResponseError(
                f"unexpected stream event: {type(exc).__name__}", request_id=request_id
            ) from exc
        raise LLMStreamInterruptedError("stream ended before message_stop", request_id=request_id)
