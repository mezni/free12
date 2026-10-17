"""Wraps any LLMClient with a span, metrics and cost per call (works for Fake and real)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping

from resolve_agent.config import ModelPrice
from resolve_agent.llm.cost import estimate_cost
from resolve_agent.llm.errors import LLMError
from resolve_agent.llm.types import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    StreamDone,
    StreamEvent,
    TextDelta,
)
from resolve_agent.obs import get_metrics, get_tracer, llm_attributes
from resolve_agent.obs.tracing import Span


class ObservedLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        *,
        prices: Mapping[str, ModelPrice] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._prices = prices or {}
        self._monotonic = monotonic

    def complete(self, request: LLMRequest) -> LLMResponse:
        with get_tracer().span(
            "llm.call", kind="llm", attributes=self._start(request, False)
        ) as span:
            try:
                response = self._inner.complete(request)
            except LLMError as err:
                self._failed(span, request, err)
                raise
            self._finished(span, request, response)
            return response

    def stream(self, request: LLMRequest) -> Iterator[StreamEvent]:
        started = self._monotonic()
        first_token_seen = False
        with get_tracer().span(
            "llm.call", kind="llm", attributes=self._start(request, True)
        ) as span:
            try:
                for event in self._inner.stream(request):
                    if isinstance(event, TextDelta) and not first_token_seen:
                        first_token_seen = True
                        span.set_attribute(
                            "llm.first_token_ms", (self._monotonic() - started) * 1000
                        )
                    if isinstance(event, StreamDone):
                        self._finished(span, request, event.response)
                    yield event
            except LLMError as err:
                self._failed(span, request, err)
                raise

    # -- helpers ------------------------------------------------------------------------

    @staticmethod
    def _start(request: LLMRequest, streaming: bool) -> dict[str, object]:
        return {
            **llm_attributes(model=request.model, prompt_version=request.prompt_version),
            "llm.purpose": request.purpose,
            "llm.streaming": streaming,
            "llm.tool_count": len(request.tools),
        }

    def _finished(self, span: Span, request: LLMRequest, response: LLMResponse) -> None:
        model = response.model or request.model
        cost = estimate_cost(request.model, response.usage, self._prices)
        for key, value in llm_attributes(
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
        ).items():
            span.set_attribute(key, value)
        span.set_attribute("llm.stop_reason", response.stop_reason)
        span.set_attribute("llm.attempts", response.attempts)
        if response.request_id:
            span.set_attribute("llm.request_id", response.request_id)
        metrics = get_metrics()
        metrics.inc("llm.calls", 1.0, model=request.model, purpose=request.purpose, outcome="ok")
        metrics.inc(
            "llm.tokens", response.usage.input_tokens, model=request.model, direction="input"
        )
        metrics.inc(
            "llm.tokens", response.usage.output_tokens, model=request.model, direction="output"
        )
        if cost is not None:
            metrics.inc("llm.cost_usd", cost, model=request.model)

    @staticmethod
    def _failed(span: Span, request: LLMRequest, err: LLMError) -> None:
        span.set_attribute("llm.error_type", type(err).__name__)
        if err.status is not None:
            span.set_attribute("llm.status", err.status)
        if err.request_id:
            span.set_attribute("llm.request_id", err.request_id)
        span.record_error(type(err).__name__)
        get_metrics().inc(
            "llm.calls",
            1.0,
            model=request.model,
            purpose=request.purpose,
            outcome=type(err).__name__,
        )
