"""Intent router: a small, fast model classifies each message; code decides what happens.

The router never decides policy and never chooses tools. It answers three questions quickly:
what does the customer want, how do they feel, and must a person take over? Escalation is a
code decision that the model can raise but never suppress.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from resolve_agent.agent.rules import (
    Intent,
    Sentiment,
    classify_by_rules,
    explicit_human_request,
    extract_order_id,
    safety_signals,
    worse,
)
from resolve_agent.llm.errors import LLMError
from resolve_agent.llm.types import LLMClient, Message
from resolve_agent.obs import get_metrics, get_tracer
from resolve_agent.registry import Registry
from resolve_agent.tools.commerce import EscalationReason

E = TypeVar("E", bound=StrEnum)
MIN_CONFIDENCE = 0.5
_VALID_ORDER_ID = re.compile(r"^ord_[A-Za-z0-9]+$")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class RouteSource(StrEnum):
    SHORTCUT = "shortcut"  # explicit request for a person, no model call
    LLM = "llm"
    RULES = "rules"  # model unavailable or unusable


class RouterOutputError(ValueError):
    """The model's reply could not be used. Messages name fields only, never model output."""


@dataclass(frozen=True, slots=True)
class RouterOutput:
    intent: Intent
    confidence: float
    sentiment: Sentiment
    escalate: bool
    order_id: str | None


def _enum(cls: type[E], data: Mapping[str, object], key: str) -> E:
    value = data.get(key)
    if not isinstance(value, str):
        raise RouterOutputError(f"{key}: missing or not a string")
    try:
        return cls(value.strip().lower())
    except ValueError:
        raise RouterOutputError(f"{key}: unknown value") from None


def parse_router_output(text: str) -> RouterOutput:
    """Strict parse of the router model's JSON, tolerating code fences and surrounding prose."""
    cleaned = _FENCE.sub("", text.strip())
    start = cleaned.find("{")
    if start < 0:
        raise RouterOutputError("no JSON object found")
    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned, start)
    except ValueError:
        raise RouterOutputError("invalid JSON") from None
    if not isinstance(data, dict):
        raise RouterOutputError("JSON is not an object")

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise RouterOutputError("confidence: missing or not a number")
    if not 0.0 <= confidence <= 1.0:
        raise RouterOutputError("confidence: out of range")
    escalate = data.get("escalate")
    if not isinstance(escalate, bool):
        raise RouterOutputError("escalate: missing or not a boolean")
    order_id = data.get("order_id")
    if order_id is not None and not isinstance(order_id, str):
        raise RouterOutputError("order_id: not a string")
    return RouterOutput(
        intent=_enum(Intent, data, "intent"),
        confidence=float(confidence),
        sentiment=_enum(Sentiment, data, "sentiment"),
        escalate=escalate,
        order_id=order_id,
    )


@dataclass(frozen=True, slots=True)
class Route:
    intent: Intent
    confidence: float
    sentiment: Sentiment
    escalate: bool
    escalate_reason: EscalationReason | None
    order_id: str | None
    source: RouteSource
    degraded: bool = False  # the model was unavailable or unusable

    @property
    def confident(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE


def _verified_order_id(candidate: str | None, text: str) -> str | None:
    """Accept the model's order id only if it is well-formed and actually in the message."""
    if candidate and _VALID_ORDER_ID.match(candidate) and candidate.lower() in text.lower():
        return candidate.lower()
    return extract_order_id(text)


class IntentRouter:
    """Route one customer message. Pass guard-masked text only."""

    def __init__(self, llm: LLMClient, registry: Registry) -> None:
        self._llm = llm
        self._registry = registry

    def route(self, text: str) -> Route:
        with get_tracer().span("router.route", kind="router") as span:
            route = self._route(text)
            span.set_attribute("router.intent", route.intent.value)
            span.set_attribute("router.source", route.source.value)
            span.set_attribute("router.confidence", round(route.confidence, 3))
            span.set_attribute("router.sentiment", route.sentiment.value)
            span.set_attribute("router.escalate", route.escalate)
            span.set_attribute("router.degraded", route.degraded)
            span.set_attribute("router.has_order_id", route.order_id is not None)
            if route.escalate_reason:
                span.set_attribute("router.escalate_reason", route.escalate_reason.value)
            metrics = get_metrics()
            metrics.inc("router.routes", 1.0, intent=route.intent.value, source=route.source.value)
            if route.escalate_reason:
                metrics.inc("router.escalations", 1.0, reason=route.escalate_reason.value)
            return route

    # -- internals ----------------------------------------------------------------------

    def _route(self, text: str) -> Route:
        signals = safety_signals(text)
        if explicit_human_request(text):  # never make someone wait for a model to reach a person
            return self._finalize(
                Intent.HUMAN_REQUEST,
                1.0,
                signals.sentiment,
                False,
                None,
                text,
                RouteSource.SHORTCUT,
            )

        try:
            request = self._registry.request("router", messages=[Message.user(text)])
            output = parse_router_output(self._llm.complete(request).text)
        except LLMError as err:
            return self._fallback(text, type(err).__name__)
        except RouterOutputError:
            return self._fallback(text, "invalid_output")
        return self._finalize(
            output.intent,
            output.confidence,
            output.sentiment,
            output.escalate,
            output.order_id,
            text,
            RouteSource.LLM,
        )

    def _fallback(self, text: str, reason: str) -> Route:
        get_metrics().inc("router.fallbacks", 1.0, reason=reason)
        rules = classify_by_rules(text)
        return self._finalize(
            rules.intent,
            rules.confidence,
            rules.sentiment,
            False,
            rules.order_id,
            text,
            RouteSource.RULES,
            degraded=True,
        )

    @staticmethod
    def _finalize(
        intent: Intent,
        confidence: float,
        sentiment: Sentiment,
        model_escalate: bool,
        order_id: str | None,
        text: str,
        source: RouteSource,
        *,
        degraded: bool = False,
    ) -> Route:
        signals = safety_signals(text)  # the model can raise sentiment, never lower it
        sentiment = worse(sentiment, signals.sentiment)
        reason: EscalationReason | None = None
        if intent is Intent.HUMAN_REQUEST:
            reason = EscalationReason.CUSTOMER_REQUEST
        elif sentiment is Sentiment.ANGRY:
            reason = EscalationReason.FRUSTRATED_CUSTOMER
        elif model_escalate or signals.legal_or_fraud:
            reason = EscalationReason.OTHER
        return Route(
            intent=intent,
            confidence=confidence,
            sentiment=sentiment,
            escalate=reason is not None,
            escalate_reason=reason,
            order_id=_verified_order_id(order_id, text),
            source=source,
            degraded=degraded,
        )


__all__ = [
    "IntentRouter",
    "Route",
    "RouteSource",
    "RouterOutput",
    "RouterOutputError",
    "parse_router_output",
]
