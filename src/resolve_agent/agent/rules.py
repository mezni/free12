"""Deterministic routing rules: the instant human shortcut, a safety net, and the LLM fallback.

These run in plain Python, so they work when the model is slow, down, or being manipulated:
a customer who asks for a person gets one immediately, and an angry or legal-threat message
escalates no matter what the classifier says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

ORDER_ID = re.compile(r"\bord_[A-Za-z0-9]+\b", re.IGNORECASE)

_PEOPLE = (
    r"(?:human|real person|person|representative|rep|someone|manager|supervisor|agent|team member)"
)
_HUMAN_REQUEST = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\b(?:talk|speak|chat)(?:ing)?\s+(?:to|with)\s+(?:a|an|the|your)?\s*{_PEOPLE}\b",
        rf"\b(?:connect|transfer|put)\s+me\s+(?:to|through to|with)\s+(?:a|an|the)?\s*{_PEOPLE}\b",
        r"\b(?:real|live)\s+(?:person|human|agent)\b",
        rf"\bi\s+(?:want|need)\s+(?:a|an|to see a)?\s*{_PEOPLE}\b",
        r"\bescalate\s+(?:this|it|my)\b",
        r"\b(?:human|representative|operator|supervisor|manager)\s+(?:please|now)\b",
    )
)
_LEGAL_OR_FRAUD = re.compile(
    r"\b(?:lawyer|attorney|sue|suing|lawsuit|legal action|small claims|chargeback|"
    r"better business bureau|bbb|scam|scammed|fraud|fraudulent|rip[- ]?off|stolen)\b",
    re.IGNORECASE,
)
_ANGRY = re.compile(
    r"\b(?:furious|unacceptable|ridiculous|outrageous|disgusting|worst|useless|incompetent|"
    r"fed up|sick of|terrible service)\b|\bwtf\b",
    re.IGNORECASE,
)
_FRUSTRATED = re.compile(
    r"\b(?:still|again|third time|second time|waiting|waited|annoyed|frustrated|"
    r"frustrating|disappointed|no response)\b",
    re.IGNORECASE,
)
_EXCHANGE = re.compile(
    r"\b(?:exchange|swap|replace(?:ment)?|different size|another size|bigger size|"
    r"smaller size|other colou?r)\b",
    re.IGNORECASE,
)
_RETURN = re.compile(
    r"\b(?:return|refund|send (?:it )?back|money back|reimburse\w*)\b", re.IGNORECASE
)
_ORDER_STATUS = re.compile(
    r"\b(?:where(?:'s| is)(?: my)? (?:order|package|parcel)|track(?:ing)?|hasn'?t arrived|"
    r"has not arrived|not arrived|when will (?:it|my (?:order|package)) (?:arrive|come)|"
    r"delivery (?:date|status)|shipped|shipping status|late)\b",
    re.IGNORECASE,
)


class Intent(StrEnum):
    ORDER_STATUS = "order_status"
    RETURN = "return"
    EXCHANGE = "exchange"
    HUMAN_REQUEST = "human_request"
    OUT_OF_SCOPE = "out_of_scope"


class Sentiment(StrEnum):
    CALM = "calm"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


_SEVERITY = {Sentiment.CALM: 0, Sentiment.FRUSTRATED: 1, Sentiment.ANGRY: 2}


def worse(a: Sentiment, b: Sentiment) -> Sentiment:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


def explicit_human_request(text: str) -> bool:
    """High precision on purpose: 'the delivery agent' or 'my manager said' must not match."""
    return any(p.search(text) for p in _HUMAN_REQUEST)


def extract_order_id(text: str) -> str | None:
    match = ORDER_ID.search(text)
    return match.group(0).lower() if match else None


@dataclass(frozen=True, slots=True)
class Signals:
    sentiment: Sentiment
    legal_or_fraud: bool


def safety_signals(text: str) -> Signals:
    sentiment = Sentiment.CALM
    if _FRUSTRATED.search(text):
        sentiment = Sentiment.FRUSTRATED
    letters = [c for c in text if c.isalpha()]
    shouting = len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.8
    if shouting or text.count("!") >= 3:
        sentiment = worse(sentiment, Sentiment.FRUSTRATED)
    legal = bool(_LEGAL_OR_FRAUD.search(text))
    if _ANGRY.search(text) or legal:
        sentiment = Sentiment.ANGRY
    return Signals(sentiment, legal)


@dataclass(frozen=True, slots=True)
class RuleResult:
    intent: Intent
    confidence: float
    sentiment: Sentiment
    legal_or_fraud: bool
    order_id: str | None


def classify_by_rules(text: str) -> RuleResult:
    """Keyword fallback used when the model is unavailable or returns something unusable."""
    signals = safety_signals(text)
    if explicit_human_request(text):
        intent, confidence = Intent.HUMAN_REQUEST, 0.9
    elif _EXCHANGE.search(text):
        intent, confidence = Intent.EXCHANGE, 0.6
    elif _RETURN.search(text):
        intent, confidence = Intent.RETURN, 0.6
    elif _ORDER_STATUS.search(text):
        intent, confidence = Intent.ORDER_STATUS, 0.6
    else:
        intent, confidence = Intent.OUT_OF_SCOPE, 0.3
    return RuleResult(
        intent, confidence, signals.sentiment, signals.legal_or_fraud, extract_order_id(text)
    )
