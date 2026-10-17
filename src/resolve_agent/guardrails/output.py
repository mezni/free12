"""Output guardrail: check a reply against the evidence in this turn's tool results.

The prompt asks the model not to invent amounts, rules or outcomes; this check enforces it.
Anything the reply states that no tool result supports is a violation, and the caller falls
back to a safe message (and usually a human handoff) instead of sending it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from resolve_agent.config import GuardrailSettings
from resolve_agent.guardrails import messages, pii
from resolve_agent.tools.result import ToolResult

MONEY_KEYS = frozenset(
    {"amount", "total", "unit_price", "item_total", "already_refunded", "auto_refund_limit"}
)
DAY_KEYS = frozenset({"days_since_delivery", "window_days"})
SHINGLE_WORDS = 8

_DOLLAR = re.compile(r"(?<![\w.])\$\s?(\d+(?:,\d{3})*(?:\.\d+)?)")
_WORD_AMOUNT = re.compile(r"\b(\d+(?:\.\d{1,2})?)\s?(?:dollars|usd|cad)\b", re.IGNORECASE)
_DAYS = re.compile(r"\b(\d+)[- ]days?\b", re.IGNORECASE)

# up to 60 characters within one sentence; a decimal point between digits is not a sentence end
_SPAN = r"(?:[^.!?]|(?<=\d)\.(?=\d)){0,60}"

_REFUND_DONE = (
    re.compile(
        r"\b(?:i|we)(?:'ve| have)?\s+(?:just\s+)?(?:issued|processed|refunded|credited)\b", re.I
    ),
    re.compile(
        rf"\brefund\b{_SPAN}\b(?:has been|was|is)\s+(?:successfully\s+)?"
        r"(?:issued|processed|sent|credited|completed)\b",
        re.I,
    ),
)
_RETURN_APPROVED = re.compile(rf"\breturn\b{_SPAN}\b(?:has been|was|is)\s+approved\b", re.I)
_ESCALATED = re.compile(
    r"\b(?:i(?:'ve| have)|i am|i'm|i will|i'll)\s+(?:now\s+)?"
    r"(?:escalat\w+|transfer\w*|hand\w*\s+(?:you|this|it)\s+over|connect\w*\s+you)\b",
    re.I,
)


class Severity(StrEnum):
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class Violation:
    code: str
    severity: Severity
    detail: str = ""  # non-sensitive: an amount, a day count or an id, never customer text


@dataclass(frozen=True, slots=True)
class OutputResult:
    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def escalate(self) -> bool:
        return any(v.severity is Severity.HIGH for v in self.violations)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(v.code for v in self.violations)

    def reply_or_fallback(self, reply: str) -> str:
        return reply if self.ok else messages.OUTPUT_FALLBACK_REPLY


def _decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def _walk(value: object, key: str = "") -> Iterable[tuple[str, object]]:
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from _walk(v, str(k))
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _walk(item, key)
    else:
        yield key, value


@dataclass(frozen=True, slots=True)
class Evidence:
    """What this turn's tool results actually established."""

    amounts: frozenset[Decimal] = frozenset()
    day_counts: frozenset[int] = frozenset()
    ids: frozenset[str] = frozenset()
    refund_issued: bool = False
    return_approved: bool = False
    escalated: bool = False

    @classmethod
    def from_results(
        cls, results: Iterable[ToolResult], settings: GuardrailSettings | None = None
    ) -> Evidence:
        id_pattern = re.compile((settings or GuardrailSettings()).id_pattern, re.IGNORECASE)
        amounts: set[Decimal] = set()
        days: set[int] = set()
        ids: set[str] = set()
        flags = {"refund_issued": False, "return_approved": False, "escalated": False}
        for result in results:
            leaves = list(_walk(result.data))
            if result.message:
                leaves.append(("message", result.message))
            for key, value in leaves:
                text = str(value)
                ids.update(m.lower() for m in id_pattern.findall(text))
                if key in MONEY_KEYS and (amount := _decimal(text)) is not None:
                    amounts.add(amount)
                if key in DAY_KEYS and text.isdigit():
                    days.add(int(text))
                if isinstance(value, str):
                    amounts.update(
                        a for m in _DOLLAR.findall(value) if (a := _decimal(m)) is not None
                    )
                    days.update(int(m) for m in _DAYS.findall(value))
            data = result.data
            status = data.get("status")
            flags["refund_issued"] |= bool(data.get("refund_id")) and status == "issued"
            flags["return_approved"] |= bool(data.get("return_id")) and status == "approved"
            flags["escalated"] |= data.get("escalated") is True
        return cls(frozenset(amounts), frozenset(days), frozenset(ids), **flags)


def _reply_amounts(text: str) -> list[Decimal]:
    found = _DOLLAR.findall(text) + _WORD_AMOUNT.findall(text)
    return [a for m in found if (a := _decimal(m)) is not None]


def _shingles(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}


class OutputGuard:
    def __init__(self, settings: GuardrailSettings | None = None) -> None:
        self._settings = settings or GuardrailSettings()
        self._id_pattern = re.compile(self._settings.id_pattern, re.IGNORECASE)

    def check(
        self,
        reply: str,
        evidence: Evidence,
        *,
        customer_text: str = "",
        system_prompt: str = "",
    ) -> OutputResult:
        violations: list[Violation] = []

        for amount in _reply_amounts(reply):
            if amount not in evidence.amounts:
                violations.append(Violation("unsupported_amount", Severity.MEDIUM, str(amount)))
        for days in (int(m) for m in _DAYS.findall(reply)):
            if days not in evidence.day_counts:
                violations.append(Violation("unsupported_day_count", Severity.MEDIUM, str(days)))

        known_ids = evidence.ids | {m.lower() for m in self._id_pattern.findall(customer_text)}
        for found in sorted({m.lower() for m in self._id_pattern.findall(reply)}):
            if found not in known_ids:
                violations.append(Violation("unknown_id", Severity.MEDIUM, found))

        if not evidence.refund_issued and any(p.search(reply) for p in _REFUND_DONE):
            violations.append(Violation("unsupported_refund_claim", Severity.HIGH))
        if not evidence.return_approved and _RETURN_APPROVED.search(reply):
            violations.append(Violation("unsupported_return_claim", Severity.MEDIUM))
        if not evidence.escalated and _ESCALATED.search(reply):
            violations.append(Violation("unsupported_escalation_claim", Severity.MEDIUM))

        if pii.find_secrets(reply):
            violations.append(Violation("pii_in_reply", Severity.HIGH))
        if system_prompt and _shingles(reply) & _shingles(system_prompt):
            violations.append(Violation("system_prompt_leak", Severity.HIGH))
        return OutputResult(tuple(violations))
