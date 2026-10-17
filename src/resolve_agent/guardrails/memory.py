"""Gate for text about to be written to long-term memory (used from step 11).

Memory is replayed into prompts later, so it must hold neutral facts only: no PII and nothing
that reads like an instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from resolve_agent.config import GuardrailSettings
from resolve_agent.guardrails import injection, pii

_INSTRUCTION_LIKE = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\byou (?:must|should|will|shall)\b",
        r"\b(?:always|never)\b[^.\n]{0,40}\b(?:approve|refund|ignore|skip|escalate|allow)\b",
        r"\bwhen (?:the )?(?:customer|user|they)\b",
        r"\bfrom now on\b",
    )
)


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    allowed: bool
    text: str  # normalised text to store; empty when rejected
    reason: str | None = None


class MemoryWriteGuard:
    def __init__(self, settings: GuardrailSettings | None = None) -> None:
        self._settings = settings or GuardrailSettings()

    def check(self, fact: str) -> MemoryDecision:
        text, hidden = injection.normalize(fact)
        if not text:
            return MemoryDecision(False, "", "empty")
        if len(text) > self._settings.max_memory_chars:
            return MemoryDecision(False, "", "too_long")
        if hidden:
            return MemoryDecision(False, "", "hidden_characters")
        if pii.mask(text).counts:
            return MemoryDecision(False, "", "contains_pii")
        if injection.scan(text).level is not injection.Level.NONE or any(
            p.search(text) for p in _INSTRUCTION_LIKE
        ):
            return MemoryDecision(False, "", "instruction_like")
        return MemoryDecision(True, text)
