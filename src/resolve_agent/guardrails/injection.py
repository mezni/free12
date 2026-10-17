"""Prompt-injection screening: normalise first, then match.

This is a tripwire, not a wall. Phrase matches only FLAG a message (customers say odd things);
structural attacks (chat-template tokens, fake tool markup) BLOCK it. Real protection comes
from the design: tools enforce authorization and policy in code, so a successful injection
still cannot exceed the customer's own permissions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class Level(StrEnum):
    NONE = "none"
    FLAG = "flag"
    BLOCK = "block"


# zero-width, soft hyphen, bidi controls and the Unicode "tags" block (invisible text smuggling)
_INVISIBLE = re.compile(
    "[\u200b\u200c\u2060\ufeff\u00ad\u202a-\u202e\u2066-\u2069\U000e0000-\U000e007f]"
)

FLAG_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "ignore_instructions": (
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|all|your|these|the)\b[^.\n]{0,20}"
            r"\b(?:instructions?|rules?|guidelines?|prompts?|policies|policy)\b"
        ),
        "reveal_prompt": (
            r"\b(?:reveal|show|print|repeat|output|display|tell me)\b[^.\n]{0,40}"
            r"\b(?:system|hidden|initial|original|secret)\b[^.\n]{0,15}"
            r"\b(?:prompt|message|instructions?)\b"
        ),
        "role_reassignment": (
            r"\b(?:you are now|from now on,? you|pretend (?:to be|you are)"
            r"|act as (?:an? )?(?:admin|administrator|manager|system|developer|unrestricted))\b"
        ),
        "special_mode": r"\b(?:developer|debug|admin|god|jailbreak|dan) mode\b|\bjailbreak\b",
        "bypass_controls": (
            r"\b(?:bypass|skip|disable|turn off|override)\b[^.\n]{0,30}"
            r"\b(?:checks?|rules?|polic(?:y|ies)|verification|guardrails?|restrictions?|approval)\b"
        ),
        "authority_claim": (
            r"\bi(?:'m| am) (?:the |an? )?"
            r"(?:admin|administrator|manager|developer|owner|engineer|supervisor|anthropic)\b"
        ),
        "role_prefix": r"^\s*(?:system|assistant|developer)\s*:",
    }.items()
}
FLAG_PATTERNS["role_prefix"] = re.compile(FLAG_PATTERNS["role_prefix"].pattern, re.I | re.M)

BLOCK_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "chat_template_tokens": r"<\|[^|>\n]{1,40}\|>|\[/?INST\]|<<\s*/?SYS\s*>>",
        "fake_markup": (
            r"</?\s*(?:tool_result|tool_use|function_calls?|function_results?|system|assistant)\s*>"
        ),
        "fake_tool_json": r"[\"']type[\"']\s*:\s*[\"']tool_(?:use|result)[\"']",
    }.items()
}


def normalize(text: str) -> tuple[str, int]:
    """Canonical form for scanning and downstream use. Returns (text, hidden_chars_removed)."""
    text = unicodedata.normalize("NFKC", text)  # fullwidth and compatibility forms -> ASCII
    text, hidden = _INVISIBLE.subn("", text)
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), hidden


@dataclass(frozen=True, slots=True)
class InjectionResult:
    level: Level
    matches: tuple[str, ...]


def scan(text: str) -> InjectionResult:
    """Scan already-normalised text."""
    blocked = tuple(name for name, p in BLOCK_PATTERNS.items() if p.search(text))
    flagged = tuple(name for name, p in FLAG_PATTERNS.items() if p.search(text))
    if blocked:
        return InjectionResult(Level.BLOCK, blocked + flagged)
    if flagged:
        return InjectionResult(Level.FLAG, flagged)
    return InjectionResult(Level.NONE, ())
