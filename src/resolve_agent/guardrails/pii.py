"""PII detection and masking. Findings carry counts only, never the matched values.

Cards, Canadian SINs and US SSNs are validated (Luhn / format rules) to limit false positives,
so order IDs and most tracking numbers are left alone. Numeric-only tracking numbers of 13-19
digits that happen to pass the Luhn check will be masked; that is an accepted trade-off.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

TOKENS = {
    "card": "[CARD]",
    "ssn": "[SSN]",
    "sin": "[SIN]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
}
SECRET_KINDS = frozenset({"card", "ssn", "sin"})  # must never appear in a reply or in memory

_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_SSN = re.compile(r"(?<![\w-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w-])")
_SIN = re.compile(r"(?<![\w-])\d{3}[ -]\d{3}[ -]\d{3}(?![\w-])")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<![\w-])(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?![\w-])")


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _is_card(match: str) -> bool:
    digits = _digits(match)
    return 13 <= len(digits) <= 19 and luhn_valid(digits)


def _is_sin(match: str) -> bool:
    return luhn_valid(_digits(match))


# (kind, pattern, validator); order matters: longer and more specific numbers first
_DETECTORS: tuple[tuple[str, re.Pattern[str], Callable[[str], bool]], ...] = (
    ("card", _CARD, _is_card),
    ("ssn", _SSN, lambda _: True),
    ("sin", _SIN, _is_sin),
    ("email", _EMAIL, lambda _: True),
    ("phone", _PHONE, lambda _: True),
)


@dataclass(frozen=True, slots=True)
class MaskResult:
    text: str
    counts: Mapping[str, int]


def mask(text: str, kinds: Iterable[str] | None = None) -> MaskResult:
    wanted = set(kinds) if kinds is not None else set(TOKENS)
    counts: Counter[str] = Counter()
    for kind, pattern, is_valid in _DETECTORS:
        if kind not in wanted:
            continue

        def replace(
            match: re.Match[str], kind: str = kind, valid: Callable[[str], bool] = is_valid
        ) -> str:
            if not valid(match.group(0)):
                return match.group(0)
            counts[kind] += 1
            return TOKENS[kind]

        text = pattern.sub(replace, text)
    return MaskResult(text, dict(counts))


def find_secrets(text: str) -> Mapping[str, int]:
    """Counts of card, SSN and SIN numbers in `text` (values are never returned)."""
    return mask(text, SECRET_KINDS).counts
