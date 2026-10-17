"""Input guardrail: size, hidden characters, injection screening, then PII masking.

The returned text is the only version that may be sent to the LLM, stored or logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from resolve_agent.config import GuardrailSettings
from resolve_agent.guardrails import injection, messages, pii


class Action(StrEnum):
    ALLOW = "allow"
    FLAG = "flag"  # proceed, but treat with suspicion and say so in the trace
    BLOCK = "block"  # do not call the LLM; send `reply` to the customer


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str  # e.g. "pii:card", "injection:reveal_prompt", "hidden_characters"
    count: int = 1


@dataclass(frozen=True, slots=True)
class InputResult:
    action: Action
    text: str  # sanitised and masked; empty when blocked
    reply: str | None
    findings: tuple[Finding, ...]
    notices: tuple[str, ...] = ()

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(f.kind for f in self.findings)


class InputGuard:
    def __init__(self, settings: GuardrailSettings | None = None) -> None:
        self._settings = settings or GuardrailSettings()

    def check(self, raw: str) -> InputResult:
        if len(raw) > self._settings.max_input_chars:  # cheap check before any processing
            return self._blocked(messages.TOO_LONG_REPLY, Finding("too_long"))
        text, hidden = injection.normalize(raw)
        findings: list[Finding] = []
        if hidden:
            findings.append(Finding("hidden_characters", hidden))
        if not text:
            return self._blocked(messages.EMPTY_REPLY, *findings, Finding("empty"))

        verdict = injection.scan(text)
        findings.extend(Finding(f"injection:{name}") for name in verdict.matches)
        if verdict.level is injection.Level.BLOCK:
            return self._blocked(messages.BLOCKED_INPUT_REPLY, *findings)

        masked = pii.mask(text)
        findings.extend(Finding(f"pii:{kind}", n) for kind, n in sorted(masked.counts.items()))
        notices = (messages.SENSITIVE_NOTICE,) if pii.SECRET_KINDS & masked.counts.keys() else ()
        suspicious = verdict.level is injection.Level.FLAG or hidden > 0
        return InputResult(
            Action.FLAG if suspicious else Action.ALLOW,
            masked.text,
            None,
            tuple(findings),
            notices,
        )

    @staticmethod
    def _blocked(reply: str, *findings: Finding) -> InputResult:
        return InputResult(Action.BLOCK, "", reply, tuple(findings))
