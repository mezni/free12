"""Input and output guardrails (step 7), traced and counted like everything else."""

from __future__ import annotations

from collections.abc import Iterable

from resolve_agent.config import GuardrailSettings
from resolve_agent.guardrails import messages
from resolve_agent.guardrails.injection import Level
from resolve_agent.guardrails.input import Action, Finding, InputGuard, InputResult
from resolve_agent.guardrails.memory import MemoryDecision, MemoryWriteGuard
from resolve_agent.guardrails.output import (
    Evidence,
    OutputGuard,
    OutputResult,
    Severity,
    Violation,
)
from resolve_agent.obs import get_metrics, get_tracer
from resolve_agent.tools.result import ToolResult


class Guardrails:
    """The single entry point for the agent loop. Spans and metrics never contain customer text."""

    def __init__(self, settings: GuardrailSettings | None = None) -> None:
        self.settings = settings or GuardrailSettings()
        self._input = InputGuard(self.settings)
        self._output = OutputGuard(self.settings)
        self._memory = MemoryWriteGuard(self.settings)

    def check_input(self, text: str) -> InputResult:
        with get_tracer().span("guard.input", kind="guardrail") as span:
            result = self._input.check(text)
            span.set_attribute("guard.action", result.action.value)
            span.set_attribute("guard.findings", ",".join(result.kinds))
            span.set_attribute("guard.input_chars", len(text))
            metrics = get_metrics()
            metrics.inc("guardrail.checks", 1.0, guard="input", outcome=result.action.value)
            for finding in result.findings:
                metrics.inc(
                    "guardrail.findings", float(finding.count), guard="input", kind=finding.kind
                )
            return result

    def check_output(
        self,
        reply: str,
        tool_results: Iterable[ToolResult],
        *,
        customer_text: str = "",
        system_prompt: str = "",
    ) -> OutputResult:
        with get_tracer().span("guard.output", kind="guardrail") as span:
            evidence = Evidence.from_results(tool_results, self.settings)
            result = self._output.check(
                reply, evidence, customer_text=customer_text, system_prompt=system_prompt
            )
            span.set_attribute("guard.ok", result.ok)
            span.set_attribute("guard.violations", ",".join(result.codes))
            metrics = get_metrics()
            metrics.inc(
                "guardrail.checks", 1.0, guard="output", outcome="ok" if result.ok else "violation"
            )
            for violation in result.violations:
                metrics.inc("guardrail.findings", 1.0, guard="output", kind=violation.code)
            return result

    def check_memory_write(self, fact: str) -> MemoryDecision:
        with get_tracer().span("guard.memory", kind="guardrail") as span:
            decision = self._memory.check(fact)
            span.set_attribute("guard.allowed", decision.allowed)
            if decision.reason:
                span.set_attribute("guard.reason", decision.reason)
            get_metrics().inc(
                "guardrail.checks",
                1.0,
                guard="memory",
                outcome="allow" if decision.allowed else (decision.reason or "reject"),
            )
            return decision


__all__ = [
    "Action",
    "Evidence",
    "Finding",
    "Guardrails",
    "InputGuard",
    "InputResult",
    "Level",
    "MemoryDecision",
    "MemoryWriteGuard",
    "OutputGuard",
    "OutputResult",
    "Severity",
    "Violation",
    "messages",
]
