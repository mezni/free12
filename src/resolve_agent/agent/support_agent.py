"""The support agent: one customer turn from raw text to a safe reply.

    input guard -> router -> (handoff) or model/tool loop -> output guard -> reply

Everything that can go wrong ends in one of two places: a verified answer, or a human
handoff. The loop is bounded by steps, tool failures and a time budget, and it never sends a
model reply that the output guard rejected.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

from resolve_agent.agent import messages as msgs
from resolve_agent.agent.handoff import build_handoff_summary
from resolve_agent.agent.router import IntentRouter, Route
from resolve_agent.agent.rules import Sentiment
from resolve_agent.agent.session import Session
from resolve_agent.config import Settings
from resolve_agent.guardrails import Action, Guardrails
from resolve_agent.llm.errors import LLMError
from resolve_agent.llm.types import LLMClient, Message, TextBlock, ToolResultBlock, ToolUseBlock
from resolve_agent.obs import get_metrics, get_tracer
from resolve_agent.registry import Registry
from resolve_agent.tools import ToolContext, ToolRegistry
from resolve_agent.tools.commerce import EscalationReason
from resolve_agent.tools.result import ToolResult

MIN_REMAINING_S = 0.5


class Outcome(StrEnum):
    ANSWERED = "answered"
    ESCALATED = "escalated"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class AgentLimits:
    max_steps: int = 6  # model calls per customer turn
    turn_budget_s: float = 20.0
    max_regenerations: int = 1  # retries after the output guard rejects a reply
    max_frustrated_turns: int = 2  # consecutive frustrated messages before a handoff
    max_tool_failures: int = 2  # consecutive internal tool errors before a handoff
    max_tool_calls_per_step: int = 4


@dataclass(frozen=True, slots=True)
class AgentReply:
    text: str
    outcome: Outcome
    reason: str | None = None
    route: Route | None = None
    steps: int = 0
    tool_calls: int = 0
    regenerations: int = 0
    escalation_reference: str | None = None
    violations: tuple[str, ...] = ()
    trace_id: str | None = None


@dataclass(slots=True)
class _Turn:
    messages: list[Message]  # session history + this turn's messages, as sent to the model
    new_start: int  # index of this turn's first message
    dropped: set[int] = field(default_factory=set)  # rejected replies and corrections
    steps: int = 0
    tool_calls: int = 0
    regenerations: int = 0


class SupportAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        router: IntentRouter,
        prompts: Registry,
        tools: ToolRegistry,
        guardrails: Guardrails,
        settings: Settings,
        limits: AgentLimits | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._llm = llm
        self._router = router
        self._prompts = prompts
        self._tools = tools
        self._guardrails = guardrails
        self._settings = settings
        self._limits = limits or AgentLimits()
        self._monotonic = monotonic

    # -- public -------------------------------------------------------------------------

    def handle(self, session: Session, text: str, ctx: ToolContext) -> AgentReply:
        """Handle one customer message. `ctx` carries the authenticated customer."""
        if ctx.customer_id != session.customer_id:
            raise ValueError("tool context and session belong to different customers")
        attributes = {"session_id": session.session_id, "request_id": ctx.request_id}
        with get_tracer().span("agent.turn", kind="agent", attributes=attributes) as span:
            reply = self._handle(session, text, ctx)
            span.set_attribute("agent.outcome", reply.outcome.value)
            span.set_attribute("agent.steps", reply.steps)
            span.set_attribute("agent.tool_calls", reply.tool_calls)
            span.set_attribute("agent.regenerations", reply.regenerations)
            if reply.reason:
                span.set_attribute("agent.reason", reply.reason)
            if reply.route:
                span.set_attribute("agent.intent", reply.route.intent.value)
            metrics = get_metrics()
            metrics.inc("agent.turns", 1.0, outcome=reply.outcome.value)
            metrics.observe("agent.steps", float(reply.steps))
            return replace(reply, trace_id=span.trace_id)

    # -- turn orchestration -------------------------------------------------------------

    def _handle(self, session: Session, text: str, ctx: ToolContext) -> AgentReply:
        session.turns += 1
        checked = self._guardrails.check_input(text)
        if checked.action is Action.BLOCK:
            reason = checked.kinds[0] if checked.kinds else "blocked"
            return AgentReply(checked.reply or "", Outcome.BLOCKED, reason=reason)

        session.customer_messages.append(checked.text)
        if session.escalated:
            reply = msgs.ALREADY_ESCALATED
            return AgentReply(
                reply,
                Outcome.ESCALATED,
                reason="already_escalated",
                escalation_reference=session.escalation_reference,
            )

        route = self._router.route(checked.text)
        if route.sentiment is Sentiment.FRUSTRATED:
            session.frustrated_turns += 1
        else:
            session.frustrated_turns = 0

        turn = _Turn([*session.history, Message.user(checked.text)], len(session.history))
        if route.escalate and route.escalate_reason is not None:
            why = f"router:{route.escalate_reason.value}"
            return self._handoff(
                session, ctx, turn, route, route.escalate_reason, why, checked.notices
            )
        if session.frustrated_turns >= self._limits.max_frustrated_turns:
            return self._handoff(
                session, ctx, turn, route, EscalationReason.FRUSTRATED_CUSTOMER,
                "repeat_frustration", checked.notices,
            )  # fmt: skip
        return self._answer(session, ctx, turn, route, checked.text, checked.notices)

    # -- the model / tool loop ----------------------------------------------------------

    def _answer(
        self,
        session: Session,
        ctx: ToolContext,
        turn: _Turn,
        route: Route,
        customer_text: str,
        notices: Sequence[str],
    ) -> AgentReply:
        limits = self._limits
        deadline = self._monotonic() + limits.turn_budget_s
        role = self._prompts.role("agent")
        variables = {"store_name": self._settings.store_name, "today": ctx.today.isoformat()}
        violations: tuple[str, ...] = ()
        reply_text: str | None = None
        stop = ""

        while True:
            if turn.steps >= limits.max_steps:
                stop = "step_limit"
                break
            remaining = deadline - self._monotonic()
            if remaining < MIN_REMAINING_S:
                stop = "deadline"
                break
            request = self._prompts.request(
                "agent",
                messages=tuple(turn.messages),  # snapshot: requests are immutable
                variables=variables,
                tools=self._tools.schemas(),
                timeout_s=min(role.timeout_s, remaining),
            )
            turn.steps += 1
            try:
                response = self._llm.complete(request)
            except LLMError:
                stop = "llm_unavailable"
                break
            turn.messages.append(response.assistant_message())

            if response.tool_calls:
                self._run_tools(session, ctx, turn, response.tool_calls)
                if session.tool_failures >= limits.max_tool_failures:
                    stop = "tool_failures"
                    break
                continue

            last = len(turn.messages) - 1
            candidate = response.text.strip()
            if response.stop_reason == "max_tokens" or not candidate:
                turn.dropped.add(last)
                stop = "truncated" if candidate else "empty_reply"
                break

            verdict = self._guardrails.check_output(
                candidate,
                session.tool_results,
                customer_text=customer_text,
                system_prompt=request.system,
            )
            if verdict.ok:
                turn.dropped.add(last)  # replaced by a clean text-only message on commit
                reply_text = candidate
                break
            violations = verdict.codes
            turn.dropped.add(last)
            if turn.regenerations < limits.max_regenerations:
                turn.regenerations += 1
                correction = msgs.CORRECTION.format(codes=", ".join(verdict.codes))
                turn.messages.append(Message.user(correction))
                turn.dropped.add(len(turn.messages) - 1)
                continue
            stop = "output_guard"
            break

        if reply_text is not None:
            return self._finish(session, turn, route, reply_text, notices, violations)
        why = stop
        return self._handoff(
            session, ctx, turn, route, EscalationReason.OTHER, why, notices, violations=violations
        )

    def _run_tools(
        self, session: Session, ctx: ToolContext, turn: _Turn, calls: Sequence[ToolUseBlock]
    ) -> None:
        blocks: list[ToolResultBlock] = []
        for index, call in enumerate(calls):
            if index >= self._limits.max_tool_calls_per_step:
                result = ToolResult(
                    False, {}, "too_many_calls", "Too many tool calls at once; call fewer per step."
                )
            else:
                result = self._tools.call(call.name, call.input, ctx)
                turn.tool_calls += 1
            session.record_result(call.name, result)
            if result.error_code == "internal_error":
                session.tool_failures += 1
            elif result.ok:
                session.tool_failures = 0
            if result.data.get("escalated") is True and not session.escalated:
                session.escalated = True
                reference = result.data.get("reference")
                session.escalation_reference = str(reference) if reference else None
            blocks.append(ToolResultBlock(call.id, result.to_llm(), is_error=not result.ok))
        turn.messages.append(Message.tool_results(blocks))

    # -- endings ------------------------------------------------------------------------

    def _commit(self, session: Session, turn: _Turn, final_reply: str) -> None:
        kept = [
            m for i, m in enumerate(turn.messages) if i >= turn.new_start and i not in turn.dropped
        ]
        session.history.extend(kept)
        session.history.append(Message.assistant([TextBlock(final_reply)]))

    def _finish(
        self,
        session: Session,
        turn: _Turn,
        route: Route,
        reply_text: str,
        notices: Sequence[str],
        violations: tuple[str, ...],
    ) -> AgentReply:
        self._commit(session, turn, reply_text)
        escalated_by_model = session.escalated
        return AgentReply(
            "\n\n".join([*notices, reply_text]),
            Outcome.ESCALATED if escalated_by_model else Outcome.ANSWERED,
            reason="model_requested" if escalated_by_model else None,
            route=route,
            steps=turn.steps,
            tool_calls=turn.tool_calls,
            regenerations=turn.regenerations,
            escalation_reference=session.escalation_reference,
            violations=violations,
        )

    def _handoff(
        self,
        session: Session,
        ctx: ToolContext,
        turn: _Turn,
        route: Route,
        reason: EscalationReason,
        why: str,
        notices: Sequence[str],
        *,
        violations: tuple[str, ...] = (),
    ) -> AgentReply:
        if not session.escalated:
            summary = build_handoff_summary(session, route, why)
            result = self._tools.call(
                "escalate_to_human", {"reason": reason.value, "summary": summary}, ctx
            )
            session.escalated = True
            reference = result.data.get("reference") if result.ok else None
            session.escalation_reference = str(reference) if reference else None
        get_metrics().inc("agent.escalations", 1.0, reason=why)

        text = msgs.HANDOFF[self._handoff_kind(why, route)]
        if session.escalation_reference:
            text += " " + msgs.REFERENCE.format(reference=session.escalation_reference)
        self._commit(session, turn, text)
        return AgentReply(
            "\n\n".join([*notices, text]),
            Outcome.ESCALATED,
            reason=why,
            route=route,
            steps=turn.steps,
            tool_calls=turn.tool_calls,
            regenerations=turn.regenerations,
            escalation_reference=session.escalation_reference,
            violations=violations,
        )

    @staticmethod
    def _handoff_kind(why: str, route: Route) -> str:
        if why == "router:customer_request":
            return "requested"
        if why.startswith("router:") or why == "repeat_frustration":
            return "upset"
        if why == "output_guard":
            return "accuracy"
        return "technical"
