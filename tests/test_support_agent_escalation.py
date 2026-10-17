import sqlite3

import pytest

from agent_helpers import history_text, make_agent, route_reply, tool_result_payloads
from resolve_agent.agent import AgentLimits, Outcome, Session
from resolve_agent.config import Settings
from resolve_agent.domain.repositories import AuditLog
from resolve_agent.llm import (
    LLMOverloadedError,
    LLMResponse,
    ToolUseBlock,
    Usage,
    text_response,
    tool_call_response,
)
from resolve_agent.obs import InMemoryExporter, Metrics, set_metrics
from resolve_agent.tools import ToolContext, ToolRegistry, ToolResult
from resolve_agent.tools.commerce import escalate_to_human, get_order_status


@pytest.fixture
def session() -> Session:
    return Session("cus_alice")


def _escalations(conn: sqlite3.Connection) -> list[dict[str, object]]:
    entries = AuditLog(conn).entries("customer", "cus_alice")
    return [e.details for e in entries if e.action == "escalation.requested"]


# --- router-driven handoffs ----------------------------------------------------------------


def test_an_explicit_request_for_a_person_skips_every_model(
    ctx: ToolContext, session: Session, seeded: sqlite3.Connection
) -> None:
    rig = make_agent([], router_script=[])
    reply = rig.agent.handle(session, "I want to speak to a manager about ord_1001", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == "router:customer_request"
    assert reply.text.startswith("Of course.") and "esc_" in reply.text
    assert rig.llm.requests == [] and rig.router_llm.requests == []
    [details] = _escalations(seeded)
    assert details["reason"] == "customer_request"
    assert "order=ord_1001" in str(details["summary"])
    assert session.history[-1].role == "assistant" and session.escalated


def test_later_messages_after_a_handoff_do_not_reach_the_models(
    ctx: ToolContext, session: Session
) -> None:
    rig = make_agent([], router_script=[])
    rig.agent.handle(session, "let me talk to a human", ctx)
    reply = rig.agent.handle(session, "hello? anyone?", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == "already_escalated"
    assert reply.escalation_reference == session.escalation_reference
    assert rig.llm.requests == [] and rig.router_llm.requests == []
    assert session.customer_messages[-1] == "hello? anyone?"


def test_an_angry_customer_is_handed_over_even_if_the_classifier_says_calm(
    ctx: ToolContext, session: Session, seeded: sqlite3.Connection
) -> None:
    rig = make_agent([], router_script=[route_reply(sentiment="calm")])
    reply = rig.agent.handle(session, "This is unacceptable, where is my parcel?", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.text.startswith("I'm sorry")
    assert _escalations(seeded)[0]["reason"] == "frustrated_customer"
    assert rig.llm.requests == []


def test_repeated_frustration_escalates_on_the_second_consecutive_message(
    ctx: ToolContext, session: Session
) -> None:
    rig = make_agent([text_response("Let me check on that for you.")])
    first = rig.agent.handle(session, "I'm still waiting for my parcel", ctx)
    second = rig.agent.handle(session, "still no update, annoyed", ctx)
    assert first.outcome is Outcome.ANSWERED
    assert second.outcome is Outcome.ESCALATED and second.reason == "repeat_frustration"
    assert session.frustrated_turns == 2


def test_a_calm_message_resets_the_frustration_counter(ctx: ToolContext, session: Session) -> None:
    rig = make_agent([text_response("Checking.")] * 3)
    outcomes = [
        rig.agent.handle(session, text, ctx).outcome
        for text in ("still waiting", "where is ord_1005", "still waiting")
    ]
    assert outcomes == [Outcome.ANSWERED] * 3 and session.frustrated_turns == 1


# --- the model asks for a handoff itself ----------------------------------------------------


def test_a_model_requested_handoff_is_honoured_and_final(
    ctx: ToolContext, session: Session, seeded: sqlite3.Connection
) -> None:
    rig = make_agent(
        [
            tool_call_response(
                "escalate_to_human",
                {"reason": "policy_exception", "summary": "Wants to return after the window."},
            ),
            text_response("I've escalated this to a team member who will review it."),
        ]
    )
    reply = rig.agent.handle(session, "my return window passed but the jacket was faulty", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == "model_requested"
    assert reply.escalation_reference and session.escalated
    assert len(_escalations(seeded)) == 1
    again = rig.agent.handle(session, "any news?", ctx)
    assert again.reason == "already_escalated" and len(rig.llm.requests) == 2


def test_claiming_an_escalation_without_the_tool_is_rejected(
    ctx: ToolContext, session: Session, seeded: sqlite3.Connection
) -> None:
    rig = make_agent(
        [
            text_response("I've escalated this to a team member."),
            text_response("I've escalated this to a team member."),
        ]
    )
    reply = rig.agent.handle(session, "can someone review my case", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == "output_guard"
    assert len(_escalations(seeded)) == 1  # the handoff really happened


# --- limits and failures -------------------------------------------------------------------


def test_the_step_limit_ends_in_a_handoff(ctx: ToolContext, session: Session) -> None:
    rig = make_agent(
        [tool_call_response("list_my_orders", {}, tool_id=f"toolu_{i}") for i in range(10)],
        limits=AgentLimits(max_steps=3),
    )
    reply = rig.agent.handle(session, "show me everything", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == "step_limit"
    assert reply.steps == 3 and len(rig.llm.requests) == 3
    roles = [m.role for m in session.history]
    assert roles[0] == "user" and roles[-1] == "assistant"
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False))  # roles alternate


def test_the_time_budget_ends_in_a_handoff(ctx: ToolContext, session: Session) -> None:
    times = iter([0.0, 1.0, 100.0, 100.0, 100.0])
    rig = make_agent(
        [tool_call_response("list_my_orders", {}), text_response("never reached")],
        limits=AgentLimits(turn_budget_s=10.0),
        monotonic=lambda: next(times),
    )
    reply = rig.agent.handle(session, "orders?", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == "deadline"
    assert len(rig.llm.requests) == 1


def test_the_model_timeout_shrinks_to_the_remaining_budget(
    ctx: ToolContext, session: Session
) -> None:
    times = iter([0.0, 5.0, 5.0, 5.0])
    rig = make_agent(
        [text_response("Hello!")],
        limits=AgentLimits(turn_budget_s=6.0),
        monotonic=lambda: next(times),
    )
    rig.agent.handle(session, "hi", ctx)
    assert rig.llm.requests[0].timeout_s == pytest.approx(1.0)  # min(8.0, 6.0 - 5.0)


def test_a_model_outage_ends_in_a_handoff_not_an_error(
    ctx: ToolContext, session: Session, seeded: sqlite3.Connection
) -> None:
    rig = make_agent([LLMOverloadedError("busy")])
    reply = rig.agent.handle(session, "where is ord_1005", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == "llm_unavailable"
    assert reply.text.startswith("I'm having trouble") and "esc_" in reply.text
    assert len(_escalations(seeded)) == 1


def _failing_tools() -> ToolRegistry:
    def crash(ctx: ToolContext, order_id: str) -> ToolResult:
        """Always crash."""
        raise RuntimeError("database on fire")

    crash.__name__ = "get_order_status"
    registry = ToolRegistry()
    registry.add(crash)
    registry.add(escalate_to_human, writes=True)
    return registry


def test_two_consecutive_tool_crashes_end_in_a_handoff(ctx: ToolContext, session: Session) -> None:
    rig = make_agent(
        [
            tool_call_response("get_order_status", {"order_id": "ord_1005"}, tool_id="t1"),
            tool_call_response("get_order_status", {"order_id": "ord_1005"}, tool_id="t2"),
            text_response("never reached"),
        ],
        tools=_failing_tools(),
    )
    reply = rig.agent.handle(session, "where is ord_1005", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == "tool_failures"
    assert len(rig.llm.requests) == 2 and "database on fire" not in history_text(session.history)
    assert tool_result_payloads(rig.llm.requests[1])[0]["error"] == "internal_error"


def test_a_success_resets_the_tool_failure_count(ctx: ToolContext, session: Session) -> None:
    flaky_calls = iter([True, False, True])  # crash, succeed, crash

    def flaky(ctx: ToolContext, order_id: str) -> ToolResult:
        """Crashes on every other call."""
        if next(flaky_calls):
            raise RuntimeError("blip")
        return get_order_status(ctx, order_id)

    flaky.__name__ = "get_order_status"
    tools = ToolRegistry()
    tools.add(flaky)
    tools.add(escalate_to_human, writes=True)
    rig = make_agent(
        [
            tool_call_response("get_order_status", {"order_id": "ord_1005"}, tool_id="t1"),
            tool_call_response("get_order_status", {"order_id": "ord_1005"}, tool_id="t2"),
            tool_call_response("get_order_status", {"order_id": "ord_1005"}, tool_id="t3"),
            text_response("Order ord_1005 has shipped."),
        ],
        tools=tools,
    )
    reply = rig.agent.handle(session, "where is ord_1005", ctx)
    assert reply.outcome is Outcome.ANSWERED and session.tool_failures == 1


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (
            LLMResponse((), "max_tokens", Usage(1, 1), "m"),
            "empty_reply",
        ),
        (
            LLMResponse(
                (
                    __import__("resolve_agent.llm", fromlist=["TextBlock"]).TextBlock(
                        "Your order is"
                    ),
                ),
                "max_tokens",
                Usage(1, 1),
                "m",
            ),
            "truncated",
        ),
        (text_response("   "), "empty_reply"),
    ],
)
def test_truncated_or_empty_replies_are_never_sent(
    ctx: ToolContext, session: Session, response: LLMResponse, reason: str
) -> None:
    rig = make_agent([response])
    reply = rig.agent.handle(session, "where is ord_1005", ctx)
    assert reply.outcome is Outcome.ESCALATED and reply.reason == reason
    assert "Your order is" not in reply.text and "Your order is" not in history_text(
        session.history
    )


def test_excess_parallel_tool_calls_are_capped(ctx: ToolContext, session: Session) -> None:
    six_calls = LLMResponse(
        tuple(ToolUseBlock(f"t{i}", "list_my_orders", {}) for i in range(6)),
        "tool_use",
        Usage(1, 1),
        "m",
    )
    rig = make_agent([six_calls, text_response("Here you go.")])
    reply = rig.agent.handle(session, "orders", ctx)
    assert reply.tool_calls == 4
    results = tool_result_payloads(rig.llm.requests[1])
    assert len(results) == 6 and [r.get("error") for r in results].count("too_many_calls") == 2


def test_the_handoff_summary_is_bounded_and_masked(ctx: ToolContext, session: Session) -> None:
    rig = make_agent([], router_script=[])
    rig.agent.handle(
        session, "card 4111111111111111 ord_1001 " + "x" * 150 + " talk to a human", ctx
    )
    summary = [
        e
        for e in AuditLog(ctx.conn).entries("customer", "cus_alice")
        if e.action == "escalation.requested"
    ][0].details["summary"]
    assert len(str(summary)) <= 500 and "4111" not in str(summary) and "[CARD]" in str(summary)


# --- observability -------------------------------------------------------------------------


def test_a_turn_is_one_trace_with_the_expected_spans(
    ctx: ToolContext, session: Session, spans: InMemoryExporter
) -> None:
    metrics = Metrics()
    set_metrics(metrics)
    rig = make_agent(
        [
            tool_call_response("get_order_status", {"order_id": "ord_1005"}),
            text_response("Order ord_1005 has shipped."),
        ]
    )
    reply = rig.agent.handle(session, "My secret address is 12 Foo Street. Where is ord_1005?", ctx)
    names = [s.name for s in spans.spans]
    assert names.count("llm.call") == 0  # fakes are unwrapped here; guard/router/tool spans exist
    for expected in (
        "guard.input",
        "router.route",
        "tool.get_order_status",
        "guard.output",
        "agent.turn",
    ):
        assert expected in names, expected
    assert len({s.trace_id for s in spans.spans}) == 1 and reply.trace_id == spans.spans[0].trace_id
    root = next(s for s in spans.spans if s.name == "agent.turn")
    assert root.parent_id is None and root.kind == "agent"
    assert root.attributes["agent.outcome"] == "answered" and root.attributes["agent.steps"] == 2
    assert "Foo Street" not in str(root.to_dict())
    assert metrics.counter("agent.turns", outcome="answered") == 1
    summary = metrics.summary("agent.steps")
    assert summary is not None and summary.max == 2.0


def test_escalations_are_counted_by_reason(ctx: ToolContext, session: Session) -> None:
    metrics = Metrics()
    set_metrics(metrics)
    make_agent([], router_script=[]).agent.handle(session, "talk to a human", ctx)
    assert metrics.counter("agent.escalations", reason="router:customer_request") == 1
    assert metrics.counter("agent.turns", outcome="escalated") == 1


def test_settings_store_name_reaches_the_prompt(ctx: ToolContext, session: Session) -> None:
    assert Settings.from_env({"RESOLVE_STORE_NAME": "  Acme  "}).store_name == "Acme"
    assert Settings.from_env({"RESOLVE_STORE_NAME": "  "}).store_name == "our store"
