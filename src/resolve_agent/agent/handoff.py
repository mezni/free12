"""Deterministic handoff summary for the human agent. Built from masked text only."""

from __future__ import annotations

from resolve_agent.agent.router import Route
from resolve_agent.agent.session import Session

MAX_SUMMARY_CHARS = 500  # the escalate_to_human tool's argument limit
RECENT_MESSAGES = 3
MESSAGE_CLIP = 100


def build_handoff_summary(session: Session, route: Route | None, why: str) -> str:
    parts = [f"Why: {why}"]
    if route is not None:
        parts.append(f"intent={route.intent.value} sentiment={route.sentiment.value}")
        if route.order_id:
            parts.append(f"order={route.order_id}")
    if session.tools_used:
        parts.append("tools=" + ",".join(session.tools_used))
    recent = [m[:MESSAGE_CLIP] for m in session.customer_messages[-RECENT_MESSAGES:]]
    if recent:
        parts.append("customer said: " + " / ".join(recent))
    return " | ".join(parts)[:MAX_SUMMARY_CHARS]
