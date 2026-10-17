"""Intent router, agent loop and session state (steps 8-10)."""

from resolve_agent.agent.factory import build_support_agent
from resolve_agent.agent.router import (
    IntentRouter,
    Route,
    RouterOutput,
    RouterOutputError,
    RouteSource,
    parse_router_output,
)
from resolve_agent.agent.rules import Intent, Sentiment
from resolve_agent.agent.session import Session
from resolve_agent.agent.support_agent import AgentLimits, AgentReply, Outcome, SupportAgent

__all__ = [
    "AgentLimits",
    "AgentReply",
    "Intent",
    "IntentRouter",
    "Outcome",
    "Route",
    "RouteSource",
    "RouterOutput",
    "RouterOutputError",
    "Sentiment",
    "Session",
    "SupportAgent",
    "build_support_agent",
    "parse_router_output",
]
