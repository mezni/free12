"""Compose a SupportAgent from settings. Tests pass fakes; production passes real clients."""

from __future__ import annotations

from resolve_agent.agent.router import IntentRouter
from resolve_agent.agent.support_agent import AgentLimits, SupportAgent
from resolve_agent.config import Settings
from resolve_agent.guardrails import Guardrails
from resolve_agent.llm.types import LLMClient
from resolve_agent.registry import Registry
from resolve_agent.tools import build_registry


def build_support_agent(
    settings: Settings,
    *,
    llm: LLMClient,
    router_llm: LLMClient | None = None,
    registry: Registry | None = None,
    limits: AgentLimits | None = None,
) -> SupportAgent:
    prompts = registry or Registry.load(settings)
    return SupportAgent(
        llm=llm,
        router=IntentRouter(router_llm or llm, prompts),
        prompts=prompts,
        tools=build_registry(),
        guardrails=Guardrails(settings.guardrails),
        settings=settings,
        limits=limits,
    )
