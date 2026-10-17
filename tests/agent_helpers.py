"""Builders for end-to-end agent tests: scripted router and agent models on a seeded database."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from resolve_agent.agent import AgentLimits, IntentRouter, SupportAgent
from resolve_agent.config import Settings
from resolve_agent.guardrails import Guardrails
from resolve_agent.llm import (
    FakeLLM,
    LLMRequest,
    LLMResponse,
    ToolResultBlock,
    text_response,
    tool_call_response,
)
from resolve_agent.registry import Registry
from resolve_agent.tools import ToolRegistry, build_registry

AgentItem = LLMResponse | Exception | Callable[[LLMRequest], LLMResponse]

_REGISTRY = Registry.load()


def route_reply(
    intent: str = "order_status",
    sentiment: str = "calm",
    escalate: bool = False,
    confidence: float = 0.9,
    order_id: str | None = None,
) -> LLMResponse:
    return text_response(
        json.dumps(
            {
                "intent": intent,
                "confidence": confidence,
                "sentiment": sentiment,
                "escalate": escalate,
                "order_id": order_id,
            }
        )
    )


@dataclass
class Rig:
    agent: SupportAgent
    llm: FakeLLM  # the agent model
    router_llm: FakeLLM


def make_agent(
    agent_script: Sequence[AgentItem],
    *,
    router_script: Sequence[AgentItem] | None = None,
    limits: AgentLimits | None = None,
    tools: ToolRegistry | None = None,
    monotonic: Callable[[], float] | None = None,
) -> Rig:
    settings = Settings.from_env({})
    llm = FakeLLM(list(agent_script))
    router_llm = FakeLLM(list(router_script) if router_script is not None else [route_reply()] * 20)
    kwargs = {"monotonic": monotonic} if monotonic else {}
    agent = SupportAgent(
        llm=llm,
        router=IntentRouter(router_llm, _REGISTRY),
        prompts=_REGISTRY,
        tools=tools or build_registry(),
        guardrails=Guardrails(settings.guardrails),
        settings=settings,
        limits=limits,
        **kwargs,  # type: ignore[arg-type]
    )
    return Rig(agent, llm, router_llm)


def tool_result_payloads(request: LLMRequest) -> list[dict]:  # type: ignore[type-arg]
    """All tool results visible to the model in a request, parsed from their JSON content."""
    payloads = []
    for message in request.messages:
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                payloads.append(json.loads(block.content))
    return payloads


def call_with_latest(
    tool: str, key: str, source_key: str, *, tool_id: str = "toolu_x"
) -> Callable[[LLMRequest], LLMResponse]:
    """A scripted model step that reuses a value from an earlier tool result."""

    def respond(request: LLMRequest) -> LLMResponse:
        value = next(
            p[source_key] for p in reversed(tool_result_payloads(request)) if source_key in p
        )
        return tool_call_response(tool, {key: value}, tool_id=tool_id)

    return respond


def history_text(history: Sequence) -> str:  # type: ignore[type-arg]
    return json.dumps([m.to_api() for m in history])
