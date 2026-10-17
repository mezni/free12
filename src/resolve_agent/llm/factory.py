"""Real LLM clients for production: one shared connection pool, a tight budget for the router."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from resolve_agent.config import Settings
from resolve_agent.llm.anthropic_client import AnthropicClient
from resolve_agent.llm.observed import ObservedLLMClient
from resolve_agent.llm.retry import RetryPolicy
from resolve_agent.llm.types import LLMClient


@dataclass(slots=True)
class LLMClients:
    agent: LLMClient
    router: LLMClient
    _http: httpx.Client

    def close(self) -> None:
        self._http.close()


def build_llm_clients(settings: Settings) -> LLMClients:
    http = httpx.Client()
    llm = settings.llm
    router_policy = RetryPolicy(
        max_attempts=min(2, llm.max_retries + 1),
        base_delay_s=0.2,
        max_delay_s=0.5,
        total_budget_s=min(3.0, llm.total_budget_s),
    )
    agent = AnthropicClient(llm, http_client=http)
    router = AnthropicClient(llm, http_client=http, retry=router_policy)
    return LLMClients(
        agent=ObservedLLMClient(agent, prices=llm.prices),
        router=ObservedLLMClient(router, prices=llm.prices),
        _http=http,
    )
