from resolve_agent.config import LLMSettings, Settings
from resolve_agent.llm import ObservedLLMClient
from resolve_agent.llm.factory import build_llm_clients


def test_clients_are_observed_and_share_one_connection_pool() -> None:
    clients = build_llm_clients(Settings(llm=LLMSettings(api_key="k")))
    assert isinstance(clients.agent, ObservedLLMClient)
    assert isinstance(clients.router, ObservedLLMClient)
    inner_agent, inner_router = clients.agent._inner, clients.router._inner  # type: ignore[attr-defined]
    assert inner_agent._http is inner_router._http  # type: ignore[attr-defined]
    assert inner_router._retry.max_attempts == 2  # type: ignore[attr-defined]
    assert inner_router._retry.total_budget_s <= 3.0  # type: ignore[attr-defined]
    assert inner_agent._retry.total_budget_s == 15.0  # type: ignore[attr-defined]
    clients.close()


def test_a_zero_retry_setting_applies_to_the_router_too() -> None:
    settings = Settings(llm=LLMSettings(api_key="k", max_retries=0))
    clients = build_llm_clients(settings)
    assert clients.router._inner._retry.max_attempts == 1  # type: ignore[attr-defined]
    clients.close()
