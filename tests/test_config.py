import dataclasses
from decimal import Decimal

import pytest

from resolve_agent.config import Settings


def test_defaults() -> None:
    s = Settings.from_env({})
    assert s.env == "dev"
    assert s.policy.return_window_days == 30
    assert s.policy.auto_refund_limit == Decimal("50")


def test_env_overrides() -> None:
    s = Settings.from_env(
        {"RESOLVE_ENV": "staging", "RESOLVE_LOG_LEVEL": "debug", "RESOLVE_DB_PATH": "/tmp/x.db"}
    )
    assert s.env == "staging"
    assert s.log_level == "DEBUG"
    assert s.db_path == "/tmp/x.db"


def test_policy_is_not_overridable_from_env() -> None:
    s = Settings.from_env({"RESOLVE_RETURN_WINDOW_DAYS": "365"})
    assert s.policy.return_window_days == 30


def test_invalid_env_rejected() -> None:
    with pytest.raises(ValueError, match="RESOLVE_ENV"):
        Settings.from_env({"RESOLVE_ENV": "production"})


def test_prod_requires_api_key() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Settings.from_env({"RESOLVE_ENV": "prod"})


def test_api_key_never_in_repr() -> None:
    s = Settings.from_env({"ANTHROPIC_API_KEY": "sk-secret"})
    assert "sk-secret" not in repr(s)


def test_settings_are_frozen() -> None:
    s = Settings.from_env({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.env = "prod"  # type: ignore[misc]


def test_observability_defaults() -> None:
    s = Settings.from_env({})
    assert s.log_json is True
    assert s.trace_path == "traces/spans.jsonl"


def test_observability_overrides() -> None:
    s = Settings.from_env({"RESOLVE_LOG_JSON": "false", "RESOLVE_TRACE_PATH": ""})
    assert s.log_json is False
    assert s.trace_path == ""


def test_llm_defaults_and_overrides() -> None:
    d = Settings.from_env({}).llm
    assert (d.max_retries, d.base_url, d.prices) == (2, "https://api.anthropic.com", {})
    s = Settings.from_env(
        {
            "RESOLVE_LLM_BASE_URL": "http://localhost:8080/",
            "RESOLVE_LLM_MAX_RETRIES": "0",
            "RESOLVE_LLM_BUDGET_S": "5",
        }
    )
    assert (s.llm.base_url, s.llm.max_retries, s.llm.total_budget_s) == (
        "http://localhost:8080",
        0,
        5.0,
    )


def test_negative_retries_are_rejected() -> None:
    with pytest.raises(ValueError, match="MAX_RETRIES"):
        Settings.from_env({"RESOLVE_LLM_MAX_RETRIES": "-1"})


def test_roles_path_setting() -> None:
    assert Settings.from_env({}).roles_path == ""
    assert (
        Settings.from_env({"RESOLVE_ROLES_PATH": "/etc/roles.toml"}).roles_path == "/etc/roles.toml"
    )


def test_guardrail_settings_are_code_not_environment() -> None:
    s = Settings.from_env({"RESOLVE_MAX_INPUT_CHARS": "99999"})
    assert s.guardrails.max_input_chars == 2000 and s.guardrails.max_memory_chars == 300
