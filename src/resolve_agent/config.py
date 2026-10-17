"""Typed, immutable application settings loaded from environment variables.

Business policy values (return window, refund limit) are deliberately NOT
overridable from the environment: they are versioned code, reviewed like code.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal

VALID_ENVS = ("dev", "staging", "prod")


@dataclass(frozen=True)
class PolicySettings:
    return_window_days: int = 30
    auto_refund_limit: Decimal = Decimal("50")
    exchangeable_categories: frozenset[str] = frozenset({"apparel", "footwear", "accessories"})


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens. Fill these in from your pricing page; nothing is hardcoded."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal


@dataclass(frozen=True)
class LLMSettings:
    api_key: str = field(default="", repr=False)
    timeout_s: float = 10.0  # per attempt; set tighter per request for latency-critical calls
    base_url: str = "https://api.anthropic.com"
    max_retries: int = 2
    total_budget_s: float = 15.0  # attempts plus backoff must fit in this
    prices: Mapping[str, ModelPrice] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    env: str = "dev"
    db_path: str = "data/resolve.db"
    log_level: str = "INFO"
    log_json: bool = True
    trace_path: str = "traces/spans.jsonl"  # empty string disables the JSONL trace file
    roles_path: str = ""  # empty: use the packaged registry/roles.toml
    policy: PolicySettings = field(default_factory=PolicySettings)
    llm: LLMSettings = field(default_factory=LLMSettings)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if environ is None else environ
        defaults = cls()

        env = e.get("RESOLVE_ENV", defaults.env)
        if env not in VALID_ENVS:
            raise ValueError(f"RESOLVE_ENV must be one of {VALID_ENVS}, got {env!r}")

        api_key = e.get("ANTHROPIC_API_KEY", "")
        if env == "prod" and not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when RESOLVE_ENV=prod")

        llm = LLMSettings(
            api_key=api_key,
            timeout_s=float(e.get("RESOLVE_LLM_TIMEOUT_S", defaults.llm.timeout_s)),
            base_url=e.get("RESOLVE_LLM_BASE_URL", defaults.llm.base_url).rstrip("/"),
            max_retries=int(e.get("RESOLVE_LLM_MAX_RETRIES", defaults.llm.max_retries)),
            total_budget_s=float(e.get("RESOLVE_LLM_BUDGET_S", defaults.llm.total_budget_s)),
        )
        if llm.max_retries < 0:
            raise ValueError("RESOLVE_LLM_MAX_RETRIES must be >= 0")
        return cls(
            env=env,
            db_path=e.get("RESOLVE_DB_PATH", defaults.db_path),
            log_level=e.get("RESOLVE_LOG_LEVEL", defaults.log_level).upper(),
            log_json=e.get("RESOLVE_LOG_JSON", "1").lower() not in {"0", "false", "no"},
            trace_path=e.get("RESOLVE_TRACE_PATH", defaults.trace_path),
            roles_path=e.get("RESOLVE_ROLES_PATH", defaults.roles_path),
            llm=llm,
        )
