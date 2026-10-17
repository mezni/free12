"""Attribute names shared by every span, so reports can rely on them."""

from __future__ import annotations


def llm_attributes(
    *,
    model: str,
    prompt_version: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    first_token_ms: float | None = None,
) -> dict[str, object]:
    """Attributes for an LLM-call span (used from step 5 on). Omits anything unknown."""
    values: dict[str, object] = {
        "llm.model": model,
        "llm.prompt_version": prompt_version,
        "llm.input_tokens": input_tokens,
        "llm.output_tokens": output_tokens,
        "llm.cost_usd": cost_usd,
        "llm.first_token_ms": first_token_ms,
    }
    return {k: v for k, v in values.items() if v is not None}
