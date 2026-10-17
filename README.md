# resolve-agent

Ecommerce customer-support agent in pure Python: policy engine, guardrails, evals, observability.

Scope: returns, exchanges, where is my order.

## Dev setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv, install deps from uv.lock
uv run pytest
uv run ruff check .
uv run mypy
```

Add dependencies with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev).

## Layout

| Package | Responsibility |
|---|---|
| `domain/` | Entities and repositories (SQLite) |
| `policy/` | Business rules as pure functions |
| `tools/` | Typed, validated actions the agent can call |
| `agent/` | Router, agent loop, memory |
| `guardrails/` | Input and output safety checks |
| `obs/` | Tracing, logging, metrics |
| `evals/` | Golden sets, graders, reports |

## Database (step 1)

SQLite via the standard library, integer cents for money, constraints enforced by the schema.

```bash
uv run python -m resolve_agent.domain.seed   # dev only: recreates data/resolve.db with edge-case data
sqlite3 data/resolve.db "select id, status, total_cents from orders"
```

## Policy engine (step 2)

Pure functions in `policy/rules.py` return a `Decision` (allowed, reason code, requires_approval, facts).
The LLM never decides policy; it only phrases a `Decision` via `explain()`. Assumptions are listed
at the top of `rules.py`.

## Tool layer (step 3)

`tools/registry.py` builds JSON schemas from type hints, validates arguments strictly, runs write
tools in one transaction, never raises, and audits every call. `tools/commerce.py` holds the tools.
The customer comes from `ToolContext` (set by the auth layer), never from model arguments.

## Observability (step 4)

`obs/` provides contextvars-based tracing (spans exported to `traces/spans.jsonl`), JSON logs that
carry `trace_id`/`span_id`, and in-process metrics with p50/p95/p99. Call `obs.setup(settings)` once
at startup. Inspect a trace with:

```bash
uv run python -m resolve_agent.obs.render traces/spans.jsonl [trace_id]
```

## LLM client (step 5)

`llm/` has provider-neutral types, `AnthropicClient` (plain `httpx`: retries with jittered backoff
inside a time budget, per-attempt timeouts, SSE streaming, error classification), `FakeLLM` for
tests and evals, and `ObservedLLMClient` (span, token/cost metrics per call). Wrap the real client:

```python
client = ObservedLLMClient(AnthropicClient(settings.llm), prices=settings.llm.prices)
```

## Prompt and model registry (step 6)

Prompts live in `registry/library/<name>/v<N>.md`; `registry/roles.toml` pins which model, parameters
and prompt version each role (router, agent, judge) uses. Every LLM span carries the prompt id and
hash. Released versions are locked and can never be edited:

```bash
uv run python -m resolve_agent.registry              # show active roles and prompt versions
uv run python -m resolve_agent.registry.lock check   # CI: fails on edits or unreleased prompts
uv run python -m resolve_agent.registry.lock update  # release new versions
```

To change a prompt: add `v<N+1>.md`, run `lock update`, then bump `prompt_version` in `roles.toml`.

## Guardrails (step 7)

`guardrails/` holds the input guard (size, hidden characters, injection screening, PII masking),
the output guard (a reply may only state amounts, day counts, ids and outcomes that this turn's tool
results support), and a gate for long-term memory writes. Use the `Guardrails` facade: it traces
every check and counts findings without ever recording customer text. Security limits live in
`GuardrailSettings` (code), not in environment variables. `tests/test_authorization_matrix.py`
proves every id-taking tool refuses another customer's identifiers.

## Intent router (step 8)

`agent/router.py` classifies each (guard-masked) customer message with the small router model and
returns a `Route`: intent, confidence, sentiment, escalation decision, verified order id. Rules in
`agent/rules.py` give an instant path to a person (no model call), a safety net the model cannot
override (anger, legal threats), and a keyword fallback when the model is slow, down or unusable.
The router makes exactly one model call per message; give it an LLM client with a tight retry budget.
