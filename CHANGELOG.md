# Changelog

All notable changes to **resolve-agent** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`resolve-agent` is built in reviewed steps (see the [README](README.md) and
[architecture](docs/architecture.md)). Steps 1–9 are shipped in `0.1.0`.

---

## [Unreleased]

### Planned

- Long-term memory writes, gated by the existing `MemoryWriteGuard` (step 10)
- Session persistence and multi-turn memory across process restarts
- Evaluation harness: golden sets, deterministic graders and an LLM-as-a-judge
  report driven by the `judge` role (step 13)
- Human-in-the-loop approval for refunds that `requires_approval`
- A production CLI/HTTP entrypoint and container packaging for the agent loop

---

## [0.1.0] - 2026-10-17

First public release: a policy-grounded ecommerce customer-support agent in
pure Python, with ecommerce scope limited to returns, exchanges and
"where is my order". No LangChain/LangGraph dependency: the loop, tools,
guardrails and observability are plain, typed Python.

### Added

#### Domain & persistence (step 1)

- SQLite schema v1 (`src/resolve_agent/domain/schema.py`, `eda6de3`) with
  `customers`, `orders`, `order_items`, `returns`, `refunds` and an `audit_log`;
  constraints (`CHECK`, foreign keys, `UNIQUE`) live in the database, not only in
  Python, and `PRAGMA user_version` tracks the schema version.
- Autocommit connection with `foreign_keys`, `busy_timeout` and WAL, plus
  explicit `transaction()` / re-entrant `atomic()` helpers
  (`domain/db.py`).
- `Decimal` in Python, integer cents in SQLite, never `float`
  (`domain/money.py`).
- Repositories as the only SQL boundary; reads are customer-scoped
  (`get_for_customer`), refund creation is idempotent by key, and every write
  records an audit entry (`domain/repositories.py`).
- Deterministic, edge-case dev data seeded relative to `today`, with an
  injectable `Clock` (`domain/seed.py`, `domain/clock.py`).

#### Policy engine (step 2)

- Business rules as pure functions — no I/O, clock, environment or LLM — in
  `policy/rules.py`: `check_return_eligibility`, `check_exchange_eligibility`
  and `check_refund`.
- Machine-readable `Decision` results (`allowed`, `reason`, `requires_approval`,
  `facts`) with a `Reason` enum and a customer-safe `explain()` renderer
  (`policy/decisions.py`). The LLM never decides policy; it only phrases a
  `Decision`.
- Policy values live in versioned code (`PolicySettings`: 30-day exclusive
  return window, $50 auto-refund limit, `apparel`/`footwear`/`accessories`
  exchangeable), never in environment variables.

#### Tool layer (step 3)

- Registry that derives JSON schemas from type hints (`Annotated`, `Literal`,
  `Enum`, `X | None`), validates arguments strictly, rejects unknown arguments,
  and caps string length (`tools/registry.py`).
- Safe dispatch: write tools run in a single transaction, every call is
  traced, metered and audited, and tools return a `ToolResult` value instead of
  raising (`tools/result.py`).
- The agent's tools (`tools/commerce.py`): `get_order_status`, `list_my_orders`,
  `check_return_eligibility`, `check_exchange_eligibility`, `create_return`,
  `issue_refund`, `escalate_to_human`.
- Authorization and policy enforced in code: the customer comes from
  `ToolContext`, money is computed server-side, and "not yours" looks identical
  to "does not exist".

#### Observability (step 4)

- `contextvars`-based tracing with parent/child spans, `trace_id`/`span_id`,
  bounded attributes, and in-memory/JSONL/metrics exporters
  (`obs/tracing.py`).
- Structured JSON logs carrying `trace_id`/`span_id` with secret redaction
  (`obs/logs.py`).
- In-process labelled counters and bounded histograms with p50/p95/p99
  (`obs/metrics.py`), one-call setup via `obs.setup`, and a trace-tree dev
  renderer (`python -m resolve_agent.obs.render`).

#### LLM client (step 5)

- Provider-neutral request/response/message types and a streaming event model
  (`llm/types.py`).
- `AnthropicClient` on plain `httpx` with jittered exponential backoff bounded
  by a total time budget, per-attempt timeouts, SSE streaming, and error
  classification; request bodies are never logged or raised
  (`llm/anthropic_client.py`, `llm/retry.py`, `llm/errors.py`).
- `FakeLLM` for deterministic tests/evals and `ObservedLLMClient` for
  per-call spans, token/cost metrics and first-token latency
  (`llm/fake.py`, `llm/observed.py`, `llm/cost.py`).
- `build_llm_clients` shares one connection pool and gives the latency-critical
  router a tighter retry budget (`llm/factory.py`).

#### Prompt & model registry (step 6)

- Prompts as versioned files (`registry/library/<name>/v<N>.md`) rendered
  strictly with `string.Template`, each carrying a sha256 and a
  `name@vN+hash` id stamped on every LLM span (`registry/prompts.py`).
- `roles.toml` pins the model, parameters and prompt version per role
  (`router`, `agent`, `judge`); `Registry.load` fails at startup if a role
  points at a missing prompt (`registry/roles.py`, `registry/registry.py`).
- Release lock (`registry/prompts.lock.json`): released prompt versions are
  immutable — `python -m resolve_agent.registry.lock check|update`
  (`registry/lock.py`).

#### Guardrails (step 7)

- `InputGuard`: size limit, Unicode normalisation, hidden/invisible-character
  detection, prompt-injection screening and PII masking; the masked text is the
  only version allowed to reach the LLM, logs or storage
  (`guardrails/input.py`, `guardrails/injection.py`, `guardrails/pii.py`).
- `OutputGuard`: a reply may only state amounts, day counts, ids and outcomes
  that this turn's tool results actually support, and may not leak PII or the
  system prompt (`guardrails/output.py`).
- `MemoryWriteGuard` gate for long-term memory writes (reserved for step 10)
  (`guardrails/memory.py`).
- A `Guardrails` facade that traces and counts every check without ever
  recording customer text (`guardrails/__init__.py`), and an authorization
  matrix test proving every id-taking tool refuses another customer's
  identifiers (`tests/test_authorization_matrix.py`).

#### Intent router (step 8)

- `IntentRouter` classifies each guard-masked message into an intent
  (`order_status`, `return`, `exchange`, `human_request`, `out_of_scope`),
  confidence, sentiment and escalation decision, verifying any order id the
  model returns against the message (`agent/router.py`).
- Deterministic rules (`agent/rules.py`) give an instant path to a person with
  no model call, a safety net the model cannot override (anger, legal threats,
  fraud), and a keyword fallback when the model is slow, down or unusable. The
  model may raise sentiment but never lower it.

#### Agent loop (step 9)

- `SupportAgent.handle` runs one customer turn:
  `input guard -> router -> (handoff) or model/tool loop -> output guard -> reply`,
  all inside one `agent.turn` span (`agent/support_agent.py`).
- Bounded execution: max steps, a per-turn time budget (per-call timeouts shrink
  to the remaining budget), parallel tool calls per step, consecutive tool
  failures and repeated frustration all lead to a real human handoff.
- A reply the output guard rejects is regenerated once with a correction that
  names only violation codes, is never stored and never sent; otherwise the turn
  ends in a handoff.
- Deterministic, masked, bounded handoff summaries
  (`agent/handoff.py`), session state with bounded evidence
  (`agent/session.py`), and `build_support_agent` to wire it all together
  (`agent/factory.py`).

#### Tooling

- `uv` project with a locked dependency set (`pyproject.toml`, `uv.lock`);
  runtime dependency is `httpx` only.
- `ruff` lint/import-sort and strict `mypy` configuration.
- Pytest suite (~550 tests) covering domain, policy, tools, obs, LLM, registry,
  guardrails, router and the end-to-end agent loop.

### Changed

- The repository was repurposed from the earlier `relay-agent` multi-agent
  research prototype into `resolve-agent`. The prototype's modules
  (`src/agents/`, `src/workflows/`, `src/main.py`, `app.py`, `evals/`) remain in
  the tree for reference but are not part of the `resolve_agent` package.

---

## Archived prototype: `relay-agent` (unreleased, 2026-10-17)

Before `resolve-agent`, this repository held an early "multi-agent research and
synthesis engine" prototype built on LangGraph, LangChain and OpenAI
(`Supervisor`/`Researcher`/`Writer` state graph, a DuckDuckGo `web_search` tool,
a SQLite LangGraph checkpointer, a CLI streamer, a Streamlit UI and a small
LLM-as-a-judge eval set). It was superseded by the policy-first support agent
described above and is no longer the project's direction.
