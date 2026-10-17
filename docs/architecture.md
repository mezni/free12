# 🏗 Architecture & Technical Specification — `resolve-agent`

`resolve-agent` is a policy-grounded ecommerce customer-support agent written in
pure, typed Python. It answers three things — **where is my order**, **returns**
and **exchanges** — and, when it cannot, hands the conversation to a human. The
design goal is simple: **the model phrases decisions; code makes them.**

The package has no agent framework dependency. The turn loop, tools, guardrails,
prompt registry, LLM client and observability are all ~10 small modules, which
makes the safety properties inspectable and testable.

## Overview

| Concern               | Choice                                                                 |
| :-------------------- | :--------------------------------------------------------------------- |
| Language / runtime    | Python ≥ 3.11, standard library first                                  |
| LLM                   | Anthropic Messages API over plain `httpx` (no SDK, no framework)       |
| Business rules        | Pure functions in `policy/` — the LLM never decides policy             |
| Persistence           | SQLite (stdlib `sqlite3`), integer cents, schema constraints           |
| Tools                 | Typed functions; JSON schema derived from type hints                   |
| Safety                | Input / output / memory guardrails, enforced in code                   |
| Prompts & models      | Versioned prompt files + pinned roles + a release lock                 |
| Observability         | `contextvars` tracing, JSON logs, in-process metrics                   |
| Quality               | `pytest` · `ruff` · strict `mypy` · GitHub Actions                     |

## Contents

1. [Layered Modules & Turn Lifecycle](#1-layered-modules--turn-lifecycle)
2. [Domain & Persistence](#2-domain--persistence)
3. [Policy Engine](#3-policy-engine)
4. [Tool Layer](#4-tool-layer)
5. [Guardrails](#5-guardrails)
6. [Intent Router](#6-intent-router)
7. [Agent Loop](#7-agent-loop)
8. [LLM Client & Retry](#8-llm-client--retry)
9. [Prompt & Model Registry](#9-prompt--model-registry)
10. [Observability](#10-observability)
11. [Configuration](#11-configuration)
12. [Testing & Quality Gates](#12-testing--quality-gates)

---

## 1. Layered Modules & Turn Lifecycle

Every layer only depends on the layers below it. The LLM sits at the top and is
never trusted with authorization, money or policy.

```text
                     +-----------------------------------------+
   customer text --> |  InputGuard   size · hidden · injection |  masked text
                     |               · PII masking              |
                     +-------------------+---------------------+
                                         |
                                         v
                     +-----------------------------------------+
                     |  IntentRouter   rules shortcut -> small  |
                     |                 model -> rules fallback  |
                     +-------------------+---------------------+
                          escalate / handoff |        answer path
                                             |              |
                                             v              v
                     +-----------------------+   +-----------------------+
                     |     HANDOFF            |   |  SupportAgent loop    |
                     |  escalate_to_human     |   |  model <-> tools      |
                     |  + safe fixed message  |   |  (bounded)            |
                     +-----------------------+   +----------+------------+
                                                            |
                                                            v
                     +-----------------------------------------+
                     |  OutputGuard   evidence-checked reply    |  accept
                     |                amounts · days · ids ·    |  / regenerate
                     |                outcomes · PII · leak     |  / handoff
                     +-------------------+---------------------+
                                         |
                                         v
                                   reply to customer
```

| Layer       | Package / modules                          | Responsibility                                                |
| :---------- | :----------------------------------------- | :------------------------------------------------------------ |
| Domain      | `domain/`                                  | Entities, SQLite schema, repositories, audit log, clock, money |
| Policy      | `policy/`                                  | Pure business rules → `Decision`                               |
| Tools       | `tools/`                                   | Schema-validated, authorized, audited actions                  |
| Guardrails  | `guardrails/`                              | Input screening/masking, output evidence checks, memory gate   |
| Router      | `agent/router.py`, `agent/rules.py`        | Intent, sentiment, escalation decision                         |
| Agent       | `agent/support_agent.py`, `session.py`     | The bounded turn loop and its endings                          |
| LLM         | `llm/`                                     | Provider-neutral types, Anthropic client, fake, observability  |
| Registry    | `registry/`                                | Versioned prompts, pinned roles, release lock                  |
| Obs         | `obs/`                                     | Tracing, structured logs, metrics                              |
| Config      | `config.py`                                | Immutable, typed settings from the environment                 |

---

## 2. Domain & Persistence

SQLite via the standard library. `schema.py` defines **schema version 1**;
integrity is enforced by the database, not only by Python.

```text
customers ──< orders ──< order_items
                │            │
                └──< returns ┴──< refunds
                          (and an append-only audit_log)
```

- **Money** is `Decimal` in Python and integer cents in SQLite (`money.py`); a
  value with more than two decimals is rejected. No `float` anywhere.
- **Constraints in the schema**: status/condition enums as `CHECK`, non-negative
  and positive amounts, `UNIQUE` customer emails, `UNIQUE` refund idempotency
  keys, a delivery-date invariant on delivered orders, and foreign keys.
- **Connections** are autocommit with `PRAGMA foreign_keys=ON`,
  `busy_timeout=5000` and WAL journaling. `transaction()` issues
  `BEGIN IMMEDIATE ... COMMIT` and rolls back on any exception; `atomic()` joins
  an open transaction or starts one, so repositories compose.
- **Repositories** are the only place that talks SQL. Reads used by tools are
  customer-scoped (`get_for_customer`). `RefundRepository.create` is idempotent
  by key and `total_active_for_return` guards against double refunds.
- **Audit log**: tools, returns and refunds append entries
  (`audit_log`), giving a durable trail independent of the trace files.
- **Clock** is injectable (`clock.py`) so time-dependent policy is testable;
  seed data (`seed.py`) is generated relative to "today".

---

## 3. Policy Engine

`policy/rules.py` contains the three rules as **pure functions**: no I/O, no
clock, no environment, no LLM. Callers pass everything the rule needs, including
`today`, so each rule is deterministic and testable at its boundaries.

| Rule                         | Checks                                                                                          |
| :--------------------------- | :---------------------------------------------------------------------------------------------- |
| `check_return_eligibility`   | item belongs to order · order not cancelled · order delivered · not already returned · window · condition |
| `check_exchange_eligibility` | the return rule, then category is exchangeable                                                  |
| `check_refund`               | amount > 0 · amount + already refunded ≤ item value · amount vs auto-approve limit              |

Each returns a `Decision(allowed, reason, requires_approval, facts)` with a
machine-readable `Reason` and the facts behind it. `explain()` renders the
decision with a fixed template — never model text — from the facts only.

**Assumptions (change in one place only):**

- The return window starts at delivery and is **exclusive**: with a 30-day
  window, 29 days after delivery is accepted and 30 days is not.
- A return covers a whole order line (all units of an item).
- Refund approval applies per refund, not cumulatively per order.

**Policy values are code, not configuration.** `PolicySettings` holds the
30-day window, the $50 auto-refund limit and the exchangeable categories; they
are deliberately not read from environment variables so they are reviewed like
code.

---

## 4. Tool Layer

The agent can call exactly seven tools (`tools/commerce.py`):

| Tool                         | Writes | Purpose                                                    |
| :--------------------------- | :----- | :--------------------------------------------------------- |
| `get_order_status`           | –      | Status, tracking and items of one customer order           |
| `list_my_orders`             | –      | Most recent orders when the ID is unknown                  |
| `check_return_eligibility`   | –      | Ask policy without creating anything                       |
| `check_exchange_eligibility` | –      | Ask policy without creating anything                       |
| `create_return`              | ✓      | Start a return, only if policy allows                      |
| `issue_refund`               | ✓      | Refund an approved return; amount computed by the system   |
| `escalate_to_human`          | ✓      | Hand off with a reason and a summary                       |

`ToolRegistry` (`tools/registry.py`) turns a typed function into a tool:

- **Schema from type hints** — parameters may be `str`, `int`, `float`, `bool`,
  `Literal`, an `Enum`, or `X | None`, optionally described with `Annotated[...]`.
  The first parameter must be `ctx: ToolContext`, and a docstring is required
  (its first paragraph is the tool description). `input_schema()` produces the
  exact shape LLM tool APIs expect.
- **Strict validation** — unknown arguments are rejected, required arguments are
  enforced, numbers must not be booleans, choice values must be in range, and
  strings are length-capped (`MAX_STRING_LENGTH = 500`).
- **Safe dispatch** — write tools run inside one transaction; any exception is
  logged and converted to a generic `internal_error` result. **A tool never
  raises to the loop.**
- **Observation** — every call runs in a `tool.<name>` span (with
  `tool.name`, `customer_id`, `request_id`, `tool.ok`, `tool.error_code`) and
  increments `tool.calls{tool,outcome}`. Model-supplied names are never used as
  metric/span labels (unknown tools are labelled `unknown`).

**Authorization and policy live in code, not prompts.** The customer is taken
from `ToolContext` (set by the auth layer), never from model arguments; lookups
are customer-scoped and return the same "not found" whether an id is missing or
belongs to someone else; refund amounts are computed server-side; and eligibility
comes from `policy/`, never from the model.

---

## 5. Guardrails

Three checks, exposed through one `Guardrails` facade that traces and counts
every call (`guardrails/__init__.py`). **No customer text is ever written to a
span, log or metric.**

### Input guard — `InputGuard`

1. **Size** — reject messages longer than `max_input_chars` (2000) before any
   processing.
2. **Normalise** — NFKC-normalise, strip zero-width/bidi/"tags" characters and
   other control characters, collapse whitespace; report how many hidden
   characters were removed.
3. **Injection screening** — phrase matches *flag* a message; structural attacks
   *block* it (see below).
4. **PII masking** — cards (Luhn-validated 13–19 digits), SSNs, Canadian SINs
   (Luhn), emails and phones are replaced with `[CARD]`, `[SSN]`, `[SIN]`,
   `[EMAIL]`, `[PHONE]`. Findings carry counts only, never matched values.

The result is `ALLOW`, `FLAG` (proceed, but treat with suspicion) or `BLOCK`
(do not call the model; the customer gets a fixed reply). The masked,
normalised text is the **only** version allowed to reach the LLM, logger or
storage.

### Prompt-injection screening — `injection.py`

This is a tripwire, not a wall. Real protection comes from the design: tools
enforce authorization and policy in code, so even a successful injection cannot
exceed the customer's own permissions.

| Level | Examples                                                                                          |
| :---- | :------------------------------------------------------------------------------------------------ |
| BLOCK | chat-template tokens (`<\|...\|>`, `[INST]`, `<<SYS>>`), fake tool/system markup, fake tool JSON |
| FLAG  | "ignore previous instructions", "reveal the system prompt", role reassignment, "developer mode", bypass controls, authority claims, `system:`/`assistant:` prefixes |

### Output guard — `OutputGuard`

The prompt asks the model not to invent amounts, rules or outcomes; this check
enforces it. It builds `Evidence` from **this turn's tool results** (amounts,
day counts, ids, and whether a refund was issued, a return approved, or an
escalation performed) and rejects anything unsupported:

| Violation                        | Severity | Meaning                                             |
| :------------------------------- | :------- | :-------------------------------------------------- |
| `unsupported_refund_claim`       | HIGH     | Says a refund was done, but no `issued` refund       |
| `pii_in_reply`                   | HIGH     | A card/SSN/SIN appears in the reply                  |
| `system_prompt_leak`             | HIGH     | Reply overlaps the system prompt (8-word shingles)   |
| `unsupported_amount`             | MEDIUM   | States an amount that no tool result supports        |
| `unsupported_day_count`          | MEDIUM   | States a day count that no tool result supports      |
| `unknown_id`                     | MEDIUM   | Mentions an id not seen in tool results or the message |
| `unsupported_return_claim`       | MEDIUM   | Says a return was approved, but none was              |
| `unsupported_escalation_claim`   | MEDIUM   | Says it escalated, but no escalation happened         |

### Memory guard — `MemoryWriteGuard`

Reserved for long-term memory writes (step 10). It only accepts short, neutral
facts: normalised text, within `max_memory_chars` (300), with no hidden
characters, no PII and nothing that reads like an instruction.

---

## 6. Intent Router

`IntentRouter.route(text)` (`agent/router.py`) answers three questions quickly:
**what does the customer want, how do they feel, and must a person take over?**
It never decides policy and never chooses tools.

```text
text ──► explicit human request? ──yes──► SHORTCUT (no model call)
                 │ no
                 v
          small model call ──► valid JSON? ──no──► RULES (degraded)
                 │ yes
                 v
          safety net: sentiment can only get worse; escalate on anger/legal
```

- **Intents**: `order_status`, `return`, `exchange`, `human_request`,
  `out_of_scope`. **Sentiments**: `calm`, `frustrated`, `angry`.
- **Shortcut** (`RouteSource.SHORTCUT`) — an explicit request for a person is
  routed instantly in plain Python, never waiting on a model.
- **Model** (`RouteSource.LLM`) — a small model returns strict JSON, parsed
  tolerantly of code fences and surrounding prose. Any order id it proposes must
  be well-formed and actually appear in the message.
- **Fallback** (`RouteSource.RULES`, `degraded=True`) — when the model errors or
  returns something unusable, deterministic keyword rules classify instead.
- **Safety net** — `safety_signals()` upgrades sentiment for shouting, "still"/
  "again"/"waiting", anger or legal/fraud language, and the model **can raise
  sentiment but never lower it**. Escalation is a code decision the model can
  raise but not suppress. A confidence below `MIN_CONFIDENCE` (0.5) is treated
  as uncertain.

Rules live in `agent/rules.py`; high-precision patterns keep "the delivery
agent" or "my manager said" from being mistaken for a request for a person.

---

## 7. Agent Loop

`SupportAgent.handle(session, text, ctx)` (`agent/support_agent.py`) runs one
customer message inside a single `agent.turn` span:

```text
input guard ──► router ──► escalate? ──yes──► handoff
                  │ no
                  v
            model / tool loop  (bounded)
                  │
        accepted reply? ──yes──► output guard ──ok──► reply
                  │ no                          │ violation
                  v                             v
              handoff                    regenerate once ──► else handoff
```

`AgentLimits` bounds every turn:

| Limit                      | Default | On breach                    |
| :------------------------- | :------ | :--------------------------- |
| `max_steps`                | 6       | handoff (`step_limit`)       |
| `turn_budget_s`            | 20 s    | handoff (`deadline`)         |
| `max_regenerations`        | 1       | handoff (`output_guard`)     |
| `max_frustrated_turns`     | 2       | handoff (`repeat_frustration`) |
| `max_tool_failures`        | 2       | handoff (`tool_failures`)    |
| `max_tool_calls_per_step`  | 4       | extra calls return `too_many_calls` |

- **Per-call timeout shrinks to the remaining budget**, so a slow model cannot
  outrun the turn deadline.
- **Rejected replies** are dropped (never stored, never sent) and regenerated
  once with a correction that names **only violation codes**, never customer
  text. If it fails again, the turn ends in a handoff.
- **Handoffs are truthful**: every failure path performs a real
  `escalate_to_human` before returning, so the fixed handoff message is never a
  lie. After a handoff the models are no longer called.
- **Session** (`session.py`) holds masked history (what the model sees), tool
  results as output-guard evidence (capped at 50), masked customer messages for
  handoff summaries, tools used, counters, and the escalation reference. One
  request at a time per session.
- **Handoff summaries** (`handoff.py`) are deterministic, masked, bounded to
  500 chars.

`AgentReply` reports the outcome (`answered` / `escalated` / `blocked`), reason,
route, steps, tool calls, regenerations, escalation reference, violation codes
and `trace_id`.

---

## 8. LLM Client & Retry

`llm/types.py` defines provider-neutral types (`Message`, `TextBlock`,
`ToolUseBlock`, `ToolResultBlock`, `LLMRequest`, `LLMResponse`, `Usage`) and a
streaming event model (`TextDelta`, `ToolUseStarted`, `StreamDone`). The
`LLMClient` protocol is `complete()` + `stream()`.

- **`AnthropicClient`** (`anthropic_client.py`) talks to `/v1/messages` over
  plain `httpx`: `x-api-key`, `anthropic-version: 2023-06-01`, per-attempt
  timeouts (connect capped at 5 s), JSON and SSE streaming. **Request bodies
  contain customer text and are never logged or placed in exceptions** — only
  metadata (status, request id, attempt, delay) leaves the module.
- **Retry** (`retry.py`) uses exponential backoff with "equal jitter" (never
  below half the ceiling), honours `Retry-After`, and is bounded by a **total
  time budget**: a third retry after two slow failures is worse than an honest
  error and a handoff. Defaults: 3 attempts, 0.5 s base, 4 s cap, 15 s budget.
- **Errors** (`errors.py`) are classified retryable (rate limit, overloaded,
  server, timeout, connection, interrupted stream) vs terminal (bad request,
  auth, invalid response). A stream that has already delivered events is not
  retried — that would duplicate output.
- **`FakeLLM`** (`fake.py`) plays a script back in order and records every
  request; scripted exceptions simulate real failures.
- **`ObservedLLMClient`** (`observed.py`) wraps any client with an `llm.call`
  span (model, prompt version, purpose, streaming, tool count, tokens, cost,
  stop reason, attempts, request id, first-token latency) and the
  `llm.calls`/`llm.tokens`/`llm.cost_usd` metrics.
- **`build_llm_clients`** (`factory.py`) shares one connection pool and gives
  the router a tighter retry budget (≤ 2 attempts, ≤ 3 s).

---

## 9. Prompt & Model Registry

Prompts are files, not strings in code: `registry/library/<name>/v<N>.md`.
Templates use `string.Template` placeholders (`$name`); rendering is strict —
missing or unexpected variables raise. Each version carries a sha256 and a
`name@vN+short-hash` id stamped on every LLM span, so a silent edit is visible.

`roles.toml` pins a model, parameters and prompt version per role; `Registry`
validates at startup that every role's prompt version exists.

| Role     | Model               | Prompt version | max_tokens | temp | timeout |
| :------- | :------------------ | :------------- | :--------- | :--- | :------ |
| `router` | `claude-haiku-5-5`  | `router@v1`    | 150        | 0.0  | 2.5 s   |
| `agent`  | `claude-sonnet-5-5` | `agent@v1`     | 1024       | 0.2  | 8.0 s   |
| `judge`  | `claude-sonnet-5-5` | `judge@v1`     | 400        | 0.0  | 30.0 s  |

The **release lock** (`registry/prompts.lock.json`) makes released prompt
versions immutable:

```bash
uv run python -m resolve_agent.registry              # show active roles and prompt versions
uv run python -m resolve_agent.registry.lock check   # CI: fails on edits or unreleased prompts
uv run python -m resolve_agent.registry.lock update  # release new versions (never rewrites)
```

To change a prompt: add `v<N+1>.md`, run `lock update`, then point `roles.toml`
at the new version in a reviewed commit.

---

## 10. Observability

- **Tracing** (`obs/tracing.py`) — spans propagate through `contextvars`, so
  nested spans inherit the parent and share its trace id. Spans carry
  `trace_id`, `span_id`, `parent_id`, kind, start, duration and attributes
  (clipped to 1000 chars). Exporters: `InMemoryExporter` (tests),
  `JsonlExporter` (traces on disk) and `MetricsSpanExporter` (latency/errors).
  Exporter failures are swallowed — observability never breaks the app.
- **Logs** (`obs/logs.py`) — one JSON line per event with `ts`, `level`,
  `logger`, `event`, `trace_id`, `span_id` and `fields`; values under known
  secret keys are redacted.
- **Metrics** (`obs/metrics.py`) — labelled counters and bounded histograms with
  p50/p95/p99 (count and mean cover all samples); `snapshot()` exports
  everything. Purely in-process and stdlib-only.
- **Setup** — call `obs.setup(settings)` once at process start to configure
  logging, tracing and metrics; `render.py` prints a trace as an indented tree:

```bash
uv run python -m resolve_agent.obs.render traces/spans.jsonl [trace_id]
```

Key span kind attributes: `agent.turn` (outcome, steps, tool_calls,
regenerations, intent, reason), `router.route` (intent, source, confidence,
sentiment, escalate, degraded), `tool.<name>` (ok, error_code), `llm.call`
(model, prompt version, tokens, cost), `guard.input` / `guard.output` /
`guard.memory`.

---

## 11. Configuration

`Settings.from_env()` (`config.py`) loads immutable, typed settings. Business
policy and security limits live in **code** (`PolicySettings`,
`GuardrailSettings`), not in the environment.

| Variable                 | Default                     | Notes                                   |
| :----------------------- | :-------------------------- | :-------------------------------------- |
| `RESOLVE_ENV`            | `dev`                       | one of `dev`, `staging`, `prod`         |
| `ANTHROPIC_API_KEY`      | –                           | required when `RESOLVE_ENV=prod`        |
| `RESOLVE_DB_PATH`        | `data/resolve.db`           | SQLite file                             |
| `RESOLVE_LOG_LEVEL`      | `INFO`                      |                                         |
| `RESOLVE_LOG_JSON`       | `1`                         | set `0` for human-readable logs         |
| `RESOLVE_TRACE_PATH`     | `traces/spans.jsonl`        | empty disables the JSONL trace file     |
| `RESOLVE_ROLES_PATH`     | packaged `roles.toml`       | override the role file                  |
| `RESOLVE_STORE_NAME`     | `our store`                 | shown in the agent's system prompt      |
| `RESOLVE_LLM_TIMEOUT_S`  | `10.0`                      | per attempt                             |
| `RESOLVE_LLM_BASE_URL`   | `https://api.anthropic.com` |                                         |
| `RESOLVE_LLM_MAX_RETRIES`| `2`                         | must be ≥ 0                             |
| `RESOLVE_LLM_BUDGET_S`   | `15.0`                      | attempts + backoff fit in this budget   |

---

## 12. Testing & Quality Gates

- **Unit tests** (`tests/`, ~550 cases) exercise the schema and repositories,
  money, the policy rules at their boundaries, the tool registry and each
  commerce tool, guardrails (injection, input, output, PII, memory, facade),
  the LLM types/retry/client/fake/observed, the registry and release lock,
  observability, the router (rules, parsing, routing) and the agent loop.
- **End-to-end agent tests** (`tests/agent_helpers.py`) build a `SupportAgent`
  from scripted router/agent `FakeLLM`s over a seeded database, so full turns —
  tool use, output-guard rejection and regeneration, escalation — run offline
  and deterministically.
- **Authorization matrix** (`tests/test_authorization_matrix.py`) proves every
  id-taking tool refuses another customer's identifiers.
- **Isolation** — an autouse fixture resets the global tracer and metrics
  between tests so spans and counters never leak.
- **Static checks** — `ruff` (lint + import sort) and strict `mypy` for the
  `resolve_agent` package; GitHub Actions runs linting and the pytest suite on
  every push and pull request.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

---

*See also: [Product Overview & Requirements](product.md) · [README](../README.md) ·
[Changelog](../CHANGELOG.md)*
