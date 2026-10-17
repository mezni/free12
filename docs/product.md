# 📦 Product Overview & Requirements — `resolve-agent`

`resolve-agent` is an ecommerce customer-support agent you can run in
production without trusting it. It resolves the three highest-volume support
requests — **where is my order**, **returns** and **exchanges** — by applying
the store's real policy in code, and it hands the conversation to a human
whenever it cannot answer safely.

## Contents

1. [Vision & Core Value Proposition](#1-vision--core-value-proposition)
2. [Scope & Non-Goals](#2-scope--non-goals)
3. [Target User Personas](#3-target-user-personas)
4. [Key Feature Breakdown](#4-key-feature-breakdown)
5. [How a Conversation Works](#5-how-a-conversation-works)
6. [Build Roadmap](#6-build-roadmap)
7. [Success Metrics](#7-success-metrics)
8. [Safety Principles](#8-safety-principles)

---

## 1. Vision & Core Value Proposition

Ecommerce support teams spend most of their time on a small set of predictable
questions, but automating them traditionally means trusting an LLM to
*decide* — which refunds are allowed, which orders are real, what amounts to
quote. That is where assistants hallucinate refunds, leak data, or over-promise.

**`resolve-agent` separates decision from phrasing.** Business policy, customer
authorization and money are decided by plain Python; the model only explains
the result in warm, natural language. Every claim in a reply is checked against
evidence from that turn's tool calls before it is sent.

| Pillar                          | Why it matters                                                                         |
| :------------------------------ | :------------------------------------------------------------------------------------- |
| 🧾 **Policy in code**           | The LLM never decides eligibility, amounts or approval — so answers are consistent and auditable. |
| 🛡️ **Guardrails both ways**     | Input is screened and PII-masked; output may only state what tool results support.     |
| 🙋 **Truthful handoff**         | Every failure path reaches a real human first, so "I'm connecting you" is never a lie. |
| 🔎 **Observable by default**    | Traces, structured logs and metrics are built in, and never contain customer text.     |
| 🔒 **Authorized, scoped access**| Tools act only on the signed-in customer's own orders; "not yours" looks like "not found". |

---

## 2. Scope & Non-Goals

**In scope**

- Order status, tracking and delivery expectations (WISMO).
- Return eligibility and return creation.
- Exchange eligibility.
- Refunds for approved returns, with manager approval above the auto-refund
  limit.
- Escalation to a human when asked, when the customer is upset, when the request
  is out of scope, or when anything fails.

**Explicit non-goals**

- It does not price, discount, or negotiate.
- It does not change policy; the model can never override a rule.
- It does not act on accounts other than the authenticated customer's.
- It does not invent order details, dates, ids or amounts.

---

## 3. Target User Personas

| Persona                              | Primary needs                                                            | How `resolve-agent` helps                                                |
| :----------------------------------- | :----------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **Support Operations Lead**          | Deflect routine tickets without policy mistakes or angry escalations     | Policy-grounded answers, bounded loops, automatic human handoff          |
| **Compliance / Risk Reviewer**       | Prove what the assistant can and cannot say or do                        | Decisions and authorization in code; output evidence checks; audit log   |
| **AI / Platform Engineer**           | A readable, testable reference for a safe tool-using agent               | Pure-Python modules, provider-neutral LLM client, fakes and evals        |
| **Customer**                         | A fast, correct, polite answer — or a real person                        | Plain-language replies, instant human shortcut, truthful handoff         |

---

## 4. Key Feature Breakdown

### 🧾 Policy engine, not prompt engineering

- Returns, exchanges and refunds are pure functions with explicit reason codes.
- Refunds are computed server-side and either auto-issued or marked
  `pending_approval`.
- Every decision renders to customer-safe language from its own facts.

### 🛡️ Guardrails on both sides of the model

- **Input:** size limits, Unicode normalisation, hidden-character detection,
  prompt-injection screening, and PII masking (cards, SSN/SIN, email, phone).
- **Output:** a reply may only state amounts, day counts, ids and outcomes that
  this turn's tool results support. Unsupported claims trigger a regeneration or
  a handoff — never a sent hallucination.
- **Memory (planned):** long-term memory writes are limited to short, neutral,
  PII-free facts.

### 🔒 Authorization in the tool layer

- The customer is established by the auth layer and injected into `ToolContext`;
  it can never come from model arguments.
- Every lookup is customer-scoped, and the same "not found" is returned whether
  an id is missing or belongs to someone else.

### 🙋 Real, bounded escalation

- An explicit request for a person is routed instantly, without waiting on a
  model.
- Anger, legal threats or fraud language escalate regardless of what the
  classifier says.
- Limits on steps, time, tool failures and repeated frustration end in a
  handoff with a reference the customer can keep.

### 🔎 Observability and prompt governance

- Every turn, route, tool call, LLM call and guardrail check is a span with
  metrics; logs carry trace ids; none of it contains customer text.
- Prompts are versioned files, roles pin model + prompt version, and released
  prompt versions are locked and cannot be edited silently.

---

## 5. How a Conversation Works

1. **Screen** the message, mask sensitive data, and detect manipulation.
2. **Route** it: an explicit "I want a person" goes straight to a human; a
   small model classifies intent and sentiment; keyword rules cover outages.
3. **Answer** through tools: look up orders, check policy, create returns or
   refunds — with the amount and eligibility decided by the system.
4. **Verify** the drafted reply against this turn's tool results.
5. **Reply** — or, if anything is uncertain, unsafe or out of scope, connect the
   customer to a human being.

---

## 6. Build Roadmap

`resolve-agent` is developed in reviewed steps; the [README](../README.md)
walks through each one.

### ✅ Steps 1–9 — Core agent *(Completed)*

- [x] **1. Domain & database** — SQLite schema, integer cents, repositories, audit log.
- [x] **2. Policy engine** — pure functions with reason codes and explanations.
- [x] **3. Tool layer** — schema-validated, authorized, audited, never-raising tools.
- [x] **4. Observability** — tracing, structured logs, metrics.
- [x] **5. LLM client** — provider-neutral types, Anthropic over `httpx`, retries, fake, cost.
- [x] **6. Prompt & model registry** — versioned prompts, pinned roles, release lock.
- [x] **7. Guardrails** — input, output and memory checks; authorization matrix.
- [x] **8. Intent router** — rules, small-model classification, safety net.
- [x] **9. Agent loop** — bounded tool use, guarded replies, truthful handoffs.

### 🔮 Steps 10–13 — Production hardening *(Planned)*

- [ ] **10. Memory** — persist short, verified customer facts behind the memory guard.
- [ ] **11. Session persistence** — multi-turn memory that survives restarts.
- [ ] **12. Entrypoint & packaging** — a CLI/HTTP entrypoint and container image for the loop.
- [ ] **13. Evaluation harness** — golden sets, deterministic graders and an
      LLM-as-a-judge report using the `judge` role.
- [ ] **Human-in-the-loop** — an approval gate for refunds that `requires_approval`.

---

## 7. Success Metrics

| Metric                          | Target                                             | Measured by                                                     |
| :------------------------------ | :------------------------------------------------- | :-------------------------------------------------------------- |
| **Policy adherence**            | 100% of replies consistent with the policy engine  | Output guard (`guardrails/output.py`) + deterministic tests     |
| **Unsupported-claim rate**      | 0 claims sent without tool evidence                | Output guard violations and regenerations (spans/metrics)       |
| **Containment (deflection)**    | ≥ 60% of in-scope tickets resolved without a human | `agent.turns{outcome=answered}` vs `{outcome=escalated}`        |
| **Escalation precision**        | No upset/legal customer left in the bot            | `router.escalations` + `agent.escalations`                      |
| **Turn latency**                | p95 within the 20 s turn budget                    | `span.duration_ms{name=agent.turn}` p95                         |
| **Cost per resolved ticket**    | Tracked and optimised                              | `llm.cost_usd` metric from configured model prices              |

---

## 8. Safety Principles

1. **Code decides, the model explains.** Eligibility, authorization and money
   never come from model output.
2. **Evidence or silence.** If a tool result doesn't support a claim, the claim
   isn't sent.
3. **Least privilege.** The agent can only touch the authenticated customer's
   own orders.
4. **Fail toward a human.** Every uncertain or unsafe path ends in a real
   handoff.
5. **Bounded everything.** Steps, time, tool failures, regenerations and tool
   calls per step all have hard limits.
6. **Prompts are versioned artifacts.** Released versions are immutable and
   every LLM span records the prompt id and hash it ran with.
7. **No customer text in telemetry.** Traces, logs and metrics carry ids and
   codes, never messages or PII.

---

*See also: [Architecture & Technical Specification](architecture.md) ·
[README](../README.md) · [Changelog](../CHANGELOG.md)*
