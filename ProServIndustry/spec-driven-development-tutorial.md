# Spec-Driven Development (SDD): A Hands-On Tutorial

*A learning + skilling guide — from "what is it" to a working Python example that calls a Microsoft Foundry OpenAI model configured entirely through environment variables.*

> **Who this is for:** developers and architects who already write Python and use an AI coding agent (GitHub Copilot, Claude Code, Cursor, etc.), and want a disciplined, repeatable way to ship AI-assisted code.
>
> **What you'll be able to do at the end:** stand up an SDD workflow, write a spec, derive a plan and tasks, implement a small but real feature against a Foundry-hosted model, and **test the spec** so you can prove the implementation matches intent.

---

## Table of Contents

1. [What Spec-Driven Development Is (and Isn't)](#1-what-spec-driven-development-is-and-isnt)
2. [Why It Matters Now](#2-why-it-matters-now)
3. [The Core Workflow: Specify → Plan → Tasks → Implement](#3-the-core-workflow)
4. [Where to Start (Two On-Ramps)](#4-where-to-start-two-on-ramps)
5. [The Worked Example: Customer-Feedback Classifier](#5-the-worked-example)
6. [Step-by-Step Implementation](#6-step-by-step-implementation)
   - 6.1 [Prerequisites](#61-prerequisites)
   - 6.2 [Project layout](#62-project-layout)
   - 6.3 [The Constitution](#63-step-1--the-constitution)
   - 6.4 [The Spec](#64-step-2--the-spec-specify)
   - 6.5 [The Plan](#65-step-3--the-plan-plan)
   - 6.6 [The Tasks](#66-step-4--the-tasks-tasks)
   - 6.7 [Environment variables](#67-step-5--configure-environment-variables)
   - 6.8 [The implementation](#68-step-6--implement)
7. [How to Test the Spec (5 Layers)](#7-how-to-test-the-spec)
8. [Files: Necessary vs. Not](#8-files-necessary-vs-not)
9. [Pros and Cons of SDD](#9-pros-and-cons-of-spec-driven-development)
10. [Common Pitfalls & Best Practices](#10-common-pitfalls--best-practices)
11. [A 1-Week Skilling Plan](#11-a-1-week-skilling-plan)
12. [References](#12-references)

---

## 1. What Spec-Driven Development Is (and Isn't)

**Spec-Driven Development (SDD)** is a methodology where a **structured, version-controlled specification — not the code — is the single source of truth.** You first write a detailed spec describing *what* the system should do and *why*, derive an implementation plan, break it into atomic tasks, and only then generate the code. When requirements change, you **edit the spec and regenerate** the affected code.

This **inverts the traditional flow**. Normally, code is king and the spec rots within a sprint. In SDD, the spec lives in the repo (typically a `specs/` or `.specify/` directory), evolves with the project, and the code is treated as a *regenerable build output* of the spec — much like a `.c` file compiles to a binary.

**It is NOT:**

| Misconception | Reality |
|---|---|
| "It's just Waterfall again." | The feedback loop is **minutes**, not months. You spec → the agent generates in 5–15 min → you review → you fix the spec → regenerate. Waterfall's loop was 3–6 months. |
| "It's the same as TDD." | TDD writes *tests* before code. SDD writes *complete feature specifications* before code. **Tests are derived from the spec.** SDD operates at a broader, strategic level — it defines *what* to build before you worry about *how* to test it. |
| "It's exhaustive paperwork nobody reads." | The goal is **clarity, not bureaucracy.** A spec is a *living contract* that surfaces hidden assumptions early — when changing direction costs a few keystrokes instead of entire sprints. |
| "It's only for big teams." | It helps solo developers too — it gives the AI agent the explicit goals, constraints, and acceptance criteria it otherwise has to guess. |

### SDD vs. Vibe Coding

"Vibe coding" is the informal, conversational style: *"Build a login page."* → *"Now add OAuth."* → *"Make it multi-tenant."* It's fast, creative, and perfect for prototypes — but it lacks centralized requirements, persistent constraints, versioned intent, and acceptance criteria. As a project grows it produces scope creep, conflicting logic, and architecture drift.

| | Spec-Driven | Vibe Coding |
|---|---|---|
| Upfront structure | High | Minimal |
| Time to first output | Slower | Instant |
| Long-term scaling | Faster | Slows down (refactoring) |
| Maintainability | High | Degrades at scale |
| Best for | Production systems, teams, APIs, regulated/AI-generated code | Prototypes, solo experiments, exploration |
| Failure mode | Stale specs | Implementation drift |

---

## 2. Why It Matters Now

SDD emerged in 2025 as a direct response to three failure modes that appeared once LLM coding agents went mainstream:

1. **Intent drift.** A prompt like *"add login"* is wildly underspecified. The model picks reasonable defaults, and those defaults rarely match what the team actually wanted.
2. **Context decay.** As a codebase grows past the agent's effective context window, it forgets older decisions and silently contradicts them.
3. **Unverifiable output.** Without explicit acceptance criteria, there's no way to know whether the agent's code is "right" — code reviews become endless.

A precise spec is the missing layer between human intent and machine execution. The reported payoff is meaningful: industry write-ups cite **roughly 3–10× higher first-pass success rates** and **5–10× fewer "regenerate from scratch" cycles** for teams that adopt the workflow (early-adopter figures from GitHub/AWS — treat as directional, not benchmarks).

> **Reality check:** A 2026 METR study found developers using AI assistants felt ~20% faster but were ~19% slower on real tasks — largely because vibe-coded work shifts effort to a later debugging cycle. SDD's value proposition is **speed *later*** (less rework, safer change), bought with **front-loaded effort** writing the spec.

---

## 3. The Core Workflow

SDD has four phases, **each with a human checkpoint** between them. Every phase produces a Markdown artifact that becomes structured context for the next.

```
┌──────────┐    ┌────────┐    ┌────────┐    ┌────────────┐
│ SPECIFY  │ -> │  PLAN  │ -> │ TASKS  │ -> │ IMPLEMENT  │
│  what &  │    │  how / │    │ atomic │    │  agent     │
│   why    │    │  stack │    │ items  │    │  writes    │
└──────────┘    └────────┘    └────────┘    └────────────┘
   spec.md       plan.md      tasks.md       code + tests
      ▲              ▲             ▲               │
      └──────────────┴─────────────┴───────────────┘
         If reality disagrees with the spec, fix the SPEC, then regenerate.
```

1. **Specify** — A clear description of *what* you're building and *why*: problem, users/personas, key flows, success metrics, constraints, **acceptance criteria**, edge cases, and explicit **non-goals**. Focus on *what* and *why* — avoid implementation detail here. The agent can draft this; you refine it. Ambiguities get flagged with `[NEEDS CLARIFICATION: ...]` markers you must resolve.
2. **Plan** — Technical constraints: stack, patterns, architecture, data models. The agent produces an implementation plan from the spec + constraints.
3. **Tasks** — The plan is broken into concrete, **testable** work items. Each task has clear inputs, expected outputs, and validation criteria.
4. **Implement** — The agent works through tasks systematically, using the spec and plan as context for every decision.

A fifth, cross-cutting artifact is the **Constitution** — project-wide, non-negotiable rules (security, performance, compliance, coding standards) that every plan and task must honor.

---

## 4. Where to Start (Two On-Ramps)

You do **not** need a special tool to do SDD — any well-structured Markdown spec works. But tooling automates the scaffolding.

### On-ramp A — Manual (recommended for learning)
Create a `specs/` folder, hand-write `constitution.md`, `spec.md`, `plan.md`, `tasks.md`, and feed them to your AI agent as context. This is what the rest of this tutorial does, so you understand every moving part.

### On-ramp B — GitHub Spec Kit (recommended for real projects)
[GitHub Spec Kit](https://github.com/github/spec-kit) is an open-source, model-agnostic toolkit that ships the Spec → Plan → Tasks → Implement process with templates, quality checklists, and cross-artifact analysis. It integrates with 30+ agents (Copilot, Claude, Gemini, Cursor, etc.).

```bash
# Install the Specify CLI (requires uv: https://docs.astral.sh/uv/)
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git

# Scaffold a project wired for your agent of choice
specify init my-project --integration copilot
cd my-project
```

You then drive it from inside your agent with slash commands: `/speckit.specify`, `/speckit.plan`, `/speckit.tasks`, `/speckit.implement`. Each command generates/updates the corresponding Markdown artifact.

> Microsoft Learn also publishes a free 3-hour module, *"Implement spec-driven development using the GitHub Spec Kit,"* which walks an enterprise brownfield scenario end-to-end. (Link in [References](#12-references).)

The example below uses **On-ramp A** so nothing is hidden, but the artifacts map 1:1 onto Spec Kit's files.

---

## 5. The Worked Example

**Feature: a Customer-Feedback Classifier.** Given a free-text customer comment, return a structured classification using a Microsoft Foundry OpenAI model.

Why this example: it's tiny enough to finish in one sitting, but it has **real, testable acceptance criteria** (a fixed output schema, a closed label set, latency and cost constraints) — exactly what makes a spec verifiable.

**Output contract (what "done" looks like):**
```json
{
  "sentiment": "positive | negative | neutral",
  "category": "product | support | billing | other",
  "reason": "a one-sentence justification (<= 200 chars)",
  "confidence": 0.0
}
```

---

## 6. Step-by-Step Implementation

### 6.1 Prerequisites

- **Python 3.9+**
- A **Microsoft Foundry project** with a chat model deployed (e.g., a GPT‑class deployment). Note your **project endpoint** — it looks like `https://<your-resource>.services.ai.azure.com/api/projects/<your-project>`.
- One of:
  - **Azure CLI** logged in (`az login`) for Entra ID auth — recommended, no keys in code; **or**
  - An **API key** for the resource (simpler to start, less secure).
- Packages:

```bash
python -m pip install --upgrade \
  "azure-ai-projects>=2.0.0" azure-identity openai python-dotenv pydantic pytest
```

### 6.2 Project layout

```
feedback-classifier/
├── specs/
│   ├── constitution.md          # project-wide rules (the "why we won't break")
│   ├── spec.md                  # WHAT & WHY  (Specify phase)
│   ├── plan.md                  # HOW / stack (Plan phase)
│   └── tasks.md                 # atomic work items (Tasks phase)
├── src/
│   └── feedback_classifier/
│       ├── __init__.py
│       ├── client.py            # builds the Foundry/OpenAI client from env vars
│       ├── models.py            # the output contract (pydantic)
│       └── classifier.py        # the feature itself
├── tests/
│   ├── test_contract.py         # output-schema / contract tests
│   ├── test_spec_acceptance.py  # one test per acceptance criterion (mocked)
│   ├── test_classifier_unit.py  # logic/prompt-assembly unit tests (mocked)
│   └── test_live_integration.py # optional: hits the real model (marked "live")
├── .env.example                 # template — committed
├── .env                         # real secrets — NEVER committed
├── .gitignore
├── requirements.txt
└── pytest.ini
```

---

### 6.3 Step 1 — The Constitution

`specs/constitution.md` — the non-negotiables every later artifact must respect.

```markdown
# Project Constitution — Feedback Classifier

These rules are binding on every spec, plan, task, and generated line of code.

## Security
- C-1: Secrets (endpoints, keys) MUST come from environment variables. No secret is hard-coded or committed.
- C-2: Prefer Entra ID (DefaultAzureCredential) over API keys when available.
- C-3: Customer feedback text is treated as untrusted input; never `eval`/exec it.

## Reliability
- C-4: Every model call MUST have a timeout and a bounded retry (max 2 retries, exponential backoff).
- C-5: The function MUST always return a valid object matching the output contract, or raise a typed error — never partial/garbage output.

## Quality
- C-6: Output MUST validate against the JSON schema in spec.md before being returned.
- C-7: Public functions MUST have type hints and docstrings.
- C-8: Every functional requirement (FR-xxx) MUST map to at least one automated test.

## Cost / Performance
- C-9: Use the smallest model that meets accuracy targets; model name comes from config, not code.
- C-10: P95 latency target: < 4 seconds per classification.
```

---

### 6.4 Step 2 — The Spec (`/specify`)

`specs/spec.md` — *what* and *why*. Note the use of **EARS-style** requirement phrasing ("The system MUST/SHALL, WHEN <trigger>, …"), which turns fuzzy requirements into testable, AI-parseable statements, and the **`[NEEDS CLARIFICATION]`** markers that force decisions before you proceed.

```markdown
# Spec: Customer-Feedback Classifier

## Problem
Support and product teams receive thousands of free-text customer comments.
Manually tagging each by sentiment and topic is slow and inconsistent. We need an
automated classifier that produces a consistent, structured label for each comment.

## Users / Personas
- Support analyst: triages incoming tickets by sentiment + category.
- Product manager: aggregates feedback themes for planning.

## Goals
- Turn one free-text comment into one structured classification.
- Consistent, machine-readable output that downstream tools can aggregate.

## Non-Goals (explicitly out of scope)
- Multi-language translation (English input only for v1).
- Storing or persisting feedback (caller owns storage).
- A UI or REST service (library function only for v1).

## Functional Requirements
- **FR-001**: The system MUST accept a single English text comment (1–5,000 chars)
  and return one classification object.
- **FR-002**: The system MUST classify `sentiment` as exactly one of:
  `positive`, `negative`, `neutral`.
- **FR-003**: The system MUST classify `category` as exactly one of:
  `product`, `support`, `billing`, `other`.
- **FR-004**: The system MUST return a `reason`: a single sentence (<= 200 chars)
  justifying the labels.
- **FR-005**: The system MUST return a `confidence` float in the range [0.0, 1.0].
- **FR-006**: WHEN the input is empty or whitespace-only, the system MUST raise a
  `ValueError` (it MUST NOT call the model).
- **FR-007**: WHEN the model returns malformed or non-conforming output, the system
  MUST retry up to 2 times; if still invalid, it MUST raise `ClassificationError`.
- **FR-008**: The model deployment name and endpoint MUST be read from environment
  variables, never hard-coded.

## Non-Functional Requirements
- **NFR-001 (latency)**: P95 end-to-end latency < 4 seconds per call.
- **NFR-002 (security)**: No secrets in source or logs (see Constitution C-1, C-2).
- **NFR-003 (determinism)**: Use temperature <= 0.2 for reproducible labels.

## Output Contract (authoritative schema)
```json
{
  "sentiment": "positive|negative|neutral",
  "category": "product|support|billing|other",
  "reason": "string, <= 200 chars",
  "confidence": 0.0
}
```

## Acceptance Criteria (Given/When/Then — these become tests)
- **AC-1**: GIVEN "I love the new dashboard, it's so fast!" WHEN classified
  THEN sentiment == "positive" AND category == "product".
- **AC-2**: GIVEN "I was double-charged on my invoice this month." WHEN classified
  THEN sentiment == "negative" AND category == "billing".
- **AC-3**: GIVEN "Your support agent reset my password quickly." WHEN classified
  THEN category == "support".
- **AC-4**: GIVEN "   " (whitespace) WHEN classified THEN a ValueError is raised.
- **AC-5**: GIVEN any valid comment WHEN classified THEN the result validates
  against the Output Contract and confidence is within [0.0, 1.0].

## Open Questions
- [NEEDS CLARIFICATION: should "feature request" be its own category, or "other"?]
  → RESOLVED: fold into "product" for v1.
```

> The `[NEEDS CLARIFICATION]` → `RESOLVED` pattern is the heartbeat of SDD: ambiguity is made visible and killed *before* code exists.

---

### 6.5 Step 3 — The Plan (`/plan`)

`specs/plan.md` — *how*. This is where implementation choices live (and where the Constitution is enforced).

```markdown
# Implementation Plan: Customer-Feedback Classifier

## Stack
- Language: Python 3.9+
- Model access: Microsoft Foundry via `azure-ai-projects` (AIProjectClient ->
  get_openai_client()). Fallback provider: direct Azure OpenAI via the `openai`
  SDK's AzureOpenAI client. Provider chosen by env var `MODEL_PROVIDER`.
- Validation: pydantic v2 for the output contract.
- Tests: pytest.

## Architecture
- `client.py`  — pure config/wiring. Reads env vars, returns (client, model_name).
                 No business logic. (satisfies FR-008, C-1, C-2)
- `models.py`  — pydantic `Classification` model = the Output Contract. Enforces
                 enums for sentiment/category and the confidence range. (FR-002..005)
- `classifier.py` — `classify(comment: str) -> Classification`. Validates input
                 (FR-006), builds the prompt, calls the model with temp<=0.2
                 (NFR-003), parses + validates JSON, retries on failure (FR-007),
                 times out per call (C-4).

## Key Decisions
- Force JSON via `response_format={"type": "json_object"}` and a strict system prompt.
- Retry loop: max 2 retries, exponential backoff (C-4, FR-007).
- The label sets live in ONE place (models.py enums) and are injected into the
  prompt, so spec, prompt, and validation can never silently diverge.

## Constitution check
- C-1/C-2/C-9 → satisfied in client.py (env-driven, Entra-first).
- C-5/C-6     → satisfied by pydantic validation + typed errors in classifier.py.
- C-8         → every FR mapped to a test in tests/ (see tasks.md).
```

---

### 6.6 Step 4 — The Tasks (`/tasks`)

`specs/tasks.md` — atomic, testable work items, each traced back to a requirement.

```markdown
# Tasks: Customer-Feedback Classifier

| ID  | Task                                                        | Satisfies        | Test |
|-----|-------------------------------------------------------------|------------------|------|
| T1  | Define `Classification` pydantic model + enums              | FR-002..005, C-6 | test_contract.py |
| T2  | Implement `build_client()` from env vars (Foundry + AOAI)   | FR-008, C-1/2/9  | test_classifier_unit.py |
| T3  | Implement input validation (reject empty/whitespace)        | FR-006           | AC-4 |
| T4  | Implement prompt assembly with injected label sets          | FR-002/003       | test_classifier_unit.py |
| T5  | Implement model call (temp<=0.2, JSON mode, timeout)        | NFR-003, C-4     | test_live_integration.py |
| T6  | Implement parse + validate + retry (max 2)                  | FR-007, C-5      | test_spec_acceptance.py |
| T7  | Map every acceptance criterion AC-1..5 to a test            | C-8              | test_spec_acceptance.py |
```

> When you hand `constitution.md` + `spec.md` + `plan.md` + `tasks.md` to your AI agent and say *"implement T1–T7,"* the agent has everything it needs to produce the code below with minimal drift. The code in 6.8 is exactly what that produces.

---

### 6.7 Step 5 — Configure Environment Variables

Everything about the model is configured through env vars — **no endpoint, key, or deployment name ever appears in code** (FR-008, C-1).

`.env.example` (committed — a template with **no real values**):

```bash
# ── Choose how to reach the model: "foundry" (default) or "azure_openai" ──
MODEL_PROVIDER=foundry

# ── Option A: Microsoft Foundry project (recommended) ──
# Endpoint format: https://<resource>.services.ai.azure.com/api/projects/<project>
FOUNDRY_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
# The name of the model DEPLOYMENT in your Foundry project (not the base model name)
FOUNDRY_MODEL_DEPLOYMENT=gpt-4o-mini
# Foundry uses Entra ID — run `az login` first. No key needed.

# ── Option B: Direct Azure OpenAI (used when MODEL_PROVIDER=azure_openai) ──
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=replace-me
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# ── Tuning (optional; safe defaults baked into code) ──
MODEL_TEMPERATURE=0.1
MODEL_TIMEOUT_SECONDS=30
```

`.env` (your real values — **git-ignored, never committed**). Copy the example and fill it in:

```bash
cp .env.example .env
# then edit .env with your real endpoint / deployment / key
```

`.gitignore` (so secrets can't leak):

```gitignore
.env
__pycache__/
*.pyc
.pytest_cache/
.venv/
```

> **Why two providers?** "Microsoft Foundry OpenAI model" can be reached two ways: through the **Foundry project SDK** (`azure-ai-projects`, Entra ID, no keys) or the **direct Azure OpenAI endpoint** (`openai` SDK's `AzureOpenAI`, key or Entra). The code supports both; flip `MODEL_PROVIDER` to switch. Both are configured *only* through env vars.

---

### 6.8 Step 6 — Implement

#### `src/feedback_classifier/models.py`  (T1 — the contract)

```python
"""Output contract for the classifier. This pydantic model IS the spec's schema."""
from enum import Enum
from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class Category(str, Enum):
    product = "product"
    support = "support"
    billing = "billing"
    other = "other"


class Classification(BaseModel):
    """Matches the Output Contract in spec.md (FR-002..FR-005)."""
    sentiment: Sentiment
    category: Category
    reason: str = Field(..., max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)
```

#### `src/feedback_classifier/client.py`  (T2 — env-driven wiring)

```python
"""Builds a model client purely from environment variables (FR-008, C-1/C-2/C-9).

Returns a tuple of (openai_client, model_name). The returned client always exposes
the standard OpenAI `.chat.completions.create(...)` surface, whether it came from
Microsoft Foundry or a direct Azure OpenAI endpoint.
"""
import os
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()  # load .env if present


class ConfigError(RuntimeError):
    """Raised when required environment variables are missing."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def build_client() -> Tuple[object, str]:
    provider = os.getenv("MODEL_PROVIDER", "foundry").lower()

    if provider == "foundry":
        # Microsoft Foundry project SDK — Entra ID auth (run `az login`).
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        endpoint = _require("FOUNDRY_PROJECT_ENDPOINT")
        model = _require("FOUNDRY_MODEL_DEPLOYMENT")
        project = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )
        client = project.get_openai_client()  # standard OpenAI-shaped client
        return client, model

    if provider == "azure_openai":
        # Direct Azure OpenAI endpoint — API key (or Entra) auth.
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
            api_key=_require("AZURE_OPENAI_API_KEY"),
            api_version=_require("AZURE_OPENAI_API_VERSION"),
        )
        model = _require("AZURE_OPENAI_DEPLOYMENT")
        return client, model

    raise ConfigError(f"Unknown MODEL_PROVIDER: {provider!r}")
```

#### `src/feedback_classifier/classifier.py`  (T3–T6 — the feature)

```python
"""The feature: classify(comment) -> Classification.

Implements input validation (FR-006), env-driven model config (FR-008),
deterministic JSON output (NFR-003), and validate-with-retry (FR-007, C-5).
"""
import json
import os
import time
from typing import Optional, Tuple

from pydantic import ValidationError

from .client import build_client
from .models import Category, Classification, Sentiment

_LABELS = (
    f"sentiment must be one of {[s.value for s in Sentiment]}; "
    f"category must be one of {[c.value for c in Category]}."
)

SYSTEM_PROMPT = (
    "You are a strict customer-feedback classifier. "
    f"{_LABELS} "
    "Return ONLY a JSON object with keys: sentiment, category, reason, confidence. "
    "reason is one sentence <= 200 chars. confidence is a float between 0 and 1. "
    "Do not include any text outside the JSON object."
)

MAX_RETRIES = 2


class ClassificationError(RuntimeError):
    """Raised when the model cannot produce conforming output (FR-007)."""


def _call_model(client, model: str, comment: str) -> str:
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    timeout = float(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,           # NFR-003: deterministic labels
        timeout=timeout,                    # C-4: bounded call
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": comment},
        ],
    )
    return resp.choices[0].message.content


def classify(
    comment: str,
    *,
    _client: Optional[Tuple[object, str]] = None,
) -> Classification:
    """Classify one English customer comment.

    Args:
        comment: free-text comment, 1–5,000 chars.
        _client: optional (client, model) tuple for tests; built from env if None.

    Returns:
        A validated `Classification`.

    Raises:
        ValueError: if the comment is empty/whitespace (FR-006).
        ClassificationError: if the model never returns conforming output (FR-007).
    """
    if comment is None or not comment.strip():        # FR-006
        raise ValueError("comment must be non-empty, non-whitespace text")
    if len(comment) > 5000:                           # FR-001 bound
        raise ValueError("comment exceeds 5,000 character limit")

    client, model = _client if _client is not None else build_client()

    last_error = None
    for attempt in range(MAX_RETRIES + 1):            # FR-007: up to 2 retries
        try:
            raw = _call_model(client, model, comment)
            data = json.loads(raw)
            return Classification.model_validate(data)  # FR-002..005, C-6
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)              # C-4: exponential backoff
    raise ClassificationError(
        f"Model failed to produce conforming output after "
        f"{MAX_RETRIES + 1} attempts: {last_error}"
    )
```

#### `src/feedback_classifier/__init__.py`

```python
from .classifier import ClassificationError, classify
from .models import Category, Classification, Sentiment

__all__ = ["classify", "ClassificationError", "Classification", "Sentiment", "Category"]
```

#### Run it

```python
# quickstart.py
from feedback_classifier import classify

result = classify("I love the new dashboard, it's so fast!")
print(result.model_dump_json(indent=2))
```

```bash
az login                       # if using the Foundry provider
python quickstart.py
```

---

## 7. How to Test the Spec

This is the part most tutorials skip. **Testing the spec means proving the implementation satisfies every requirement and acceptance criterion** — and catching the moment the code and spec drift apart. Use five layers, cheapest/fastest first.

### Layer 1 — Contract tests (schema conformance)
Prove the output *shape* always matches the Output Contract. No model needed.

`tests/test_contract.py`:
```python
import pytest
from pydantic import ValidationError
from feedback_classifier.models import Classification


def test_valid_object_passes():
    obj = Classification(sentiment="positive", category="product",
                         reason="Great speed.", confidence=0.9)
    assert obj.sentiment.value == "positive"


def test_confidence_out_of_range_rejected():   # FR-005
    with pytest.raises(ValidationError):
        Classification(sentiment="neutral", category="other",
                       reason="x", confidence=1.4)


def test_unknown_label_rejected():             # FR-002 / FR-003
    with pytest.raises(ValidationError):
        Classification(sentiment="furious", category="product",
                       reason="x", confidence=0.5)


def test_reason_length_capped():               # FR-004
    with pytest.raises(ValidationError):
        Classification(sentiment="negative", category="billing",
                       reason="x" * 201, confidence=0.5)
```

### Layer 2 — Acceptance tests (one per AC, mocked model)
Map **every** acceptance criterion in `spec.md` to a test. Mock the model so these are fast, free, and deterministic — they test *your* logic (validation, parsing, retry), not the model's accuracy.

`tests/test_spec_acceptance.py`:
```python
import json
import pytest
from feedback_classifier import classify
from feedback_classifier.classifier import ClassificationError


class FakeClient:
    """Stands in for the OpenAI client; returns a scripted JSON payload."""
    def __init__(self, payload):
        self._payload = payload
        self.chat = self  # so .chat.completions.create resolves
        self.completions = self

    def create(self, **kwargs):
        msg = type("M", (), {"content": self._payload})
        choice = type("C", (), {"message": msg})
        return type("R", (), {"choices": [choice]})


def _client_returning(payload):
    return (FakeClient(payload), "fake-model")


def test_ac1_positive_product():
    payload = json.dumps({"sentiment": "positive", "category": "product",
                          "reason": "Loves dashboard speed.", "confidence": 0.95})
    r = classify("I love the new dashboard, it's so fast!",
                 _client=_client_returning(payload))
    assert r.sentiment.value == "positive"
    assert r.category.value == "product"


def test_ac2_negative_billing():
    payload = json.dumps({"sentiment": "negative", "category": "billing",
                          "reason": "Double charged.", "confidence": 0.9})
    r = classify("I was double-charged on my invoice this month.",
                 _client=_client_returning(payload))
    assert r.sentiment.value == "negative"
    assert r.category.value == "billing"


def test_ac4_whitespace_raises():               # FR-006
    with pytest.raises(ValueError):
        classify("   ", _client=_client_returning("{}"))


def test_ac5_result_validates_and_confidence_in_range():
    payload = json.dumps({"sentiment": "neutral", "category": "other",
                          "reason": "Mixed.", "confidence": 0.5})
    r = classify("It's fine I guess.", _client=_client_returning(payload))
    assert 0.0 <= r.confidence <= 1.0


def test_fr007_retry_then_fail_on_garbage():    # FR-007
    with pytest.raises(ClassificationError):
        classify("anything", _client=_client_returning("not json at all"))
```

### Layer 3 — Unit tests (logic in isolation)
Test prompt assembly, env-var wiring, and edge cases (`build_client` raising on missing vars, oversized input rejected, etc.).

### Layer 4 — Live integration tests (real model, opt-in)
Hit the **real Foundry model** to check accuracy against the golden examples. These are slow and cost tokens, so gate them behind a marker so they don't run in normal CI.

`tests/test_live_integration.py`:
```python
import os
import pytest
from feedback_classifier import classify

live = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to run tests that call the real model",
)


@live
def test_live_positive_product():               # AC-1 against the real model
    r = classify("I love the new dashboard, it's so fast!")
    assert r.sentiment.value == "positive"
    assert r.category.value == "product"


@live
def test_live_billing_detected():               # AC-2
    r = classify("I was double-charged on my invoice this month.")
    assert r.category.value == "billing"
```

`pytest.ini`:
```ini
[pytest]
pythonpath = src
markers =
    live: tests that call the real model (require credentials + RUN_LIVE_TESTS=1)
```

Run them:
```bash
pytest -q                       # layers 1–3: fast, no network, no cost
RUN_LIVE_TESTS=1 pytest -q      # also run layer 4 against the real Foundry model
```

### Layer 5 — Spec-coverage check (anti-drift)
The discipline that keeps the spec honest: **assert that every `FR-xxx` and `AC-x` in `spec.md` is referenced by at least one test.** A simple script enforces Constitution rule C-8.

`tests/test_spec_coverage.py`:
```python
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ids(pattern: str, text: str):
    return set(re.findall(pattern, text))


def test_every_requirement_has_a_test():
    spec = (ROOT / "specs" / "spec.md").read_text(encoding="utf-8")
    tests_blob = "\n".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py")
    )
    required = _ids(r"FR-\d+", spec) | _ids(r"AC-\d+", spec)
    covered = _ids(r"FR-\d+", tests_blob) | _ids(r"AC-\d+", tests_blob)
    missing = required - covered
    assert not missing, f"Spec items with no referencing test: {sorted(missing)}"
```

> Reference the requirement ID in a comment inside each test (as shown above, e.g. `# FR-006`). Now if someone adds `FR-009` to the spec without a test, CI goes red. **That is how you test the spec, not just the code.**

**Testing summary**

| Layer | What it proves | Model? | Speed | Run in CI? |
|---|---|---|---|---|
| 1 Contract | Output always matches schema | No | ⚡ | Always |
| 2 Acceptance | Each AC behaves correctly (mocked) | No | ⚡ | Always |
| 3 Unit | Logic/wiring/edge cases | No | ⚡ | Always |
| 4 Live integration | Real model meets accuracy targets | **Yes** | 🐢 + $ | Nightly / pre-release |
| 5 Spec coverage | Every requirement has a test | No | ⚡ | Always |

---

## 8. Files: Necessary vs. Not

### ✅ Necessary (the SDD core)

| File / Dir | Why it's essential |
|---|---|
| `specs/spec.md` | The source of truth — *what* & *why*, requirements, acceptance criteria. **The heart of SDD.** |
| `specs/plan.md` | The *how* — stack, architecture, key decisions. Gives the agent constraints. |
| `specs/tasks.md` | Atomic, testable work items traced to requirements. Drives implementation. |
| `specs/constitution.md` | Project-wide non-negotiables (security, perf, quality). Keeps every plan honest. |
| `src/…` | The generated implementation (a build output of the spec). |
| `tests/test_contract.py`, `test_spec_acceptance.py`, `test_spec_coverage.py` | Prove the code matches the spec; prevent drift. **Without tests, the spec is just hope.** |
| `.env.example` | Documents required configuration without leaking secrets. Committed. |
| `.gitignore` | Ensures `.env` and secrets never reach the repo. |
| `requirements.txt` / `pyproject.toml` | Reproducible environment. |

### ⚠️ Useful but optional (add as the project grows)

| File | When you need it |
|---|---|
| `specs/checklists/*.md` | Quality/review gates for larger features (Spec Kit ships these). |
| `tests/test_live_integration.py` | When you must validate real-model accuracy, not just logic. |
| `.github/workflows/ci.yml` | When more than one person touches the repo. |
| `CHANGELOG.md`, `README.md` | Public/team-facing projects. |
| Spec Kit's `.specify/` scaffolding | If you adopt the CLI instead of hand-writing artifacts. |

### ❌ Not necessary (skip these — they add friction, not clarity)

| Anti-pattern | Why to avoid |
|---|---|
| A hard-coded `config.py` with the endpoint/key in it | Violates FR-008/C-1; secrets belong in env vars. |
| `.env` committed to git | Leaks credentials. Commit `.env.example` only. |
| A 200-page upfront requirements doc covering features you *might* build someday | That's Waterfall. Spec **only the slice you're building now**; the spec is living, not exhaustive. |
| Separate, duplicated label lists in the prompt, the validator, and the docs | Single-source them (here: the `Enum`s) so they can't drift. |
| Generated code that's been hand-edited *and* re-generated without updating the spec | Pick one source of truth. If you edit code, back-port the intent into the spec. |
| Mock-only test suite with no schema/coverage gates | 90% "coverage" of happy paths the agent imagined ≠ a tested spec. |

> **Rule of thumb:** if a file captures *intent or a guarantee*, it's necessary. If it merely *repeats* something already stated elsewhere, it's drift waiting to happen.

---

## 9. Pros and Cons of Spec-Driven Development

### Pros

| Benefit | Detail |
|---|---|
| **Less rework, safer change** | Surfaces hidden assumptions early — when changing direction costs keystrokes, not sprints. Reported 5–10× fewer "regenerate from scratch" cycles. |
| **Higher first-pass AI accuracy** | A precise spec is excellent context; reduces intent-to-implementation deviation. Reported ~3–10× higher first-pass success on non-trivial tasks. |
| **Verifiable output** | Explicit acceptance criteria make "is it right?" answerable, shrinking endless code reviews. |
| **Beats context decay** | The spec is durable context the agent re-reads, instead of relying on a fading chat window. |
| **Shared understanding** | One source of truth aligns product, engineering, and QA — fewer "I assumed…" failures. |
| **Easier onboarding & maintenance** | New people (and agents) read the spec, not tribal knowledge in someone's head. |
| **Multi-variant exploration** | Because the spec is decoupled from code, you can ask the agent for two implementations (e.g., Rust vs. Go) from the *same* spec. |
| **Best where mistakes are expensive** | APIs, platform services, regulated workflows, distributed teams, backward-compatibility-sensitive systems. |

### Cons

| Drawback | Detail / Mitigation |
|---|---|
| **Front-loaded effort** | Writing a good spec is slower to start. *Mitigation:* spec only the current slice; let the agent draft v1. |
| **Stale-spec risk** | If specs aren't wired to tests/CI/review gates, they become **another** source of drift. *Mitigation:* the Layer-5 coverage gate; update the spec first when requirements change. |
| **Over-specification** | Too much upfront detail recreates Waterfall's flaw — you can't fully specify a system before building it; users don't know what they want until they see it. *Mitigation:* keep specs lightweight and iterate in minutes. |
| **False confidence** | A polished spec can feel like proof of correctness even when the code lies. *Mitigation:* tests are the proof, not the prose. |
| **Learning curve & process overhead** | New notation (EARS), new artifacts, new commands; overkill for a throwaway prototype. *Mitigation:* use vibe coding for spikes; switch to SDD when it goes to production. |
| **Tooling churn** | The 2026 landscape (Spec Kit, Kiro, OpenSpec, BMAD, …) is young and moving fast. *Mitigation:* the *method* is tool-agnostic — any structured Markdown spec works. |

**Bottom line:** SDD trades **speed now for speed later**. It shines for production systems, APIs, teams, and AI-generated code; it's overkill for quick experiments and one-off scripts.

---

## 10. Common Pitfalls & Best Practices

- **Resolve every `[NEEDS CLARIFICATION]` before coding.** An unresolved ambiguity becomes a wrong default in the implementation.
- **Write acceptance criteria as Given/When/Then.** They convert 1:1 into tests (Layers 2 & 4).
- **Single-source your invariants.** Label sets, schemas, and limits live in exactly one place and get injected everywhere else.
- **Fix the spec, then the code.** If reality disagrees with the spec, update the spec first; treat the code as regenerable.
- **Keep specs small and living.** Spec the slice you're building now. A spec you never update is worse than no spec.
- **Gate drift in CI.** The spec-coverage test (Layer 5) is what makes "the spec is the source of truth" actually true.
- **Keep secrets in env vars.** Endpoint, key, and deployment name are configuration, never code.

---

## 11. A 1-Week Skilling Plan

| Day | Focus | Outcome |
|---|---|---|
| **1** | Read this guide + GitHub Spec Kit README + the Microsoft Learn module intro. | Mental model of Specify → Plan → Tasks → Implement. |
| **2** | Hand-build the example in §6 (On-ramp A). Get `quickstart.py` returning a real classification from your Foundry model. | A working SDD project end-to-end. |
| **3** | Write all 5 test layers (§7). Make `pytest -q` green; run one live test. | You can *prove* the spec is satisfied. |
| **4** | Install Spec Kit (On-ramp B). Re-create the same feature with `specify init` + slash commands. | Comfort with the tooling. |
| **5** | Add a new requirement (e.g., a 5th category) **by editing the spec first**, then regenerate code + add its test. | You've felt the "edit spec → regenerate" loop. |
| **6** | Add a `constitution.md` rule (e.g., redact PII from logs) and make the agent honor it across plan + code. | Constitution-driven governance. |
| **7** | Wire `pytest` (Layers 1–3 + 5) into a CI workflow. Write a short retro: where did SDD help vs. slow you down? | A repeatable team-ready workflow + your own calibrated opinion. |

---

## 12. References

- **GitHub Spec Kit** — toolkit + docs: <https://github.com/github/spec-kit> · <https://github.github.com/spec-kit/>
- **Microsoft for Developers** — *Diving Into Spec-Driven Development With GitHub Spec Kit*: <https://developer.microsoft.com/blog/spec-driven-development-spec-kit>
- **Microsoft Learn** — *Implement spec-driven development using the GitHub Spec Kit* (free module): <https://learn.microsoft.com/en-us/training/modules/spec-driven-development-github-spec-kit-enterprise-developers/>
- **Microsoft Foundry SDK quickstart (Python)**: <https://learn.microsoft.com/en-us/azure/foundry/quickstarts/get-started-code>
- **Azure AI Projects client library for Python** (`azure-ai-projects`, `AIProjectClient.get_openai_client()`): <https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme>
- **Scalable Path** — *Spec-Driven Development Tutorial using GitHub Spec Kit* (trip-planner walkthrough): <https://www.scalablepath.com/ai/spec-driven-development-workflow>
- **BCMS** — *Spec-Driven Development: The Definitive 2026 Guide* (EARS, tool landscape): <https://thebcms.com/blog/spec-driven-development>
- **UngerAI** — *Spec-Driven Development: Pros, Cons, and Value Proposition*: <https://www.ungerai.com/spec-driven-development-report.html>

---

*Built as a learning artifact. The code samples are illustrative; validate model names, SDK versions, and endpoints against your own Microsoft Foundry project before production use.*
