# Spec-Driven Development on Snowflake Cortex AI

*A spec-first methodology for **fine-tuning and serving/inferencing LLMs on Snowflake Cortex AI** — the fully managed, serverless GenAI layer that runs **inside the Snowflake security perimeter** (your data never leaves). Same SDD method as the companion guides — **Constitution → Spec → Plan → Tasks → Implement → Test** — applied to building production-scale AI applications where AI comes to the data, in SQL.*

> **Companion to:** the SDD, Hugging Face / NVIDIA fine-tuning, vLLM/SGLang/Fireworks serving, and Databricks guides.
>
> **Who this is for:** developers and data teams building production GenAI on Snowflake who want AI **next to governed data**, with RBAC and data residency enforced — not pipelines that copy sensitive data out to an external API.
>
> **What you'll be able to do:** write a spec that *enforces* Cortex's fine-tune/inference/governance rules (RBAC privileges, row-count limits, model availability, data-stays-in-perimeter, PII redaction) before a `FINETUNE` or `AI_COMPLETE` runs; pick the right inference path for latency vs throughput; and prove behavior + credits with pre-ship tests.

---

## Table of Contents
1. [What Cortex AI Is & When to Use It](#1-what-cortex-ai-is--when-to-use-it)
2. [The Two Lifecycles: Fine-Tune & Inference](#2-the-two-lifecycles-fine-tune--inference)
3. [The SDD Method Applied to Cortex AI](#3-the-sdd-method-applied-to-cortex-ai)
4. [Step 1 — The Constitution](#4-step-1--the-constitution)
5. [Step 2 — The Spec (with enforceable matrices + SLOs)](#5-step-2--the-spec)
6. [Step 3 — The Plan](#6-step-3--the-plan)
7. [Step 4 — Tasks](#7-step-4--tasks)
8. [Step 5 — Implement](#8-step-5--implement)
9. [Cost & Inference-Path Sizing (Credits)](#9-cost--inference-path-sizing-credits)
10. [How to Test on Snowflake](#10-how-to-test-on-snowflake)
11. [Files: Necessary vs. Not](#11-files-necessary-vs-not)
12. [Pros & Cons](#12-pros--cons)
13. [References](#13-references)

---

## 1. What Cortex AI Is & When to Use It

**Snowflake Cortex AI** is a fully managed, **serverless** suite of AI/ML services that runs **directly inside the Snowflake AI Data Cloud** — within Snowflake's security perimeter, so **training and inference data never leaves**. The pieces a developer uses:

| Capability | What it does |
|---|---|
| **Cortex AI Functions (AISQL)** | SQL- and Python-callable LLM ops: **`AI_COMPLETE`** (the main generative function; formerly `COMPLETE`), `AI_CLASSIFY`, `AI_FILTER`, `AI_AGG`, `AI_EMBED`, `AI_EXTRACT`, `AI_SENTIMENT`, `AI_SUMMARIZE_AGG`, `AI_PARSE_DOCUMENT`, **`AI_REDACT`** (PII), `AI_TRANSLATE`, helpers `AI_COUNT_TOKENS` / `PROMPT` / `TO_FILE`. Models from OpenAI, Anthropic, Meta, Mistral, DeepSeek — all hosted inside the perimeter. |
| **REST API** | Low-latency endpoints for interactive apps: **Complete API**, **Embed API**, **Agents API**. |
| **Cortex Fine-tuning** (GA) | Managed **PEFT/LoRA** adapters via the **`FINETUNE`** function (CREATE/SHOW/DESCRIBE/CANCEL). Base models include `mistral-7b`, `mixtral-8x7b`, `llama3-8b/70b`, `llama3.1-8b/70b`. Output is a **model object** usable via `AI_COMPLETE`. |
| **Cortex Training** (preview, 2026) | Managed **full fine-tuning + reinforcement learning** on Snowflake-managed **GPU compute pools** via the open-source **ArcticTraining** (declarative YAML) framework. Models: Qwen, Mistral (up to ~14B). RL on proprietary data — not available in Cortex Fine-tuning. |
| **Cortex Search / Analyst / Agents** | Managed retrieval (RAG), text-to-SQL, and agent orchestration over governed data. |
| **Snowpark Container Services + Model Registry** | Bring-your-own custom model serving when you need a model Cortex doesn't host. |

**Use Cortex when** your data lives in Snowflake and **data residency / governance** matters — you want AI **on the data, in SQL**, without moving anything out. **Don't** if you need a model Snowflake doesn't host with full serving control (→ self-host vLLM/SGLang, or bring it via SPCS).

**Two inference paths (the choice the spec enforces):**
1. **AISQL functions** (`AI_COMPLETE` over a table) — optimized for **throughput / batch**: apply an LLM to millions of rows in a query.
2. **REST API** (Complete/Embed/Agents) — optimized for **low latency / interactive** apps.

---

## 2. The Two Lifecycles: Fine-Tune & Inference

```
FINE-TUNE                                              INFERENCE
  training data in a Snowflake TABLE                     choose path:
       │  (prompt, completion columns)                      ├─ AISQL: AI_COMPLETE / AI_CLASSIFY  → batch / in-SQL
       ▼                                                     └─ REST: Complete / Embed / Agents   → interactive
  Cortex Fine-tuning (PEFT/LoRA):                         every call runs inside the Snowflake perimeter
    SNOWFLAKE.CORTEX.FINETUNE('CREATE', model, …)         (data never leaves; RBAC enforced)
       │   → fine-tuned MODEL object in your schema
       ▼                                                  governance:
  (or) Cortex Training (full FT + RL):                      • SNOWFLAKE.CORTEX_USER role to call functions
    ArcticTraining YAML on managed GPU pools                • CREATE MODEL privilege to save fine-tunes
       │   → custom model                                   • AI_REDACT for PII; usage-history views for spend
       ▼
  eval → grant model usage → serve via AI_COMPLETE / REST
```

**Key rules the spec enforces:** the calling role needs **`SNOWFLAKE.CORTEX_USER`** (and `USE AI FUNCTIONS`); saving a fine-tune needs **`CREATE MODEL`**; the **training row count must stay under the model/epoch limit**; and the base model must be in the **fine-tunable** list. RL needs **Cortex Training**, not Cortex Fine-tuning.

---

## 3. The SDD Method Applied to Cortex AI

```
SPECIFY ─────────▶ PLAN ──────────▶ TASKS ─────────▶ IMPLEMENT ───────▶ (PRE-RUN GATES)
model, method,     Cortex FT (PEFT) data → FINETUNE   SQL FINETUNE /     matrix valid?
inference path,    vs Cortex Train  → eval → grant →  AI_COMPLETE /      RBAC privileges present?
SLOs, governance   (full/RL); AISQL  AI_COMPLETE/REST  REST / ArcticYAML  row-limit ok? data-in-
(RBAC + perimeter),  vs REST path                                          perimeter? cost? eval gate?
cost ceiling,
matrices
```

"Done" = **SLOs** (latency for REST, throughput for AISQL) **plus** a **credit cost ceiling**, an **eval gate** for fine-tunes, and a **governance posture** (RBAC grants + data-stays-in-perimeter + PII handling) — all enforced before the job runs.

---

## 4. Step 1 — The Constitution

`specs/constitution.md`:

```markdown
# Project Constitution — Snowflake Cortex AI

## Governance & Data Residency (RBAC + perimeter — the Snowflake core)
- G-1: All AI runs inside the Snowflake perimeter. Training/inference data MUST NOT be copied to
       an external API or out of Snowflake.
- G-2: The calling role has SNOWFLAKE.CORTEX_USER (+ USE AI FUNCTIONS). Saving a fine-tune requires
       CREATE MODEL on the target schema. Access to a fine-tuned model is via explicit GRANT.
- G-3: PII in training data and prompts is screened/redacted (AI_REDACT) before use; documented.
- G-4: Credit + token consumption is tracked (CORTEX_FINE_TUNING_USAGE_HISTORY / usage views).

## Reproducibility
- R-1: Fine-tune training data is a versioned/snapshotted table; the FINETUNE job ID + params recorded.
- R-2: The base model, dataset query, and epochs are pinned in version control; not ad-hoc in a worksheet.
- R-3: Record the fine-tuned MODEL object name + version in the deployment manifest.

## Method Integrity (platform rules — enforced by config.py)
- M-1: A run declares method ∈ {cortex_finetune (PEFT/LoRA), cortex_training (full/RL)} and
       inference_path ∈ {aisql, rest}.
- M-2: cortex_finetune base model MUST be in the fine-tunable list; training rows MUST be <= the
       model/epoch row limit.
- M-3: Reinforcement learning / full fine-tuning REQUIRES cortex_training (ArcticTraining), not
       cortex_finetune.

## Cost (Credits)
- C-1: Declare a credit ceiling. FT cost = input_tokens * epochs; inference cost = input+output tokens
       (per Service Consumption Table); plus warehouse + storage.
- C-2: Bulk/table workloads → AISQL (throughput-optimized); interactive → REST (latency-optimized).
       Right-size the warehouse; avoid oversized warehouses for AI_COMPLETE batch jobs.

## Quality
- Q-1: Structured output uses AI_COMPLETE response_format (JSON schema), not free-text parsing.
- Q-2: A fine-tune MUST pass an eval gate (beat base by a threshold) before promotion.
- Q-3: Every requirement (FR/AC) maps to >= 1 automated test.
```

---

## 5. Step 2 — The Spec

`specs/spec.md`:

```markdown
# Spec: Snowflake Cortex Fine-Tune / Inference

## Functional Requirements
- FR-001: A run is fully determined by one config (base model, method, inference_path, dataset query,
          epochs, SLOs, governance, cost ceiling).
- FR-002: Functions/objects referenced by fully-qualified names (DB.SCHEMA.OBJECT); roles explicit.
- FR-003: The config validates the Method, Inference, and Governance matrices and FAILS before any
          FINETUNE/AI_COMPLETE call.
- FR-004: cortex_finetune base model MUST be in the fine-tunable list (FR for availability). (M-2)
- FR-005: training_rows <= effective_row_limit(base_model, epochs). (M-2)
- FR-006: RL or full fine-tuning → method MUST be cortex_training. (M-3)
- FR-007: The calling role has CORTEX_USER + USE AI FUNCTIONS; fine-tune needs CREATE MODEL. (G-2)
- FR-008: Data stays in-perimeter; no external copy. PII screened with AI_REDACT where required. (G-1/G-3)
- FR-009: Structured output uses AI_COMPLETE response_format (JSON schema). (Q-1)
- FR-010: Bulk/table → AISQL; interactive low-latency → REST. (C-2)

## ── METHOD MATRIX (enforced by config.py) ──
| Field          | cortex_finetune (PEFT/LoRA)     | cortex_training (full + RL) |
|----------------|----------------------------------|------------------------------|
| status         | GA                               | preview (2026)               |
| objective      | SFT (adapters)                   | full FT + RL (ArcticTraining)|
| base models    | mistral-7b, mixtral-8x7b, llama3-8b/70b, llama3.1-8b/70b | Qwen, Mistral (≤ ~14B) |
| interface      | FINETUNE('CREATE', …)            | ArcticTraining YAML on GPU pools |
| row limit      | per model/epoch (enforced)       | scales with GPU pool         |
| RL allowed     | ❌ (M-3)                          | ✅                           |

## ── INFERENCE MATRIX ──
| inference_path | function/endpoint           | best for           |
|----------------|------------------------------|--------------------|
| aisql          | AI_COMPLETE / AI_CLASSIFY … over tables | batch / throughput / ETL |
| rest           | Complete / Embed / Agents API | interactive / low latency |

## ── ROW-LIMIT TABLE (effective rows = 1-epoch limit / epochs; enforced for cortex_finetune) ──
| base model    | 1 epoch | 3 epochs (default) |
|---------------|--------:|-------------------:|
| llama3-8b     | 186k    | 62k                |
| llama3-70b    | 21k     | 7k                 |
| llama3.1-8b   | 150k    | 50k                |
| llama3.1-70b  | 13.5k   | 4.5k               |
| mistral-7b    | 45k     | 15k                |
| mixtral-8x7b  | 27k     | 9k                 |

## SLOs & Cost (definition of done)
- SLO-LATENCY (rest): p95 latency under target for interactive requests.
- SLO-THROUGHPUT (aisql): rows/min over the table workload at the chosen warehouse size.
- COST-CEILING: projected credits (FT trained-tokens + inference tokens + warehouse) <= ceiling.
- EVAL-GATE: fine-tuned model beats base by >= threshold on a held-out table.

## Acceptance Criteria
- AC-1: cortex_finetune on a non-fine-tunable base model → rejected. (M-2/FR-004)
- AC-2: training_rows above the model/epoch limit → rejected. (M-2/FR-005)
- AC-3: method=cortex_finetune with RL requested → rejected (use cortex_training). (M-3)
- AC-4: missing CORTEX_USER / CREATE MODEL privilege → rejected at preflight. (FR-007)
- AC-5: a structured-output AI_COMPLETE returns JSON validating against the schema. (FR-009)
- AC-6: a fine-tune failing the eval gate → promotion blocked. (Q-2)
- AC-7: projected credits > ceiling → flagged for redesign (path / warehouse size). (COST-CEILING)
```

---

## 6. Step 3 — The Plan

`specs/plan.md`:

```markdown
# Plan: Snowflake Cortex AI
## Access & interfaces
- SQL (worksheets/Notebooks/Tasks), Python (snowflake.cortex), and REST (Complete/Embed/Agents).
- Roles: SNOWFLAKE.CORTEX_USER + USE AI FUNCTIONS; CREATE MODEL for fine-tunes; GRANT model usage.
## Fine-tuning (method decision)
- Adapters / cheap / GA          → Cortex Fine-tuning (FINETUNE; PEFT/LoRA; row limits apply).
- Full fine-tune or RL / preview → Cortex Training (ArcticTraining YAML; managed GPU pools; Qwen/Mistral).
## Inference (path decision)
- Bulk table processing / ETL    → AISQL: AI_COMPLETE / AI_CLASSIFY / AI_EXTRACT over tables.
- Interactive low-latency app     → REST: Complete / Embed / Agents API.
## Governance
- Everything stays in-perimeter (no external copy). AI_REDACT PII. Track credits via usage views.
- Snapshot the training table; pin base model + epochs; record FINETUNE job ID.
## RAG / agents: Cortex Search (retrieval), Cortex Analyst (text-to-SQL), Cortex Agents.
## Constitution check: G-* (RBAC/perimeter/PII), R-* (repro), M-* (method rules), C-* (credits), Q-* (eval/tests).
```

---

## 7. Step 4 — Tasks

`specs/tasks.md`:

```markdown
| ID | Task                												| Satisfies   | Test |
|----|--------------------------------------------------------------------|-------------|------|
| T1 | RunConfig schema enforcing Method + Inference + Governance matrices | FR-003, M-* | test_matrix.py |
| T2 | Row-limit + base-model-availability preflight                      | FR-004/005  | test_rowlimit.py |
| T3 | RBAC privilege preflight (CORTEX_USER / CREATE MODEL)              | FR-007      | test_rbac.py |
| T4 | Credit cost estimator vs ceiling                                  | COST, C-1   | test_cost.py |
| T5 | FINETUNE('CREATE') / ArcticTraining job from config               | FR-001      | test_finetune.py (live) |
| T6 | AI_COMPLETE (AISQL) + REST client + structured output            | FR-009/010  | test_inference.py |
| T7 | Eval gate (tuned beats base) + cost gate                          | Q-2, EVAL   | test_eval_gate.py |
| T8 | Map every FR/AC to a test                                        | Q-3         | test_spec_coverage.py |
```

---

## 8. Step 5 — Implement

### 8.1 Config enforcement (`src/cortex/config.py`)

Rejects platform-rule violations **before** any `FINETUNE`/`AI_COMPLETE` runs (saving credits + a failed job).

```python
"""RunConfig enforces the Cortex Method + Inference + Governance matrices (FR-003)."""
from enum import Enum
from typing import Optional
import yaml
from pydantic import BaseModel, Field, model_validator

# Effective 1-epoch row limits per fine-tunable base model
ONE_EPOCH_LIMIT = {"llama3-8b":186_000, "llama3-70b":21_000, "llama3.1-8b":150_000,
                   "llama3.1-70b":13_500, "mistral-7b":45_000, "mixtral-8x7b":27_000}

class Method(str, Enum):
    cortex_finetune = "cortex_finetune"      # PEFT/LoRA (GA)
    cortex_training = "cortex_training"      # full FT + RL (preview)

class InferencePath(str, Enum):
    aisql = "aisql"; rest = "rest"

class RunConfig(BaseModel):
    method: Method
    base_model: str
    inference_path: InferencePath
    epochs: int = Field(3, ge=1)
    training_rows: int = Field(0, ge=0)
    use_rl: bool = False
    role_has_cortex_user: bool = False       # from a SHOW GRANTS preflight
    role_has_create_model: bool = False
    cost_ceiling_credits: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _enforce(self):
        if self.use_rl and self.method is not Method.cortex_training:
            raise ValueError("RL/full fine-tuning requires cortex_training (AC-3/M-3)")
        if self.method is Method.cortex_finetune:
            if self.base_model not in ONE_EPOCH_LIMIT:
                raise ValueError(f"{self.base_model} is not fine-tunable via Cortex Fine-tuning (AC-1/M-2)")
            limit = ONE_EPOCH_LIMIT[self.base_model] // self.epochs   # effective row limit
            if self.training_rows > limit:
                raise ValueError(f"training_rows {self.training_rows} > limit {limit} for "
                                 f"{self.base_model}@{self.epochs} epochs (AC-2/M-2)")
            if not self.role_has_create_model:
                raise ValueError("fine-tune requires CREATE MODEL privilege (AC-4/FR-007)")
        if not self.role_has_cortex_user:
            raise ValueError("calling role needs SNOWFLAKE.CORTEX_USER (AC-4/FR-007)")
        return self

def load_config(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as fh:
        return RunConfig.model_validate(yaml.safe_load(fh))
```

`configs/finetune_mistral7b.yaml`:
```yaml
method: cortex_finetune          # PEFT/LoRA, GA
base_model: mistral-7b           # must be fine-tunable (enforced)
inference_path: aisql            # bulk ticket categorization over a table
epochs: 3
training_rows: 12000             # <= 15k limit for mistral-7b @ 3 epochs (enforced)
use_rl: false
role_has_cortex_user: true
role_has_create_model: true
cost_ceiling_credits: 50
```

### 8.2 Fine-tune (SQL `FINETUNE`)

```sql
-- Cortex Fine-tuning: PEFT/LoRA on a Snowflake table, inside the perimeter (FR-008/G-1)
SELECT SNOWFLAKE.CORTEX.FINETUNE(
  'CREATE',
  'my_db.my_schema.support_bot_ft',                 -- output MODEL object (needs CREATE MODEL)
  'mistral-7b',                                       -- base model (must be fine-tunable)
  'SELECT prompt, completion FROM my_db.my_schema.train_tickets',   -- training query
  'SELECT prompt, completion FROM my_db.my_schema.eval_tickets'     -- validation query
);
-- Track progress / status:
SELECT SNOWFLAKE.CORTEX.FINETUNE('DESCRIBE', '<job_id>');
-- Credits + trained tokens:
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FINE_TUNING_USAGE_HISTORY;
```

### 8.3 Inference — AISQL (batch) + REST (interactive) + structured output

```sql
-- AISQL: apply the fine-tuned model to a whole table (throughput path, FR-010)
SELECT id,
       AI_COMPLETE('my_db.my_schema.support_bot_ft', ticket_text)               AS reply,
       AI_CLASSIFY(ticket_text, ['billing','support','product','other'])::STRING AS category
FROM   my_db.crm.support_tickets
WHERE  created_at >= DATEADD('day', -1, CURRENT_DATE());

-- Structured output via response_format (JSON schema), FR-009:
SELECT AI_COMPLETE(
  model => 'mistral-7b',
  prompt => 'Classify sentiment: Snowflake is great!',
  response_format => {
    'type':'json','schema':{'type':'object',
      'properties':{'sentiment':{'type':'string','enum':['positive','negative','neutral']}},
      'required':['sentiment']}}
) AS result;

-- PII redaction before sending to a model (G-3):
SELECT AI_COMPLETE('mistral-7b', AI_REDACT(raw_note)) FROM my_db.clinical.notes;
```

```python
# Python (Snowpark) — same functions; and REST for low-latency interactive apps (FR-010)
from snowflake.cortex import complete           # snowflake-ml-python
out = complete("mistral-7b", "Summarize: ...", session=session)
# REST Complete API: POST https://<account>.snowflakecomputing.com/api/v2/cortex/inference:complete
#   Authorization: key-pair / OAuth; body = {"model": "...", "messages": [...]}  → low latency.
```

### 8.4 Full fine-tune / RL — Cortex Training (preview)

```yaml
# ArcticTraining (declarative YAML) runs on Snowflake-managed GPU compute pools via ML Jobs.
# Use this path for FULL fine-tuning or REINFORCEMENT LEARNING (not possible in Cortex Fine-tuning).
type: sft                  # or grpo/dpo for RL post-training
model: Qwen/Qwen3-1.7B     # quickstart; production up to ~14B (Qwen / Mistral families)
data:
  source: my_db.my_schema.train_table     # reads from Snowflake tables, stays in-perimeter
epochs: 3
# Submitted as an ML Job; output stays within Snowflake's security boundary.
```

---

## 9. Cost & Inference-Path Sizing (Credits)

Snowflake meters **credits** (per the Service Consumption Table) — the spec's estimator (T4) projects spend:

| Activity | Cost meter | Notes |
|---|---|---|
| **Cortex Fine-tuning** | trained tokens = **input_tokens × epochs** | plus storage for the adapter |
| **Inference (AI_COMPLETE / REST)** | input **+ output** tokens (credits per 1M tokens, model-specific) | bigger model = more credits/token |
| **AISQL batch** | inference tokens **+ warehouse** running the query | right-size the warehouse — oversized = wasted credits |
| **Cortex Training (preview)** | managed GPU compute-pool time | full FT / RL; "up to 2× runs per GPU $" (Snowflake claim) |

**Path rule (C-2):** **AISQL** is throughput-optimized — use it to process millions of rows; **REST** is latency-optimized — use it for interactive apps. For AISQL batch, the **warehouse size** is a real cost lever: too big wastes credits, too small slows throughput. Use `AI_COUNT_TOKENS` to estimate token volume up front and feed it to the cost gate.

> **Sizing logic:** FT credits ≈ (rows × avg_input_tokens × epochs) × FT-rate; inference credits ≈ (calls × (in+out tokens)) × model-rate; AISQL adds warehouse credits for the query runtime. Put your token estimates + the current Service Consumption rates into the T4 estimator and let the cost gate decide path + warehouse size.

---

## 10. How to Test on Snowflake

Because Cortex is serverless/managed, the highest-value tests are **cheap, no-compute checks of platform-rule + RBAC + cost correctness** — before any `FINETUNE`/`AI_COMPLETE` burns credits.

| Layer | Proves | Runs compute? | When |
|---|---|---|---|
| **1 Matrix enforcement** | Illegal method/inference configs rejected (AC-1/2/3) | No | Every commit |
| **2 Row-limit preflight** | training rows ≤ model/epoch limit; base model fine-tunable (AC-2) | No | Pre-run |
| **3 RBAC preflight** | role has CORTEX_USER / CREATE MODEL (AC-4) | SHOW GRANTS only | Pre-run |
| **4 Cost gate** | projected credits ≤ ceiling (AC-7) | No (uses AI_COUNT_TOKENS) | Every commit |
| **5 Smoke + structured output** | AI_COMPLETE returns; JSON validates (AC-5) | Yes (small) | Pre-ship |
| **6 Eval gate** | fine-tune beats base by threshold (AC-6) | Yes | Pre-promotion |
| **7 Spec coverage** | every FR/AC has a test (Q-3) | No | Every commit |

```python
import pytest
from pydantic import ValidationError
from cortex.config import RunConfig

def test_ac2_rows_over_limit_rejected():           # AC-2 / M-2
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"method":"cortex_finetune","base_model":"mistral-7b",
            "inference_path":"aisql","epochs":3,"training_rows":20000,   # > 15k limit
            "role_has_cortex_user":True,"role_has_create_model":True,"cost_ceiling_credits":50})

def test_ac3_rl_requires_cortex_training():        # AC-3 / M-3
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"method":"cortex_finetune","base_model":"llama3-8b",
            "inference_path":"rest","use_rl":True,"role_has_cortex_user":True,
            "role_has_create_model":True,"cost_ceiling_credits":50})
```

> The **RBAC preflight** (Layer 3) and **row-limit preflight** (Layer 2) are the Snowflake equivalents of the self-hosted GPU-fit preflight: a missing `CORTEX_USER` grant or an over-limit dataset fails CI in milliseconds instead of erroring out a long-running job after you've spent credits.

---

## 11. Files: Necessary vs. Not

### ✅ Necessary
| File | Why |
|---|---|
| `specs/spec.md` | Method + Inference + Governance matrices, row-limit table, SLOs, cost ceiling, eval gate. |
| `specs/constitution.md` | RBAC + perimeter + PII governance, repro, method rules, credit cost. |
| `src/cortex/config.py` | Enforces platform rules before any FINETUNE/AI_COMPLETE. |
| `src/cortex/cost.py` | Projects credits (trained tokens + inference + warehouse) vs ceiling. |
| `sql/finetune.sql`, `sql/inference.sql` | The FINETUNE + AI_COMPLETE statements, in version control. |
| `tests/test_matrix.py`, `test_rowlimit.py`, `test_rbac.py`, `test_cost.py`, `test_spec_coverage.py` | Prove rules + RBAC + cost. |
| Training-table snapshot + FINETUNE job ID + model name | Reproducibility (R-1/R-3). |
| Key-pair / OAuth via secrets, `.gitignore` | Auth via secrets, never committed. |

### ⚠️ Optional
| File | When |
|---|---|
| ArcticTraining YAML | Full fine-tuning / RL via Cortex Training. |
| Cortex Search service definition | RAG apps. |
| Snowflake Tasks / Streams DDL | Scheduled batch AISQL pipelines. |
| SPCS + Model Registry manifest | Bring-your-own custom model serving. |

### ❌ Not necessary (anti-patterns)
| Anti-pattern | Why avoid |
|---|---|
| Copying training/inference data out to an external API | Breaks data residency — the whole point of Cortex (G-1). |
| Running FINETUNE in an ad-hoc worksheet with no config in VC | Unreproducible; bypasses row-limit + RBAC + cost checks. |
| Exceeding the model/epoch row limit | The job fails — preflight it (M-2). |
| Requesting RL via Cortex Fine-tuning | Not supported — use Cortex Training (M-3). |
| Sending raw PII to a model with no AI_REDACT | Governance/compliance breach (G-3). |
| Oversized warehouse for AISQL batch | Wastes credits — size to the workload (C-2). |
| Free-text JSON parsing | Use AI_COMPLETE response_format (Q-1). |

---

## 12. Pros & Cons

**Pros:** **data never leaves Snowflake** — training + inference run inside the security perimeter under existing **RBAC** (the strongest data-residency story among managed platforms); **SQL-first** (`AI_COMPLETE` / AISQL) means data + SQL teams ship GenAI without new infra; fully managed + serverless (no GPU ops); both **PEFT fine-tuning (GA)** and **full FT + RL via Cortex Training (preview)**; rich AISQL surface (classify/extract/redact/parse/translate) for in-pipeline AI; usage-history views make credit tracking native; Cortex Search/Analyst/Agents complete the RAG/agent stack.

**Cons:** **Snowflake-centric** (best only if your data lives there); **model catalog is curated** — you serve what Snowflake hosts (or bring-your-own via SPCS, which is more work); **fine-tunable base models + row limits are constrained** (Cortex Fine-tuning); Cortex Training is **preview** and limited to Qwen/Mistral ≤ ~14B; less low-level serving control than self-hosting; credits across FT + inference + warehouse take discipline to forecast.

**vs Databricks Mosaic AI:** both put AI next to governed data. **Snowflake** is SQL-first with the tightest data-stays-in-perimeter posture and the simplest path for SQL teams; **Databricks** is broader (MLflow lineage, provisioned-throughput serving, an agent stack) and more ML-engineer-centric. Choose by where your data + team already are — see the Databricks companion guide.

---

## 13. References
- **Snowflake — Cortex Fine-tuning** (`FINETUNE`, base models, row limits, cost): <https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-finetuning>
- **Snowflake — Cortex AI Functions / AISQL** (`AI_COMPLETE`, `AI_CLASSIFY`, `AI_REDACT`, REST APIs): <https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql>
- **Snowflake — Cortex LLM REST API** (Complete / Embed / Agents): <https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions>
- **Snowflake Cortex Training + ArcticTraining** (full FT + RL, preview, 2026): <https://www.snowflake.com/en/blog/> · <https://github.com/snowflakedb/ArcticTraining>
- **Snowflake — Privileges for Cortex AI** (CORTEX_USER, USE AI FUNCTIONS, CREATE MODEL): <https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql#required-privileges>
- **GitHub Spec Kit** (the SDD methodology applied here): <https://github.com/github/spec-kit>

---
*Illustrative artifact. Snowflake Cortex features, model lists, and consumption rates change frequently (Cortex Training is preview as of mid-2026) — confirm `FINETUNE`/`AI_COMPLETE` syntax, fine-tunable models, row limits, and credit rates against the live docs + your account region. Cost ceilings and SLOs are placeholders; set them from your token estimates (AI_COUNT_TOKENS) and validate with the cost gate + a smoke test before promoting.*
