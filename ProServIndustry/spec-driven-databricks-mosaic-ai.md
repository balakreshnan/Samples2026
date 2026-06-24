# Spec-Driven Development on Databricks Mosaic AI

*A spec-first methodology for **fine-tuning and serving LLMs on Databricks Mosaic AI** — the lakehouse-native GenAI platform where every AI surface (Foundation Model APIs, AI Functions, Vector Search, Model Serving, Agent Framework) sits on **Unity Catalog** governance and **MLflow** lineage. Same SDD method as the companion guides — **Constitution → Spec → Plan → Tasks → Implement → Test** — applied to building production-scale AI applications on Databricks.*

> **Companion to:** the SDD, Hugging Face / NVIDIA fine-tuning, vLLM/SGLang/Fireworks serving, and Snowflake guides.
>
> **Who this is for:** developers and platform teams building production GenAI on the Databricks Lakehouse who want governance, reproducibility, and cost control baked in — not notebooks that bypass Unity Catalog and surprise finance with the DBU bill.
>
> **What you'll be able to do:** write a spec that *enforces* Databricks' model/serving/governance rules (UC registration, provisioned-throughput eligibility, AI Gateway guardrails, rate limits) before an endpoint is created; pick the right serving mode for cost/SLO; and prove behavior + spend with pre-ship tests.

---

## Table of Contents
1. [What Mosaic AI Is & When to Use It](#1-what-mosaic-ai-is--when-to-use-it)
2. [The Two Lifecycles: Fine-Tune & Serve](#2-the-two-lifecycles-fine-tune--serve)
3. [The SDD Method Applied to Mosaic AI](#3-the-sdd-method-applied-to-mosaic-ai)
4. [Step 1 — The Constitution](#4-step-1--the-constitution)
5. [Step 2 — The Spec (with enforceable matrices + SLOs)](#5-step-2--the-spec)
6. [Step 3 — The Plan](#6-step-3--the-plan)
7. [Step 4 — Tasks](#7-step-4--tasks)
8. [Step 5 — Implement](#8-step-5--implement)
9. [Cost & Serving-Mode Sizing (DBUs)](#9-cost--serving-mode-sizing-dbus)
10. [How to Test on Databricks](#10-how-to-test-on-databricks)
11. [Files: Necessary vs. Not](#11-files-necessary-vs-not)
12. [Pros & Cons](#12-pros--cons)
13. [References](#13-references)

---

## 1. What Mosaic AI Is & When to Use It

**Databricks Mosaic AI** is the GenAI layer of the Databricks Lakehouse. In 2026 it spans **five surfaces, all governed by Unity Catalog and traced by MLflow**:

| Surface | What it does |
|---|---|
| **Foundation Model APIs** | Pay-per-token, OpenAI-compatible inference on hosted open models (Llama, DBRX, Mixtral, gpt-oss, Gemma, plus external Claude/Gemini/GPT via External Models). |
| **AI Functions** | SQL-callable LLM ops for **batch inference at scale**: `ai_query`, `ai_classify`, `ai_extract`, `ai_summarize`, `ai_translate`, `ai_parse_document`, etc. |
| **Model Serving** | Hosts custom + fine-tuned + foundation models on **provisioned-throughput** endpoints with performance guarantees. |
| **Vector Search** | Unity-Catalog-native managed vector index (hybrid retrieval) for RAG; inherits UC access policies. |
| **Agent Framework** | Multi-step agent orchestration + **Agent Evaluation**, with MLflow tracing and UC access control. |

The differentiator vs a standalone AI provider: **every AI surface respects the same Unity Catalog governance, access control, and lineage** as the rest of your lakehouse data — and **Unity AI Gateway** is the control plane (usage tracking, payload/inference logging, guardrails, rate limits).

**Use Mosaic AI when** your data already lives in the Databricks Lakehouse and you want AI **next to the data** under one governance model. **Don't** if you're not a Databricks shop or want a thin standalone token API (→ Fireworks).

**Four ways to serve a model (the choice the spec enforces):**
1. **Pay-per-token** (Foundation Model APIs) — experimentation, spiky/low volume, no infra commitment.
2. **AI Functions (batch)** — apply LLMs to tables at scale inside SQL; best for bulk/ETL.
3. **Provisioned throughput** — **production**: dedicated capacity in "model units," autoscaling between min/max tokens-per-second, **required for fine-tuned models**.
4. **External models** — proxy OpenAI/Anthropic/Google behind Databricks governance.

---

## 2. The Two Lifecycles: Fine-Tune & Serve

```
FINE-TUNE (Mosaic AI Model Training)            SERVE (Model Serving)
  data in UC table/Volume                          choose mode:
       │                                              ├─ pay-per-token (FMAPI)    → dev / spiky
       ▼                                              ├─ AI Functions (batch)     → bulk in SQL
  fine-tune a foundation model (LoRA/full)           ├─ provisioned throughput    → PRODUCTION
       │  → registered model in Unity Catalog         │     (required for fine-tunes)
       ▼     (system schema / your catalog.schema)     └─ external models          → governed proxy
  MLflow run + eval                                 wrap every endpoint with Unity AI Gateway
       │                                              (usage tracking, inference tables,
       ▼                                               guardrails, rate limits)
  deploy → provisioned-throughput endpoint  ───────▶  query (OpenAI-compatible) + monitor
```

**Key rule:** a **fine-tuned (or custom) model must be registered in Unity Catalog and served on a provisioned-throughput (or custom) endpoint** — it is *not* available on the shared pay-per-token tier. This is the central platform constraint the spec enforces.

---

## 3. The SDD Method Applied to Mosaic AI

```
SPECIFY ─────────▶ PLAN ──────────▶ TASKS ─────────▶ IMPLEMENT ───────▶ (PRE-SHIP GATES)
model, serve mode, fine-tune?       data → train →    REST / MLflow      matrix valid?
SLOs (tok/s),      provisioned-     register(UC) →    Deploy SDK / SQL   governance preflight?
governance (UC +   throughput tier  deploy → gateway   (no clicked-       eligibility (optimizable)?
AI Gateway), cost  + gateway plan   → eval             together endpoint) cost ceiling? eval gate?
ceiling, matrices
```

"Done" = **SLOs** (provisioned tokens/sec, p95 latency) **plus** a **DBU cost ceiling**, an **eval gate** for fine-tunes, and **governance posture** (UC registration + AI Gateway guardrails/rate limits/usage tracking) — all enforced before the endpoint goes live.

---

## 4. Step 1 — The Constitution

`specs/constitution.md`:

```markdown
# Project Constitution — Databricks Mosaic AI

## Governance (Unity Catalog + AI Gateway — the lakehouse core)
- G-1: Every model (foundation, fine-tuned, custom) is registered in Unity Catalog
       (catalog.schema.model). No serving from un-governed artifacts.
- G-2: Every production endpoint has Unity AI Gateway configured: usage tracking ON,
       inference (payload) tables ON, AI Guardrails ON, and rate limits set.
- G-3: Endpoint CREATE + CAN MANAGE restricted to admins; other users get query-only.
- G-4: Data access for training + RAG flows through UC permissions (no raw credential paths).

## Reproducibility (MLflow)
- R-1: Every fine-tune is an MLflow run with the dataset version, params, and metrics logged.
- R-2: Served model is referenced by UC name + version; the endpoint config lives in version control.
- R-3: Pin the model + dataset version; record the MLflow run ID in the deployment manifest.

## Serving Integrity (platform rules — enforced by config.py)
- S-1: A fine-tuned or custom model MUST be served via provisioned throughput (or custom
       serving), NOT pay-per-token.
- S-2: A provisioned-throughput deploy MUST verify the model is "optimizable" (eligible) and size
       min/max throughput in the allowed chunk increments.
- S-3: Interactive prod traffic → provisioned throughput; bulk table processing → AI Functions.

## Cost (DBUs)
- C-1: Declare a DBU/$ ceiling per surface (FMAPI per-token, Serving DBUs, Vector Search query+storage).
- C-2: Prefer pay-per-token for spiky/low volume; switch to provisioned throughput above the crossover;
       use AI Functions (batch) for bulk. Set autoscale min to control idle spend.

## Quality
- Q-1: Structured output uses the endpoint's response_format / JSON mode, not free-text parsing.
- Q-2: A fine-tune MUST pass an eval gate (beat base by a threshold) before promotion (Agent Eval / MLflow).
- Q-3: Every requirement (FR/AC) maps to >= 1 automated test.
```

---

## 5. Step 2 — The Spec

`specs/spec.md`:

```markdown
# Spec: Databricks Mosaic AI Fine-Tune / Serve

## Functional Requirements
- FR-001: A run is fully determined by one config (model UC name, serve_mode, fine-tune settings,
          SLOs, gateway config, cost ceiling).
- FR-002: Models are referenced by Unity Catalog name (catalog.schema.model[@version]). (G-1)
- FR-003: The config validates the Serve & Governance matrices and FAILS before any API call.
- FR-004: serve_mode=pay_per_token is allowed ONLY for hosted foundation models, never fine-tuned/custom. (S-1)
- FR-005: serve_mode=provisioned_throughput REQUIRES the model to be optimizable; min/max throughput
          set in chunk increments; min <= max. (S-2)
- FR-006: A production endpoint REQUIRES gateway: usage_tracking=true, inference_tables=true,
          guardrails=true, and at least one rate_limit. (G-2)
- FR-007: A fine-tune declares dataset (UC table/Volume + version), method, and an eval gate. (R-1, Q-2)
- FR-008: Secrets/tokens come from Databricks secrets / env, never hard-coded.
- FR-009: Structured output uses response_format (JSON schema), not free-text parsing.
- FR-010: Bulk/ETL inference uses AI Functions; interactive prod uses provisioned throughput. (S-3)

## ── SERVE MATRIX (enforced by config.py) ──
| serve_mode            | model types        | best for            | gateway required |
|-----------------------|--------------------|---------------------|------------------|
| pay_per_token (FMAPI) | hosted FM only     | dev / spiky / low   | recommended      |
| ai_functions (batch)  | hosted FM (subset) | bulk table / ETL    | recommended      |
| provisioned_throughput| FM / fine-tuned / custom | PRODUCTION    | REQUIRED (prod)  |
| external_models       | OpenAI/Anthropic/… | governed proxy      | REQUIRED         |

Rules:
- M-1: serve_mode=pay_per_token + model is fine_tuned/custom → reject (S-1).
- M-2: serve_mode=provisioned_throughput + not optimizable → reject (S-2).
- M-3: provisioned_throughput + min_throughput > max_throughput → reject.
- M-4: is_production=true + (no gateway guardrails OR no rate_limit) → reject (G-2).

## SLOs & Cost (definition of done)
- SLO-TPS: provisioned endpoint sustains the declared tokens/sec at target concurrency.
- SLO-P95: p95 end-to-end latency under target for a representative request.
- COST-CEILING: projected DBU spend at expected volume <= declared ceiling.
- EVAL-GATE (fine-tunes): tuned model beats base by >= threshold on a held-out set.

## Acceptance Criteria
- AC-1: a fine-tuned model with serve_mode=pay_per_token → rejected. (M-1/S-1)
- AC-2: provisioned_throughput on a non-optimizable model → rejected. (M-2)
- AC-3: production endpoint with no guardrails/rate limit → rejected. (M-4/G-2)
- AC-4: min_throughput > max_throughput → rejected. (M-3)
- AC-5: a structured-output request returns JSON validating against the schema. (FR-009)
- AC-6: a fine-tune failing the eval gate → promotion blocked. (Q-2)
- AC-7: projected DBU spend > ceiling → flagged for redesign (mode switch / batch). (COST-CEILING)
```

---

## 6. Step 3 — The Plan

`specs/plan.md`:

```markdown
# Plan: Databricks Mosaic AI
## Access & SDKs
- OpenAI-compatible serving endpoints; Databricks REST (/api/2.0/serving-endpoints),
  MLflow Deployments SDK, databricks-sdk, and SQL (AI Functions).
## Fine-tuning
- Mosaic AI Model Training: fine-tune a foundation model on a UC table/Volume → register in UC →
  log run + eval in MLflow. Output deploys ONLY via provisioned throughput / custom serving.
## Serving mode (decision)
- Dev / spiky / low volume      → pay-per-token (Foundation Model APIs).
- Bulk table processing / ETL   → AI Functions (ai_query / ai_classify / ai_extract …).
- Production interactive / fine-tuned → provisioned throughput (size min/max in chunk increments).
- Third-party model under governance → external models.
## Governance
- Register model in UC; wrap endpoint with Unity AI Gateway (usage tracking + inference tables +
  guardrails + rate limits). Restrict CREATE/CAN MANAGE to admins.
## RAG / agents: Vector Search index (UC-governed) + Agent Framework + Agent Evaluation.
## Constitution check: G-* (UC + gateway), R-* (MLflow), S-* (serve rules), C-* (DBU ceiling), Q-* (eval/tests).
```

---

## 7. Step 4 — Tasks

`specs/tasks.md`:

```markdown
| ID | Task                                                         | Satisfies      | Test |
|----|--------------------------------------------------------------|----------------|------|
| T1 | RunConfig schema enforcing Serve + Governance matrices       | FR-003, M-*    | test_matrix.py |
| T2 | Optimizable/eligibility + throughput-chunk preflight          | FR-005, S-2    | test_preflight.py |
| T3 | DBU cost estimator vs ceiling                                 | COST, C-1      | test_cost.py |
| T4 | Fine-tune job (UC dataset) + MLflow run + register in UC      | FR-007, R-1    | test_finetune.py (live) |
| T5 | Provisioned-throughput endpoint + AI Gateway (REST/SDK)       | FR-005/006     | test_deploy.py (live) |
| T6 | OpenAI-compatible client + AI Functions (SQL) + structured-out| FR-009/010     | test_inference.py |
| T7 | Eval gate (tuned beats base) + cost gate                      | Q-2, EVAL      | test_eval_gate.py |
| T8 | Map every FR/AC to a test                                     | Q-3            | test_spec_coverage.py |
```

---

## 8. Step 5 — Implement

### 8.1 Config enforcement (`src/dbx/config.py`)

Rejects platform-rule violations **before** an endpoint is created (saving a failed deploy + DBUs).

```python
"""RunConfig enforces the Databricks Serve + Governance matrices (FR-003)."""
from enum import Enum
from typing import Optional
import yaml
from pydantic import BaseModel, Field, model_validator

class ServeMode(str, Enum):
    pay_per_token = "pay_per_token"
    ai_functions = "ai_functions"
    provisioned_throughput = "provisioned_throughput"
    external_models = "external_models"

class ModelKind(str, Enum):
    foundation = "foundation"; fine_tuned = "fine_tuned"; custom = "custom"; external = "external"

class Gateway(BaseModel):
    usage_tracking: bool = False
    inference_tables: bool = False
    guardrails: bool = False
    rate_limit_qpm: Optional[int] = None       # at least one rate limit for prod

class Throughput(BaseModel):
    optimizable: bool = False                  # from get-model-optimization-info
    chunk_size: int = Field(0, ge=0)           # throughput_chunk_size
    min_units: int = Field(0, ge=0)            # multiples of chunk_size
    max_units: int = Field(0, ge=0)

class RunConfig(BaseModel):
    model_uc_name: str                         # catalog.schema.model
    model_version: Optional[int] = None
    model_kind: ModelKind
    serve_mode: ServeMode
    is_production: bool = False
    gateway: Gateway = Gateway()
    throughput: Optional[Throughput] = None
    cost_ceiling_dbu_per_day: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _enforce(self):
        # M-1: fine-tuned/custom cannot use pay-per-token
        if self.serve_mode is ServeMode.pay_per_token and self.model_kind in (
            ModelKind.fine_tuned, ModelKind.custom):
            raise ValueError("fine-tuned/custom models require provisioned throughput, "
                             "not pay-per-token (AC-1/S-1)")
        if self.serve_mode is ServeMode.provisioned_throughput:
            t = self.throughput
            if t is None or not t.optimizable:
                raise ValueError("provisioned throughput requires an optimizable model (AC-2/M-2)")
            if t.min_units > t.max_units:
                raise ValueError("min throughput must be <= max (AC-4/M-3)")
            if t.chunk_size and (t.min_units % 1 != 0):
                raise ValueError("throughput units must be whole chunk increments (S-2)")
        if self.is_production:
            g = self.gateway
            if not (g.guardrails and g.rate_limit_qpm):
                raise ValueError("production endpoints REQUIRE guardrails + a rate limit (AC-3/M-4)")
            if not (g.usage_tracking and g.inference_tables):
                raise ValueError("production endpoints REQUIRE usage tracking + inference tables (G-2)")
        return self

def load_config(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as fh:
        return RunConfig.model_validate(yaml.safe_load(fh))
```

`configs/serve_finetune.yaml`:
```yaml
model_uc_name: ml.llm_catalog.support_bot_ft
model_version: 3
model_kind: fine_tuned
serve_mode: provisioned_throughput      # fine-tuned → must be provisioned (enforced)
is_production: true
gateway: { usage_tracking: true, inference_tables: true, guardrails: true, rate_limit_qpm: 600 }
throughput: { optimizable: true, chunk_size: 980, min_units: 1960, max_units: 2940 }
cost_ceiling_dbu_per_day: 500
```

### 8.2 Deploy a provisioned-throughput endpoint (REST)

```python
"""Create a provisioned-throughput endpoint from a validated config (FR-005)."""
import os, requests
from dbx.config import RunConfig

def deploy_provisioned(cfg: RunConfig):
    api_root, token = os.environ["DATABRICKS_HOST"], os.environ["DATABRICKS_TOKEN"]
    headers = {"Authorization": f"Bearer {token}"}
    # Preflight: confirm the model is eligible + get the chunk size (S-2)
    info = requests.get(
        f"{api_root}/api/2.0/serving-endpoints/get-model-optimization-info/"
        f"{cfg.model_uc_name}/{cfg.model_version}", headers=headers).json()
    if not info.get("optimizable"):
        raise ValueError("Model is not eligible for provisioned throughput")
    body = {"name": "support-bot-ft",
            "config": {"served_entities": [{
                "entity_name": cfg.model_uc_name, "entity_version": cfg.model_version,
                "min_provisioned_throughput": cfg.throughput.min_units,
                "max_provisioned_throughput": cfg.throughput.max_units}]},
            "ai_gateway": {                                  # G-2 governance
                "usage_tracking_config": {"enabled": True},
                "inference_table_config": {"enabled": True},
                "guardrails": {"input": {"safety": True}, "output": {"safety": True}},
                "rate_limits": [{"calls": cfg.gateway.rate_limit_qpm,
                                 "renewal_period": "minute", "key": "endpoint"}]}}
    return requests.post(f"{api_root}/api/2.0/serving-endpoints", json=body, headers=headers).json()
```

### 8.3 Inference — OpenAI-compatible + AI Functions (SQL)

```python
# Interactive: OpenAI-compatible client against the serving endpoint (FR-009)
from openai import OpenAI
import os
client = OpenAI(base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
                api_key=os.environ["DATABRICKS_TOKEN"])
resp = client.chat.completions.create(
    model="support-bot-ft",
    messages=[{"role": "user", "content": "Classify: Databricks is great!"}],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "cls",
                        "schema": {"type": "object",
                                   "properties": {"sentiment": {"enum": ["positive","negative","neutral"]}},
                                   "required": ["sentiment"]}}})
print(resp.choices[0].message.content)
```

```sql
-- Bulk / ETL: AI Functions apply LLMs to a whole table inside SQL (FR-010)
SELECT id,
       ai_classify(ticket_text, ARRAY('billing','support','product','other')) AS category,
       ai_query('support-bot-ft',
                request => ticket_text,
                returnType => 'STRING')                       AS reply
FROM   ml.crm.support_tickets
WHERE  created_at >= current_date - INTERVAL 1 DAY;
```

### 8.4 Fine-tune (Mosaic AI Model Training) + register in UC + MLflow

```python
# Conceptual flow (API names vary by release — confirm against your workspace):
# 1) launch a fine-tune run on a UC dataset (Mosaic AI Model Training / Foundation Model Fine-tuning)
# 2) the run is tracked in MLflow (params, dataset version, eval metrics)         (R-1)
# 3) the resulting model is registered in Unity Catalog: catalog.schema.model     (G-1)
# 4) deploy via §8.2 provisioned throughput; gate on §eval (Q-2)
```

---

## 9. Cost & Serving-Mode Sizing (DBUs)

Databricks meters **DBUs** per surface — the spec's estimator (T3) projects spend and picks the cheapest mode that meets SLOs:

| Surface / mode | Cost meter | Cheapest when |
|---|---|---|
| **Foundation Model APIs (pay-per-token)** | DBUs per token (model-specific; Llama < Claude; long-context costs more) | spiky / low / variable volume; dev |
| **AI Functions (batch)** | rolls into the SQL warehouse DBU bill | bulk table processing / ETL |
| **Provisioned throughput** | serving DBUs at throughput tiers (model units) | steady high-QPS production; fine-tuned/custom |
| **Vector Search** | query DBUs + index storage GB (+ reindex) | RAG; watch reindex cost on fast-changing corpora |

**Crossover rule (C-2):** pay-per-token wins until sustained volume makes a provisioned tier cheaper than the equivalent per-token DBUs; then switch. Route bulk through **AI Functions**. Set the provisioned **autoscale minimum** deliberately — it's the floor of your idle spend.

> Reported real-world starting spend lands roughly **$1K–$20K/month** in API spend for early workloads; the surprises are usually **Vector Search reindex** on daily-changing corpora and **idle provisioned capacity**. Put your real rates + projected volume into the T3 estimator and let the cost gate decide.

---

## 10. How to Test on Databricks

Because Databricks manages the GPUs, the highest-value tests are **cheap, no-cluster checks of platform-rule + governance correctness and projected DBUs** — before any endpoint is created.

| Layer | Proves | Cluster/Endpoint? | When |
|---|---|---|---|
| **1 Matrix enforcement** | Illegal serve/governance configs rejected (AC-1/2/3/4) | No | Every commit |
| **2 Eligibility preflight** | model is optimizable; throughput in chunk increments (AC-2) | API GET only | Pre-deploy |
| **3 Cost gate** | projected DBU spend ≤ ceiling (AC-7) | No | Every commit |
| **4 Governance check** | endpoint has UC registration + gateway (usage/inference/guardrails/rate-limit) (G-2) | No | Pre-deploy |
| **5 Smoke + structured output** | endpoint returns a completion; JSON validates (AC-5) | Yes | Pre-ship |
| **6 Eval gate** | fine-tune beats base by threshold (AC-6) | Yes | Pre-promotion |
| **7 Spec coverage** | every FR/AC has a test (Q-3) | No | Every commit |

```python
import pytest
from pydantic import ValidationError
from dbx.config import RunConfig

def test_ac1_finetuned_pay_per_token_rejected():       # AC-1 / S-1
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"model_uc_name":"c.s.m","model_kind":"fine_tuned",
            "serve_mode":"pay_per_token","cost_ceiling_dbu_per_day":100})

def test_ac3_prod_without_gateway_rejected():          # AC-3 / G-2
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"model_uc_name":"c.s.m","model_kind":"fine_tuned",
            "serve_mode":"provisioned_throughput","is_production":True,
            "throughput":{"optimizable":True,"chunk_size":980,"min_units":980,"max_units":980},
            "cost_ceiling_dbu_per_day":100})           # gateway not configured
```

> The **governance check** (Layer 4) is what keeps "everything is in Unity Catalog" true — it's the lakehouse equivalent of the self-hosted GPU-fit preflight: a missing guardrail or un-registered model fails CI instead of shipping ungoverned.

---

## 11. Files: Necessary vs. Not

### ✅ Necessary
| File | Why |
|---|---|
| `specs/spec.md` | Serve + Governance matrices, SLOs, cost ceiling, eval gate. |
| `specs/constitution.md` | UC + AI Gateway governance, MLflow repro, serve rules, DBU cost. |
| `src/dbx/config.py` | Enforces platform rules before any endpoint is created. |
| `src/dbx/cost.py` | Projects DBU spend per mode vs ceiling. |
| `configs/*.yaml` | One run = one config (model UC name, mode, throughput, gateway). |
| `tests/test_matrix.py`, `test_cost.py`, `test_eval_gate.py`, `test_spec_coverage.py` | Prove rules + cost + quality. |
| MLflow run + UC model version + deployment manifest | Reproducibility / lineage (R-1/R-3). |
| Databricks secrets / `.gitignore` | Tokens via secrets, never committed. |

### ⚠️ Optional
| File | When |
|---|---|
| Databricks Asset Bundles (DABs) / Terraform | IaC for endpoints + jobs. |
| Vector Search index definition | RAG apps. |
| Agent Framework + Agent Evaluation config | Multi-step agents. |

### ❌ Not necessary (anti-patterns)
| Anti-pattern | Why avoid |
|---|---|
| Clicking together a serving endpoint in the UI with no config in VC | Unreproducible; bypasses the matrix + governance checks. |
| Serving a model not registered in Unity Catalog | Ungoverned; breaks lineage/access control (G-1). |
| Production endpoint with no AI Gateway guardrails/rate limits | No safety, no quota, no usage visibility (G-2). |
| Trying to put a fine-tuned model on pay-per-token | Platform rejects it — fine-tunes need provisioned throughput (S-1). |
| Tokens hard-coded in notebooks | Use Databricks secrets (FR-008). |
| Shipping a fine-tune with no eval gate | Can't prove it beats the base (Q-2). |

---

## 12. Pros & Cons

**Pros:** **lakehouse-native** — AI runs next to your governed data with **one Unity Catalog governance model + MLflow lineage** across all five surfaces; OpenAI-compatible (code ports cleanly); **AI Functions** bring LLMs into SQL for batch/ETL at scale; provisioned throughput gives production performance guarantees; **Unity AI Gateway** centralizes usage tracking, payload logging, guardrails, and rate limits; Vector Search + Agent Framework + Agent Evaluation complete the RAG/agent stack.

**Cons:** **Databricks-centric** (best value only if your data + teams already live there); DBU metering across surfaces takes discipline to forecast (Vector Search reindex + idle provisioned capacity are the usual cost surprises); fine-tuned/custom models can't ride the cheap pay-per-token tier; rapidly evolving APIs (pin versions, confirm fine-tuning product names per release).

**vs Snowflake Cortex:** both put AI next to governed warehouse/lakehouse data. **Databricks** is broader/MLflow-centric with provisioned-throughput serving + an agent stack; **Snowflake** is SQL-first (`AI_COMPLETE`) and keeps everything inside the Snowflake perimeter. Choose by where your data + team already are — see the Snowflake companion guide.

---

## 13. References
- **Databricks — Supported foundation models on Model Serving** (pay-per-token / AI Functions / provisioned throughput / external): <https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview>
- **Databricks — Provisioned throughput Foundation Model APIs** (min/max throughput, optimizable, chunk size): <https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis>
- **Databricks — Unity AI Gateway** (usage tracking, inference tables, guardrails, rate limits): <https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints>
- **Databricks — Mosaic AI Model Training / Fine-tuning**: <https://docs.databricks.com/aws/en/large-language-models/foundation-model-training/>
- **Databricks — AI Functions (ai_query / ai_classify / …)**: <https://docs.databricks.com/aws/en/large-language-models/ai-functions>
- **MLflow** (tracking, registry, eval): <https://mlflow.org/>
- **GitHub Spec Kit** (the SDD methodology applied here): <https://github.com/github/spec-kit>

---
*Illustrative artifact. Databricks AI APIs and product names evolve quickly — confirm serving-endpoint payloads, AI Gateway fields, and fine-tuning APIs against your workspace + release. DBU cost ceilings and SLOs are placeholders; set them from your rates + projected volume and validate with the cost gate + a smoke test before promoting.*
