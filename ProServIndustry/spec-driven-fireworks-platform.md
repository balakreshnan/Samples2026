# Spec-Driven Development on the Fireworks AI Platform

*A spec-first methodology for building on **Fireworks AI** — the managed, OpenAI-compatible platform for **serverless inference, dedicated GPU deployments, managed fine-tuning (LoRA/full SFT + DPO), and batch**. Same SDD method as the companion guides — **Constitution → Spec → Plan → Tasks → Implement → Test** — applied to a **managed platform** where the spec enforces the platform's deployment rules and your cost/compliance guardrails.*

> **Companion to:** the Spec-Driven Development, fine-tuning (HF + NVIDIA), and vLLM/SGLang serving guides. Those are **self-hosted**; **Fireworks is managed** — no GPU babysitting.
>
> **Who this is for:** product/platform teams who want fast open-model inference **and** fine-tuning behind one API, without running GPU infrastructure — but still want reproducibility, governance, and cost control.
>
> **What you'll be able to do:** write a spec that *enforces* Fireworks' deployment rules (serverless vs dedicated, live-merge vs multi-LoRA, BF16-vs-addons), pick the right deployment mode for cost/SLO, and prove behavior + spend with pre-ship tests.

---

## Table of Contents
1. [What Fireworks Is & When to Use It](#1-what-fireworks-is--when-to-use-it)
2. [Platform Concepts (the vocabulary the spec enforces)](#2-platform-concepts)
3. [The SDD Method Applied to a Managed Platform](#3-the-sdd-method-applied-to-a-managed-platform)
4. [Step 1 — The Constitution](#4-step-1--the-constitution)
5. [Step 2 — The Spec (with the enforceable Deployment Matrix)](#5-step-2--the-spec)
6. [Step 3 — The Plan](#6-step-3--the-plan)
7. [Step 4 — Tasks](#7-step-4--tasks)
8. [Step 5 — Implement](#8-step-5--implement)
9. [Cost & Deployment-Mode Sizing](#9-cost--deployment-mode-sizing)
10. [How to Test on a Managed Platform](#10-how-to-test-on-a-managed-platform)
11. [Files: Necessary vs. Not](#11-files-necessary-vs-not)
12. [Pros & Cons](#12-pros--cons)
13. [References](#13-references)

---

## 1. What Fireworks Is & When to Use It

**Fireworks AI** is a production-focused, managed generative-AI platform (founded by ex-Meta PyTorch engineers) that runs 100s of open-weight models (Llama, DeepSeek, Qwen, Mixtral, Kimi K2, GLM, gpt-oss, FLUX) behind **one OpenAI-compatible API**. The four pillars:

- **Serverless inference** — pay-per-token, no GPU config, auto-scaling, no cold starts on popular models.
- **Dedicated (on-demand) GPU deployments** — private deployments on A100/H100/H200/B200/B300, billed per GPU-second, with performance guarantees.
- **Managed fine-tuning** — LoRA SFT, LoRA DPO, full-parameter SFT and DPO; a fine-tuned model is billed at the **same per-token rate as the base** (unusual in the market).
- **Batch API** — ~50% cheaper than serverless for non-real-time bulk jobs (labeling, evals, generation).

Plus function calling, structured JSON output, vision, embeddings, a reranker, **FireOptimizer** (workload-aware autotuner) + **FireAttention** kernels, and SOC 2 / HIPAA / GDPR compliance with optional zero-data-retention.

**Use Fireworks when** you want open-model inference + fine-tuning **without running GPUs**, with predictable economics and an OpenAI-compatible migration path. **Don't** if you need total control of the serving stack or the cheapest possible self-hosted token cost at scale (→ self-host with vLLM/SGLang).

---

## 2. Platform Concepts

The spec enforces choices among these — get them wrong and the deployment simply won't work:

- **Account / User / roles** — billing + quotas at the account level.
- **Models**: **base models** (`accounts/<ACCOUNT_ID>/models/<MODEL_ID>`, e.g. `llama-v3p1-70b-instruct`) and **LoRA addons** (small fine-tuned adapters).
- **Deployments**: **serverless** (shared, pay-per-token, base models only) vs **dedicated/on-demand** (private GPUs, billed GPU-second, base **and** LoRA).
- **Fine-tuning job** — offline training producing a LoRA addon (or full fine-tune) from an immutable **dataset**.
- **LoRA deployment methods** — **live merge** (LoRA merged into base at deploy time → base-model performance, 1 per deployment) vs **multi-LoRA** (many adapters loaded dynamically onto one base → share GPUs, some per-request overhead).

**The hard platform rules (these become the enforceable matrix):**
1. **LoRA addons cannot be served serverless** — they require a **dedicated** deployment.
2. **Multi-LoRA addons require a BF16 deployment shape** — `--enable-addons` is **not** supported on FP8/FP4 shapes (and many base models default to FP8/FP4).
3. `supportsServerless: true` + `supportsLora: true` on a model are **mutually exclusive in deployment** — serverless applies to the base; the LoRA needs dedicated.

---

## 3. The SDD Method Applied to a Managed Platform

On a managed platform, "done" is still SLOs (TTFT, throughput) **plus** an explicit **cost guardrail** and **compliance posture** — because the platform abstracts the GPUs, the spec's job shifts to enforcing *deployment-mode correctness* and *spend control*.

```
SPECIFY ──────────▶ PLAN ──────────▶ TASKS ─────────▶ IMPLEMENT ──────▶ (PRE-SHIP GATES)
model, deploy mode, serverless vs    dataset → FT job  firectl / SDK     deployment matrix valid?
LoRA method, SLOs,  dedicated; live- → deploy → client  from a validated  cost ceiling respected?
cost ceiling,       merge vs multi-  → eval + cost test config            structured-out ok?
compliance, matrix  LoRA; batch?                                          eval gate passed?
```

---

## 4. Step 1 — The Constitution

`specs/constitution.md`:

```markdown
# Project Constitution — Fireworks AI

## Security & Compliance
- S-1: FIREWORKS_API_KEY comes from an env var; never hard-coded or committed.
- S-2: For regulated data, the deployment MUST use a compliance-appropriate mode
       (e.g., zero-data-retention / dedicated) and that posture is recorded.
- S-3: Dataset provenance + license recorded; PII screened before any fine-tuning job.

## Deployment Integrity (platform rules — enforced by config.py)
- D-1: A run declares deploy_mode ∈ {serverless, dedicated} and, if a LoRA, a lora_method
       ∈ {live_merge, multi_lora}.
- D-2: A LoRA addon MUST NOT be deployed serverless (dedicated only).
- D-3: multi_lora REQUIRES a BF16 deployment shape (FP8/FP4 do not support addons).
- D-4: A fine-tuned model is referenced by its full resource name (accounts/.../models/...).

## Cost
- C-1: Every run declares a cost ceiling (e.g., $/day or $/1M tokens) and the expected mode's
       price basis (serverless $/token, dedicated $/GPU-hr, batch = 50% of serverless).
- C-2: Prefer serverless for spiky/low volume; switch to dedicated only above the crossover
       volume; use the Batch API for non-real-time bulk (50% off).
- C-3: Dedicated deployments MUST have autoscale-to-zero or a documented shutdown plan
       (GPU-second billing accrues 24/7 otherwise).

## Reliability & Quality
- R-1: Declare SLOs (TTFT, throughput) and verify against Fireworks' published/measured numbers.
- Q-1: Structured output uses the platform's JSON/function-calling mode, not free-text parsing.
- Q-2: Fine-tunes pass an eval gate (beat base by a threshold) before promotion.
- Q-3: Every requirement (FR/AC) maps to >= 1 automated test.
```

---

## 5. Step 2 — The Spec

`specs/spec.md`:

```markdown
# Spec: Fireworks Deployment / Fine-Tune

## Functional Requirements
- FR-001: A run is fully determined by one config (model, deploy_mode, lora_method, dataset,
          SLOs, cost ceiling, compliance).
- FR-002: Auth via FIREWORKS_API_KEY from env; OpenAI-compatible base_url.
- FR-003: The config validates the Deployment Matrix and FAILS before any API call on violation.
- FR-004: A LoRA addon → deploy_mode MUST be dedicated (D-2).
- FR-005: lora_method=multi_lora → deployment shape MUST be bf16 (D-3).
- FR-006: A fine-tune declares a dataset with provenance + license; PII screened (S-3).
- FR-007: A fine-tune MUST pass an eval gate (beat base by threshold) before promotion (Q-2).
- FR-008: The run declares a cost ceiling and the price basis for its mode (C-1).
- FR-009: Structured output uses JSON-mode / function calling, not free-text parsing.
- FR-010: Dedicated deployments declare an autoscale/shutdown policy (C-3).

## ── DEPLOYMENT MATRIX (enforced by config.py) ──
| Field          | serverless (base) | dedicated (base) | dedicated (LoRA) |
|----------------|-------------------|------------------|-------------------|
| model type     | base only         | base             | LoRA addon        |
| LoRA allowed   | ❌ (D-2)          | n/a              | ✅                |
| lora_method    | n/a               | n/a              | live_merge OR multi_lora |
| bf16 shape req | n/a               | n/a              | REQUIRED if multi_lora (D-3) |
| price basis    | $/token           | $/GPU-hour       | $/GPU-hour        |
| best for       | spiky / low vol   | steady high vol  | many variants (multi) / 1 variant (merge) |

Rules:
- M-1: deploy_mode=serverless + model is LoRA → reject (D-2).
- M-2: lora_method=multi_lora + shape != bf16 → reject (D-3).
- M-3: deploy_mode=dedicated + no shutdown/autoscale policy → reject (C-3/FR-010).
- M-4: fine_tune=true + dataset missing license → reject (FR-006/S-3).

## SLOs & Cost (definition of done)
- SLO-TTFT / SLO-TPS: meet declared targets (verify vs Fireworks measured numbers).
- COST-CEILING: projected spend at expected volume <= declared ceiling (else reject/redesign).
- EVAL-GATE (fine-tunes): tuned model beats base by >= threshold on held-out set.

## Acceptance Criteria
- AC-1: a LoRA model with deploy_mode=serverless → rejected. (M-1/D-2)
- AC-2: multi_lora on an fp8 shape → rejected. (M-2/D-3)
- AC-3: dedicated deployment with no shutdown policy → rejected. (M-3)
- AC-4: fine-tune dataset with no license → rejected. (M-4)
- AC-5: a structured-output request returns JSON validating against the schema. (FR-009)
- AC-6: a fine-tune failing the eval gate → promotion blocked. (FR-007)
- AC-7: projected cost > ceiling → run flagged for redesign (mode switch / batch). (COST-CEILING)
```

---

## 6. Step 3 — The Plan

`specs/plan.md`:

```markdown
# Plan: Fireworks
## Access
- OpenAI-compatible: base_url https://api.fireworks.ai/inference/v1, or `from fireworks import Fireworks`.
- Control plane via firectl (deployments, fine-tuning jobs, datasets).
## Deployment mode (decision)
- Spiky / low / variable volume        → serverless (pay-per-token).
- Steady high volume above crossover   → dedicated GPU (A100/H100/H200/B200), GPU-second billing.
- Non-real-time bulk (labeling/evals)  → Batch API (50% off serverless).
## Fine-tuning (managed)
- LoRA SFT / LoRA DPO / full SFT / full DPO. Output = LoRA addon (or full FT) by resource name.
- Deploy a LoRA on DEDICATED only: live_merge (1 variant, no overhead) or multi_lora (many, bf16).
## Cost control
- Estimate $ at expected volume per mode; pick the cheapest meeting SLOs; autoscale-to-zero on dedicated.
## Compliance: pick zero-data-retention / appropriate mode for regulated data; record posture.
## Constitution check: S-* (auth/compliance/data), D-* (platform rules), C-* (cost), R/Q-* (SLO/eval/tests).
```

---

## 7. Step 4 — Tasks

`specs/tasks.md`:

```markdown
| ID | Task                                                       | Satisfies      | Test |
|----|------------------------------------------------------------|----------------|------|
| T1 | RunConfig schema enforcing the Deployment Matrix           | FR-003, M-*    | test_matrix.py |
| T2 | Cost estimator: project spend per mode vs ceiling          | FR-008, COST   | test_cost.py |
| T3 | Dataset validation (provenance/license/PII) for fine-tunes | FR-006, S-3    | test_data.py |
| T4 | Fine-tune job + deploy (firectl/SDK) from validated config | FR-004/005     | test_deploy.py (live) |
| T5 | OpenAI-compatible client + JSON/function-calling helper    | FR-009         | test_structured.py |
| T6 | Eval gate (tuned beats base) + cost gate                   | FR-007, EVAL   | test_eval_gate.py |
| T7 | Map every FR/AC to a test                                  | Q-3            | test_spec_coverage.py |
```

---

## 8. Step 5 — Implement

### 8.1 Config enforcement (`src/fw/config.py`)

Rejects platform-rule violations **before** an API call (which would otherwise fail at deploy time and waste a round-trip / a job).

```python
"""RunConfig enforces the Fireworks Deployment Matrix (FR-003)."""
from enum import Enum
from typing import Optional
import yaml
from pydantic import BaseModel, Field, model_validator

class DeployMode(str, Enum):
    serverless = "serverless"; dedicated = "dedicated"

class LoraMethod(str, Enum):
    live_merge = "live_merge"; multi_lora = "multi_lora"

class Shape(str, Enum):
    bf16 = "bf16"; fp8 = "fp8"; fp4 = "fp4"

class Dataset(BaseModel):
    name: str
    license: str                                   # REQUIRED (S-3)
    pii_screened: bool = False

class RunConfig(BaseModel):
    model: str                                     # accounts/<ACCOUNT_ID>/models/<MODEL_ID>
    is_lora_addon: bool = False
    deploy_mode: DeployMode = DeployMode.serverless
    lora_method: Optional[LoraMethod] = None
    shape: Shape = Shape.bf16
    fine_tune: bool = False
    dataset: Optional[Dataset] = None
    cost_ceiling_usd_per_day: float = Field(..., gt=0)   # FR-008
    shutdown_policy: Optional[str] = None          # e.g. "autoscale-to-zero", "nightly-teardown"
    api_key_env: str = "FIREWORKS_API_KEY"

    @model_validator(mode="after")
    def _enforce(self):
        if self.is_lora_addon and self.deploy_mode is DeployMode.serverless:
            raise ValueError("LoRA addons cannot be served serverless — use dedicated (AC-1/D-2)")
        if self.lora_method is LoraMethod.multi_lora and self.shape is not Shape.bf16:
            raise ValueError("multi_lora requires a bf16 shape (FP8/FP4 reject addons) (AC-2/D-3)")
        if self.deploy_mode is DeployMode.dedicated and not self.shutdown_policy:
            raise ValueError("dedicated deployments REQUIRE a shutdown/autoscale policy (AC-3/M-3)")
        if self.fine_tune:
            if self.dataset is None or not self.dataset.license:
                raise ValueError("fine-tune REQUIRES a dataset with a license (AC-4/M-4)")
            if not self.dataset.pii_screened:
                raise ValueError("fine-tune dataset MUST be PII-screened (S-3)")
        return self

def load_config(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as fh:
        return RunConfig.model_validate(yaml.safe_load(fh))
```

`configs/serve_lora.yaml`:
```yaml
model: accounts/my-acct/models/support-bot-lora
is_lora_addon: true
deploy_mode: dedicated          # LoRA addons require dedicated (enforced)
lora_method: multi_lora         # serving several variants of one base
shape: bf16                     # required for multi-LoRA addons (enforced)
shutdown_policy: autoscale-to-zero
cost_ceiling_usd_per_day: 200
```

### 8.2 Fine-tune + deploy (firectl / SDK)

```bash
# Managed fine-tuning (LoRA SFT) and deployment via firectl (control plane):
firectl create dataset my-dataset --file train.jsonl
firectl create fine-tuning-job --base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
    --dataset my-dataset --kind sft --lora                       # → produces a LoRA addon
# Deploy the LoRA (dedicated). Live-merge = one variant, base-model performance:
firectl deployment create "accounts/<ACCOUNT_ID>/models/<FINE_TUNED_MODEL_ID>"
# Multi-LoRA: deploy a base with addons enabled (bf16), then load adapters dynamically:
firectl deployment create <base> --enable-addons               # bf16 shape required
```

### 8.3 Client + structured output (`src/fw/client.py`)

```python
"""OpenAI-compatible inference against Fireworks (serverless or dedicated)."""
import os
from openai import OpenAI

client = OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ["FIREWORKS_API_KEY"])      # FR-002 / S-1

SCHEMA = {"type": "object",
          "properties": {"sentiment": {"enum": ["positive", "negative", "neutral"]}},
          "required": ["sentiment"]}

resp = client.chat.completions.create(
    model="accounts/my-acct/models/support-bot-lora",         # full resource name (FR-004 / D-4)
    messages=[{"role": "user", "content": "Classify: Fireworks is fast!"}],
    response_format={"type": "json_object", "schema": SCHEMA}, # structured output (FR-009)
)
print(resp.choices[0].message.content)
# (Fireworks' native SDK: `from fireworks import Fireworks; client = Fireworks()` works the same way.)
```

---

## 9. Cost & Deployment-Mode Sizing

On Fireworks the question isn't VRAM — the platform handles GPUs — it's **which mode is cheapest for your volume while meeting SLOs**. The spec's cost estimator (T2) projects spend per mode:

| Mode | Price basis (illustrative — verify current) | Cheapest when |
|---|---|---|
| **Serverless** | per-token: ~$0.10/1M (<4B), ~$0.20/1M (4–16B), ~$0.90/1M (>16B dense); MoE/frontier higher | spiky / low / variable volume; dev & prototyping |
| **Batch API** | **50% of serverless** (in + out); completes within ~24h | non-real-time bulk: labeling, evals, dataset generation |
| **Dedicated** | per **GPU-hour**: ~A100-80 $2.90, H100/H200 ~$6–7, B200 ~$9–10, B300 ~$12 | steady high volume above the serverless crossover; or LoRA serving |

**The crossover rule (C-2):** serverless wins until your sustained volume makes a dedicated GPU cheaper than the equivalent per-token spend; then switch. **Always** route non-real-time bulk through Batch (50% off). **Always** put a shutdown/autoscale policy on dedicated — GPU-second billing accrues 24/7 whether or not you send traffic (C-3).

> **Worked logic:** if dedicated H100 ≈ $6/hr = $144/day, serverless is cheaper until your daily token spend on the same model exceeds ~$144. Below that, stay serverless; above it (and with steady traffic), dedicated gives predictable cost + guaranteed capacity. Put the real prices and your projected volume into the T2 estimator and let the cost gate decide.

---

## 10. How to Test on a Managed Platform

Because the platform runs the GPUs, the highest-value tests are **cheap, no-infra checks of deployment-rule correctness and projected cost** — caught before any API call.

| Layer | Proves | Calls API? | When |
|---|---|---|---|
| **1 Matrix enforcement** | Illegal deploy combos rejected (AC-1/2/3/4) | No | Every commit |
| **2 Cost gate** | Projected spend at expected volume ≤ ceiling (AC-7) | No | Every commit |
| **3 Dataset validation** | License + PII screen present for fine-tunes (AC-4) | No | Every commit |
| **4 Smoke** | endpoint returns a completion; structured output validates (AC-5) | Yes | Pre-ship |
| **5 Eval gate** | tuned model beats base by threshold (AC-6) | Yes | Pre-promotion |
| **6 Spec coverage** | every FR/AC has a test (Q-3) | No | Every commit |

```python
import pytest
from pydantic import ValidationError
from fw.config import RunConfig

def test_ac1_lora_serverless_rejected():           # AC-1 / D-2
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"model":"accounts/a/models/m-lora","is_lora_addon":True,
            "deploy_mode":"serverless","cost_ceiling_usd_per_day":100})

def test_ac2_multilora_fp8_rejected():             # AC-2 / D-3
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"model":"accounts/a/models/m","is_lora_addon":True,
            "deploy_mode":"dedicated","lora_method":"multi_lora","shape":"fp8",
            "shutdown_policy":"autoscale-to-zero","cost_ceiling_usd_per_day":100})

def test_ac7_cost_gate(projected_spend):           # AC-7 / COST-CEILING
    assert projected_spend.usd_per_day <= projected_spend.ceiling   # else: switch mode / batch
```

> The **cost gate** (Layer 2) is the managed-platform equivalent of the self-hosted GPU-fit preflight: it turns a surprise invoice into a failing test. Wire it to real, current Fireworks prices and your projected volume.

---

## 11. Files: Necessary vs. Not

### ✅ Necessary
| File | Why |
|---|---|
| `specs/spec.md` | Deployment Matrix + SLOs + cost ceiling + eval gate. |
| `specs/constitution.md` | Security/compliance/platform/cost rules. |
| `src/fw/config.py` | Enforces the platform rules before any API call. |
| `src/fw/cost.py` | Projects spend per mode vs ceiling. |
| `configs/*.yaml` | One run = one config (model, mode, lora_method, dataset, ceiling). |
| `tests/test_matrix.py`, `test_cost.py`, `test_eval_gate.py`, `test_spec_coverage.py` | Prove rules + cost + quality. |
| `.env.example`, `.gitignore` | `FIREWORKS_API_KEY` via env, never committed. |
| Run manifest (model resource name + job ID + dataset + metrics) | Reproducibility / audit. |

### ⚠️ Optional
| File | When |
|---|---|
| `firectl` scripts / IaC | Repeatable deployment lifecycle. |
| Batch-job submission scripts | Bulk offline workloads. |
| Compliance evidence (zero-data-retention config) | Regulated data. |

### ❌ Not necessary (anti-patterns)
| Anti-pattern | Why avoid |
|---|---|
| `FIREWORKS_API_KEY` hard-coded in code/notebook | Security breach (S-1). |
| Trying to deploy a LoRA serverless | Platform rejects it — LoRA needs dedicated (D-2). |
| multi-LoRA on an FP8/FP4 base shape | Addons unsupported on FP8/FP4 (D-3). |
| Dedicated deployment with no shutdown policy | GPU-second billing accrues 24/7 (C-3). |
| Real-time API for bulk labeling/evals | Use Batch (50% off) (C-2). |
| Shipping a fine-tune without an eval gate | You can't prove it beats the base (Q-2). |

---

## 12. Pros & Cons

**Pros:** **no GPU ops** (managed serverless + dedicated + batch in one platform); OpenAI-compatible (drop-in migration); **fine-tuning at base-model token price** (unusual); among the fastest TTFT/throughput on open models (FireOptimizer/FireAttention); broad + fresh model catalog (day-of release); Batch API 50% discount; function calling, structured output, vision, embeddings, reranker; SOC 2 / HIPAA / GDPR + zero-data-retention.

**Cons:** **less control** than self-hosting (you can't tune the serving stack); per-token cost can exceed self-hosting at very high sustained volume; **platform rules constrain you** (LoRA needs dedicated; addons need BF16; serverless ≠ LoRA); dedicated GPU-second billing accrues 24/7 without a shutdown policy; vendor dependency. *If you need full control or the cheapest token at scale, self-host with vLLM/SGLang.*

**vs self-hosting (vLLM/SGLang):** Fireworks trades **control for convenience** — no GPU sizing, no preflight, no on-call, but you pay the platform margin and live within its deployment rules. Use Fireworks to move fast and for fine-tuning + batch; self-host when control or scale economics dominate.

---

## 13. References
- **Fireworks AI Docs — Concepts** (accounts, models, deployments, datasets, fine-tuning jobs): <https://docs.fireworks.ai/getting-started/concepts>
- **Fireworks — Deploying Fine-Tuned Models** (live merge vs multi-LoRA; BF16-vs-addons; LoRA needs dedicated): <https://docs.fireworks.ai/fine-tuning/deploying-loras>
- **Fireworks — Fine-tuning** (LoRA/full SFT + DPO): <https://docs.fireworks.ai/fine-tuning/fine-tuning-models>
- **Fireworks — Querying / OpenAI-compatible API + structured output**: <https://docs.fireworks.ai/>
- **Fireworks — Pricing** (serverless $/token, dedicated $/GPU-hr, batch 50%): <https://fireworks.ai/pricing>
- **GitHub Spec Kit** (the SDD methodology applied here): <https://github.com/github/spec-kit>

---
*Illustrative artifact. Fireworks features, model IDs, and prices change frequently — verify deployment rules, `firectl`/SDK syntax, and current pricing against the live docs before shipping. Cost ceilings and SLO numbers are placeholders; set them from your real volume and validate with the cost gate + a smoke test.*
