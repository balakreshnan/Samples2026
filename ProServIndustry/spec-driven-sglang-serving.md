# Spec-Driven Development for LLM Serving with SGLang

*A spec-first methodology for deploying and operating LLM inference with **SGLang** — the LMSYS high-performance serving framework whose **RadixAttention** gives automatic KV-cache reuse for agentic, multi-turn, and RAG workloads. Same SDD method as the companion guides — **Constitution → Spec → Plan → Tasks → Implement → Test** — applied to **serving**.*

> **Companion to:** the Spec-Driven Development, fine-tuning (HF + NVIDIA), and vLLM serving guides.
>
> **Who this is for:** platform/MLOps engineers self-hosting open-weight models who want SGLang's prefix-cache advantage **and** a reproducible, governed, SLO-backed deployment — not a `launch_server` command nobody can reproduce.
>
> **What you'll be able to do:** write a serving spec that *enforces* model/parallelism/quantization/memory/structured-output/GPU rules before launch, exploit RadixAttention deliberately, hit declared latency/throughput SLOs, and prove the endpoint with pre-launch tests.

---

## Table of Contents
1. [What SGLang Is & When to Use It](#1-what-sglang-is--when-to-use-it)
2. [The SDD Method Applied to Serving](#2-the-sdd-method-applied-to-serving)
3. [Step 1 — The Constitution](#3-step-1--the-constitution)
4. [Step 2 — The Spec (with the enforceable Serving Matrix + SLOs)](#4-step-2--the-spec)
5. [Step 3 — The Plan](#5-step-3--the-plan)
6. [Step 4 — Tasks](#6-step-4--tasks)
7. [Step 5 — Implement](#7-step-5--implement)
8. [Serving GPU Sizing (Inference VRAM + the prefix-cache lever)](#8-serving-gpu-sizing)
9. [How to Test a Serving Deployment](#9-how-to-test-a-serving-deployment)
10. [Files: Necessary vs. Not](#10-files-necessary-vs-not)
11. [Pros & Cons](#11-pros--cons)
12. [References](#12-references)

---

## 1. What SGLang Is & When to Use It

**SGLang** (Structured Generation Language) is an open-source, high-performance serving framework for LLMs and multimodal models, developed by UC Berkeley/Stanford researchers and hosted by **LMSYS** (now in the PyTorch ecosystem). It powers 400,000+ GPUs in production (xAI/Grok, NVIDIA, AMD, LinkedIn, Cursor). Its core innovations:

- **RadixAttention** — automatic KV-cache **reuse** across requests with shared prefixes, stored in a radix tree. When a new request shares a prefix (a system prompt, few-shot examples, or prior conversation turns), SGLang **skips recomputing it** and starts from the branch point. On workloads with 60%+ prefix overlap this yields large TTFT reductions; agent fleets see 75–95% cache hit rates.
- **Zero-overhead scheduler + continuous batching** — keeps GPU utilization high.
- **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) — drop-in `base_url` swap.
- **Structured output** via constrained decoding (Compressed FSM / xgrammar) — JSON schema, regex, or grammar with near-zero overhead.
- **Multi-GPU** TP/PP/DP/EP, **quantization** (FP8/INT4/AWQ/GPTQ + FP8 KV cache), **speculative decoding** (DFlash/Spec V2), multimodal, and a Python frontend language for complex generation programs.

**Use SGLang when** your workload has **heavy shared-prefix structure** — agents with fixed tool/system prompts, multi-turn chat, RAG with a constant context — where RadixAttention's automatic cache reuse is a decisive win. **Don't** self-host at all if you'd rather pay per token (→ Fireworks).

---

## 2. The SDD Method Applied to Serving

As with any serving engine, "done" = meeting **SLOs**: TTFT, inter-token latency, throughput, p99 latency at a target concurrency. With SGLang there's an extra lever the spec should make explicit: **prefix-cache hit rate** — the metric that determines whether you're getting RadixAttention's benefit.

```
SPECIFY ─────────▶ PLAN ──────────▶ TASKS ─────────▶ IMPLEMENT ───────▶ (PRE-LAUNCH GATES)
model, SLOs,       GPU + TP/DP +    config → launch   launch_server      config valid?
mem-fraction,      quant + mem-     → client → load   from a validated   GPU-fit preflight?
structured-out,    fraction +       → SLO + cache-    config (not        health + SLO met?
prefix-cache       prefix-cache     hit test          ad-hoc flags)      cache-hit target met?
target, the matrix plan
```

---

## 3. Step 1 — The Constitution

`specs/constitution.md`:

```markdown
# Project Constitution — SGLang Serving

## Reliability & SLOs
- R-1: Declare SLOs: TTFT p95, inter-token latency, throughput, p99 latency AT target concurrency.
       Miss them in load test → not promoted.
- R-2: Health endpoint + metrics (--enable-metrics) wired to readiness probe + monitoring.
- R-3: For prefix-heavy workloads, declare a TARGET prefix-cache hit rate and verify it (the
       whole reason to pick SGLang). RadixAttention is on by default — measure that it helps.

## Security
- S-1: Front the endpoint with an auth gateway / API key; never expose unauthenticated.
- S-2: --log-requests only in non-prod, or scrub PII (request logs may contain user content).

## Reproducibility
- RP-1: Pin SGLang version + Docker image digest. Record model revision.
- RP-2: All server args live in a --config YAML in version control; no ad-hoc CLI flags.
- RP-3: If determinism is required, use --enable-deterministic-inference and document the
       throughput cost (batch-invariant ops are slower).

## Cost / Hardware
- H-1: Declare gpu_type, tp, dp, num_nodes. TP MUST stay within a node.
- H-2: GPU-fit preflight (weights + KV pool + overhead <= usable VRAM) MUST pass before launch.
- H-3: --mem-fraction-static declared (default 0.9). Too high starves decode; too low limits cache.

## Quality
- Q-1: Structured output uses constrained decoding (JSON schema/regex/grammar), not free-text parsing.
- Q-2: Every requirement (FR/AC) maps to >= 1 automated test.
```

---

## 4. Step 2 — The Spec

`specs/spec.md`:

```markdown
# Spec: SGLang Serving Deployment

## Functional Requirements
- FR-001: Deployment fully determined by one --config YAML (model, server args, GPU plan,
          structured-output, SLOs, prefix-cache target).
- FR-002: OpenAI-compatible API; endpoint fronted by auth.
- FR-003: The config validates the Serving Matrix and FAILS before launch on any violation.
- FR-004: tp MUST divide total GPUs AND tp <= gpus_per_node. dp uses the Model Gateway (router).
- FR-005: GPU-fit preflight passes: weights + KV pool(mem-fraction) + overhead <= usable VRAM.
- FR-006: FP8 weights/KV-cache REQUIRE Hopper/Blackwell GPUs.
- FR-007: mem_fraction_static in (0, 0.95]; chunked_prefill_size set for long prompts.
- FR-008: If structured output is required, requests carry a JSON-schema/regex/grammar constraint.
- FR-009: enable_metrics=true; health probe gates traffic.
- FR-010: For prefix-heavy workloads, a prefix-cache hit-rate target is declared (R-3).

## ── SERVING MATRIX (enforced by config.py) ──
| Field                  | Rule |
|------------------------|------|
| tp (tensor parallel)   | divides total_gpus; <= gpus_per_node (FR-004) |
| dp (data parallel)     | requires the Model Gateway/router when > 1 |
| quantization           | {none, fp8, awq, gptq}; fp8 → Hopper/Blackwell (FR-006) |
| kv_cache_dtype         | {auto, fp8_e4m3, fp8_e5m2}; fp8 → Hopper/Blackwell |
| mem_fraction_static    | 0 < x <= 0.95 (FR-007) |
| chunked_prefill_size   | REQUIRED (> 0) when max prompt is long |
| structured_output      | if required → a constraint is set on requests (FR-008) |
| prefix_cache_target    | REQUIRED (0–1) for prefix-heavy workloads (FR-010) |

## Service Level Objectives (definition of done)
- SLO-TTFT: p95 < 400 ms at concurrency=32 (lower than baseline thanks to RadixAttention).
- SLO-TPS:  aggregate >= 2,000 output tokens/sec at concurrency=32.
- SLO-P99:  p99 end-to-end < 7 s for a 256-token completion.
- SLO-CACHE: measured prefix-cache hit rate >= prefix_cache_target on the test workload.
(Examples — set yours per workload + hardware.)

## Acceptance Criteria
- AC-1: tp=4 on a single 2-GPU node → rejected. (FR-004)
- AC-2: quantization=fp8 on A100 → rejected. (FR-006)
- AC-3: mem_fraction_static=0.99 → rejected. (FR-007)
- AC-4: GPU-fit preflight where est_vram > usable → rejected. (FR-005)
- AC-5: a structured-output request returns JSON validating against the schema. (FR-008)
- AC-6: load test missing any SLO (incl. SLO-CACHE) → NOT promoted. (R-1/R-3)
```

---

## 5. Step 3 — The Plan

`specs/plan.md`:

```markdown
# Plan: SGLang Serving
## Engine
- SGLang (pin version + image digest). Launch via `python -m sglang.launch_server --config <yaml>`.
- For data parallelism / load balancing, front with the SGLang Model Gateway (former Router).
## GPU & parallelism
- Single GPU if it fits; else --tp N within ONE node. --dp N (via router) for throughput.
- Quantize (FP8/AWQ/GPTQ) to fit larger models. FP8 → H100/H200/B200.
## Memory & cache (the SGLang lever)
- --mem-fraction-static reserves KV-cache pool (start 0.9, or 0.92 on a single-model 80GB H100).
- RadixAttention is on by default → big win when system prompts/history are shared. Measure hit rate.
- --chunked-prefill-size for long prompts to avoid prefill OOM.
## Structured output
- Constrained decoding (xgrammar) with JSON schema / regex / grammar.
## Ops: --enable-metrics, health probe, auth gateway, optional --enable-deterministic-inference.
## Constitution check: R-* (SLOs/cache/probes), S-* (auth/logs), RP-* (pin/config-in-VC),
##   H-* (preflight/mem-fraction), Q-* (structured output + tests).
```

---

## 6. Step 4 — Tasks

`specs/tasks.md`:

```markdown
| ID | Task                                                     | Satisfies     | Test |
|----|----------------------------------------------------------|---------------|------|
| T1 | ServingConfig schema enforcing the Serving Matrix        | FR-003        | test_matrix.py |
| T2 | gpu_preflight: weights + KV pool(mem-fraction) + overhead | FR-005, H-2   | test_preflight.py |
| T3 | Render sglang.launch_server --config YAML from config    | FR-001        | test_render.py |
| T4 | OpenAI client + constrained-decoding helper              | FR-008        | test_structured.py |
| T5 | Load test + prefix-cache hit-rate measurement (SLO gate) | R-1/R-3, SLO-*| test_load_slo.py |
| T6 | Map every FR/AC to a test                                | Q-2           | test_spec_coverage.py |
```

---

## 7. Step 5 — Implement

### 7.1 Config enforcement (`src/serving/config.py`)

```python
"""ServingConfig enforces the SGLang Serving Matrix (FR-003)."""
from enum import Enum
from typing import Optional
import yaml
from pydantic import BaseModel, Field, model_validator

FP8_GPUS = {"H100-80", "H200-141", "B200-192"}

class Quant(str, Enum):
    none="none"; fp8="fp8"; awq="awq"; gptq="gptq"

class GPUPlan(BaseModel):
    gpu_type: str
    num_gpus: int = Field(1, gt=0)
    num_nodes: int = Field(1, gt=0)
    tp: int = Field(1, gt=0)               # tensor parallel
    dp: int = Field(1, gt=0)               # data parallel (needs router if > 1)

class ServingConfig(BaseModel):
    model_path: str
    model_revision: str = "main"
    quantization: Quant = Quant.none
    kv_cache_dtype: str = "auto"           # auto | fp8_e4m3 | fp8_e5m2
    mem_fraction_static: float = Field(0.9, gt=0, le=0.95)   # FR-007
    chunked_prefill_size: Optional[int] = None
    long_prompts: bool = False
    structured_output_required: bool = False
    prefix_heavy_workload: bool = False
    prefix_cache_target: Optional[float] = Field(None, ge=0, le=1)
    enable_deterministic_inference: bool = False
    gpu: GPUPlan

    @model_validator(mode="after")
    def _enforce(self):
        g = self.gpu
        gpus_per_node = g.num_gpus // g.num_nodes
        if g.num_gpus % g.tp != 0:
            raise ValueError("tp must divide num_gpus (FR-004)")
        if g.tp > gpus_per_node:
            raise ValueError("tp must stay within a node (AC-1/FR-004)")
        if self.quantization is Quant.fp8 and g.gpu_type not in FP8_GPUS:
            raise ValueError("fp8 requires Hopper/Blackwell (AC-2/FR-006)")
        if self.kv_cache_dtype.startswith("fp8") and g.gpu_type not in FP8_GPUS:
            raise ValueError("fp8 KV cache requires Hopper/Blackwell (FR-006)")
        if self.long_prompts and not self.chunked_prefill_size:
            raise ValueError("long_prompts REQUIRES chunked_prefill_size (FR-007)")
        if self.prefix_heavy_workload and self.prefix_cache_target is None:
            raise ValueError("prefix-heavy workload REQUIRES a prefix_cache_target (FR-010/R-3)")
        return self

def load_config(path: str) -> ServingConfig:
    with open(path, encoding="utf-8") as fh:
        return ServingConfig.model_validate(yaml.safe_load(fh))
```

`configs/llama8b.yaml`:
```yaml
model_path: meta-llama/Meta-Llama-3-8B-Instruct
quantization: none
mem_fraction_static: 0.92
prefix_heavy_workload: true        # agentic / multi-turn → RadixAttention matters
prefix_cache_target: 0.75
structured_output_required: true
gpu: { gpu_type: H100-80, num_gpus: 1, num_nodes: 1, tp: 1, dp: 1 }
```

### 7.2 Render & launch (`src/serving/render.py`)

```python
"""Turn a validated ServingConfig into a reproducible sglang.launch_server config (FR-001)."""
import yaml
from .config import ServingConfig

def render_sglang_yaml(cfg: ServingConfig, path: str = "configs/_sglang_args.yaml") -> str:
    args = {
        "model-path": cfg.model_path,
        "revision": cfg.model_revision,
        "host": "0.0.0.0", "port": 30000,
        "tp-size": cfg.gpu.tp,
        "mem-fraction-static": cfg.mem_fraction_static,
        "enable-metrics": True,
    }
    if cfg.quantization.value != "none":
        args["quantization"] = cfg.quantization.value
    if cfg.kv_cache_dtype != "auto":
        args["kv-cache-dtype"] = cfg.kv_cache_dtype
    if cfg.chunked_prefill_size:
        args["chunked-prefill-size"] = cfg.chunked_prefill_size
    if cfg.enable_deterministic_inference:
        args["enable-deterministic-inference"] = True
    with open(path, "w") as fh:
        yaml.safe_dump(args, fh)
    return path
```

```bash
# Launch from the rendered config (single GPU). For TP add --tp; for DP use the router:
python -m sglang.launch_server --config configs/_sglang_args.yaml
#   Multi-GPU TP:   python -m sglang.launch_server --model-path <m> --tp 2
#   DP + router:    python -m sglang_router.launch_server --model-path <m> --dp 2 --tp 1
#   Server: http://localhost:30000/v1   ·   metrics: enabled
```

### 7.3 Client + structured output (`src/serving/client.py`)

```python
from openai import OpenAI
import os

client = OpenAI(base_url="http://localhost:30000/v1", api_key=os.getenv("SGLANG_API_KEY", "EMPTY"))

# Constrained JSON via SGLang's structured-output (json_schema) response_format:
SCHEMA = {"type": "object",
          "properties": {"sentiment": {"enum": ["positive", "negative", "neutral"]}},
          "required": ["sentiment"]}

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Classify: SGLang is fast!"}],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "cls", "schema": SCHEMA}},
    # SGLang also accepts extra_body={"regex": "..."} or {"ebnf": "<grammar>"}.
)
print(resp.choices[0].message.content)
```

---

## 8. Serving GPU Sizing

Inference is **far cheaper than training** — no gradients/optimizer states:

```
inference_vram  ≈  weights  +  KV-cache pool (mem-fraction-static)  +  activations  +  overhead
```

- **Weights:** ~2 GB / 1B at FP16; **~1 GB / 1B at FP8**; ~0.5 GB / 1B at INT4.
- **KV-cache pool:** SGLang reserves `mem_fraction_static × VRAM` for it. **More pool → deeper RadixAttention cache → higher hit rate**, but too high starves decode activations.
- **Overhead:** ~1–2 GB CUDA context.

**Grounded sizing examples (serving):**

| Model | Precision | GPU | Approx VRAM | Flags |
|---|---|---|---|---|
| 7B | FP16 | 1× RTX 4090 (24 GB) | ~14 GB + cache | `--tp 1` |
| 13B | FP16 | 1× A100-40 | ~26 GB + cache | `--tp 1` |
| 70B | FP8 | 1× H100-80 | ~72 GB | `--tp 1 --quantization fp8` |
| 70B | FP16 | 2× H100-80 | ~140 GB | `--tp 2` |
| 405B | FP8 | 8× H100-80 | ~405 GB | `--tp 8 --quantization fp8` |

> **The SGLang-specific lever:** `--mem-fraction-static` trades **KV-cache depth** (RadixAttention hit rate) against **decode headroom**. Start at **0.9–0.92** on a single-model 80 GB H100; lower it if you see decode-time OOM, raise it if your prefix-cache hit rate is below target and you have headroom.

---

## 9. How to Test a Serving Deployment

| Layer | Proves | GPU? | When |
|---|---|---|---|
| **1 Matrix enforcement** | Illegal configs rejected (AC-1/2/3) | No | Every commit |
| **2 GPU-fit preflight** | weights + KV pool + overhead fit (AC-4) | No | Every commit + pre-launch |
| **3 Health/smoke** | server boots, health OK, one completion returns | Yes | Pre-launch |
| **4 Structured-output contract** | constrained request validates against schema (AC-5) | Yes | Pre-launch |
| **5 Load + cache-hit / SLO** | TTFT/TPS/p99/error AND prefix-cache hit-rate meet targets (AC-6) | Yes | Pre-promotion |
| **6 Spec coverage** | every FR/AC has a test (Q-2) | No | Every commit |

```python
def test_ac1_tp_exceeds_node(monkeypatch):        # AC-1 / FR-004
    import pytest; from pydantic import ValidationError
    from serving.config import ServingConfig
    with pytest.raises(ValidationError):
        ServingConfig.model_validate({"model_path":"x","mem_fraction_static":0.9,
            "gpu":{"gpu_type":"H100-80","num_gpus":2,"num_nodes":1,"tp":4}})

def test_slo_and_cache_gate(load_result):          # AC-6 / R-1 + R-3
    assert load_result.ttft_p95_ms < 400
    assert load_result.output_tps >= 2000
    assert load_result.p99_latency_s < 7
    assert load_result.prefix_cache_hit_rate >= 0.75   # the reason you chose SGLang
```

> The **prefix-cache hit-rate** assertion is what makes SGLang's value measurable. Drive Layer 5 with a workload that *mimics your real prefix structure* (shared system prompt + multi-turn history) and scrape the metrics endpoint for the cache-hit stat — a synthetic all-unique-prompts load test will under-report SGLang's benefit.

---

## 10. Files: Necessary vs. Not

### ✅ Necessary
| File | Why |
|---|---|
| `specs/spec.md` | Serving Matrix + SLOs + **prefix-cache target**. |
| `specs/constitution.md` | Reliability/security/repro/cost rules. |
| `src/serving/config.py` | Enforces the matrix before launch. |
| `src/serving/gpu_preflight.py` | weights + KV pool fit (mem-fraction aware). |
| `configs/*.yaml` | One deployment = one config (all server args, in VC). |
| `tests/test_matrix.py`, `test_preflight.py`, `test_load_slo.py`, `test_spec_coverage.py` | Prove the spec + SLOs + cache target. |
| Pinned SGLang version + **image digest** | Reproducibility (RP-1). |

### ⚠️ Optional
| File | When |
|---|---|
| Model Gateway/router config | Data parallelism / load balancing across replicas. |
| `Dockerfile` / k8s + `/dev/shm` sizing | Containerized prod (SGLang needs shared memory). |
| Grafana dashboards | Production observability. |

### ❌ Not necessary (anti-patterns)
| Anti-pattern | Why avoid |
|---|---|
| `launch_server` with hand-typed flags | Unreproducible; bypasses matrix + preflight. |
| `--mem-fraction-static` left at default and never tuned | You're leaving RadixAttention's win on the table — or OOMing at decode. |
| Choosing SGLang but testing with all-unique prompts | Hides the prefix-cache benefit; you'll wrongly conclude it's no better. |
| Free-text JSON parsing | Use constrained decoding (Q-1). |
| `--log-requests` in prod with PII | Logs user content (S-2). |

---

## 11. Pros & Cons

**Pros:** open-source, production-proven at massive scale (400k+ GPUs); **RadixAttention** = automatic prefix-cache reuse, a decisive win for agents/multi-turn/RAG (often faster than vLLM on shared-prefix workloads); OpenAI-compatible; strong structured output (compressed FSM), speculative decoding, multimodal, broad hardware support (NVIDIA/AMD/Intel/TPU); a frontend language for complex generation programs; deterministic-inference mode.

**Cons:** **you run the infra**; higher configuration surface than some engines (more knobs = more power but more to get wrong); `--mem-fraction-static` tuning matters; multi-node/router setup is non-trivial; fast-moving project (pin versions). *If you don't want GPU ops, use a managed platform (Fireworks).*

**vs vLLM:** both are top-tier OpenAI-compatible open-source engines. **Pick SGLang when shared-prefix reuse dominates** (RadixAttention is the differentiator); **pick vLLM** for the broadest ecosystem/adoption. The honest answer is to **benchmark both on your real traffic** — see the vLLM companion guide.

---

## 12. References
- **SGLang Documentation**: <https://docs.sglang.io/>
- **SGLang — Server Arguments** (`launch_server`, `--tp`, `--mem-fraction-static`, quant, structured output): <https://docs.sglang.io/docs/advanced_features/server_arguments>
- **SGLang GitHub** (RadixAttention, LMSYS): <https://github.com/sgl-project/sglang>
- **RadixAttention / SGLang paper**: <https://arxiv.org/abs/2312.07104>
- **Inference GPU sizing** (weights + KV cache): <https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm>
- **GitHub Spec Kit** (the SDD methodology applied here): <https://github.com/github/spec-kit>

---
*Illustrative artifact. SGLang evolves quickly (v0.5.x as of early 2026) — confirm server-arg names and structured-output options against your installed version. SLO and cache-hit numbers are placeholders; set them from your workload and validate with a prefix-representative load test before promoting.*
