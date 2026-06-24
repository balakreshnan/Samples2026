# Spec-Driven Development for LLM Serving with vLLM

*A spec-first methodology for deploying and operating LLM inference with **vLLM** — the open-source, high-throughput serving engine (PagedAttention + continuous batching, OpenAI-compatible API). Same SDD method as the companion guides — **Constitution → Spec → Plan → Tasks → Implement → Test** — applied to **serving** instead of training.*

> **Companion to:** the Spec-Driven Development, Hugging Face fine-tuning, and NVIDIA fine-tuning guides. Those produce a model; **this one serves it.**
>
> **Who this is for:** platform/MLOps engineers self-hosting open-weight models on their own GPUs who want reproducible, governed, SLO-backed deployments — not a `vllm serve` command pasted into a tmux session that nobody can reproduce.
>
> **What you'll be able to do:** write a serving spec that *enforces* model/parallelism/quantization/structured-output/GPU rules before the server launches, hit declared latency/throughput SLOs, and prove the endpoint behaves with pre-launch tests.

---

## Table of Contents
1. [What vLLM Is & When to Use It](#1-what-vllm-is--when-to-use-it)
2. [The SDD Method Applied to Serving](#2-the-sdd-method-applied-to-serving)
3. [Step 1 — The Constitution](#3-step-1--the-constitution)
4. [Step 2 — The Spec (with the enforceable Serving Matrix + SLOs)](#4-step-2--the-spec)
5. [Step 3 — The Plan](#5-step-3--the-plan)
6. [Step 4 — Tasks](#6-step-4--tasks)
7. [Step 5 — Implement](#7-step-5--implement)
8. [Serving GPU Sizing (Inference VRAM)](#8-serving-gpu-sizing-inference-vram)
9. [How to Test a Serving Deployment](#9-how-to-test-a-serving-deployment)
10. [Files: Necessary vs. Not](#10-files-necessary-vs-not)
11. [Pros & Cons](#11-pros--cons)
12. [References](#12-references)

---

## 1. What vLLM Is & When to Use It

**vLLM** is a fast, memory-efficient inference and serving engine for LLMs (Apache-2.0, originally from UC Berkeley's Sky Computing Lab). Its core innovations:

- **PagedAttention** — manages KV-cache memory like virtual memory pages, eliminating fragmentation and enabling far higher concurrency.
- **Continuous batching** — new requests join an in-flight batch instead of waiting, keeping the GPU saturated.
- **OpenAI-compatible HTTP server** — `vllm serve` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, and more, so existing OpenAI-SDK code works by changing the `base_url`.
- **Tensor parallelism** for multi-GPU, **quantization** (GPTQ/AWQ/FP8, FP8 KV cache), **multi-LoRA serving**, **speculative decoding**, **prefix caching**, and **guided/structured decoding**.

**Use vLLM when** you want to **self-host** an open-weight model on your own GPUs with maximum throughput and full control, and you can run the infrastructure. **Don't** use it when you'd rather pay per token and skip GPU ops (→ a managed platform like Fireworks).

---

## 2. The SDD Method Applied to Serving

Serving has its own version of "what's done": **Service Level Objectives (SLOs)** — time-to-first-token (TTFT), inter-token latency, throughput (tokens/sec), and p99 request latency at a target concurrency. The spec makes those explicit and the tests enforce them.

```
SPECIFY ─────────▶ PLAN ──────────▶ TASKS ─────────▶ IMPLEMENT ───────▶ (PRE-LAUNCH GATES)
model, SLOs,       GPU + parallel-  config → launch   vllm serve from    config valid?
quantization,      ism + quant +    → client → load   a validated config  GPU-fit preflight?
structured-out,    KV/prefix cache  → SLO test        (no ad-hoc flags)   health + SLO met?
GPU plan, the      plan                                                    structured-out ok?
serving matrix
```

---

## 3. Step 1 — The Constitution

`specs/constitution.md` — binding rules for every vLLM deployment.

```markdown
# Project Constitution — vLLM Serving

## Reliability & SLOs
- R-1: Every deployment declares SLOs: TTFT p95, inter-token latency, throughput,
       and p99 end-to-end latency AT a target concurrency. A deploy that misses them
       in load test is not promoted.
- R-2: /health and /metrics MUST be wired to readiness probes + monitoring before traffic.
- R-3: max_model_len and max_num_seqs are declared, not defaulted — they bound memory.

## Security
- S-1: The server MUST require an API key (--api-key) sourced from an env var; never hard-coded.
- S-2: The endpoint is not exposed publicly without an auth proxy / gateway in front.
- S-3: Request IDs enabled (--enable-request-id-headers) for traceability.

## Reproducibility
- RP-1: Pin the vLLM version + Docker image digest. Record model revision.
- RP-2: The full launch config (every engine arg) lives in version control; no ad-hoc CLI flags.
- RP-3: If determinism is required, fix seed + sampling and document that throughput drops.

## Cost / Hardware
- H-1: Declare gpu_type, num_gpus, tensor_parallel_size. TP MUST stay within a node.
- H-2: A GPU-fit preflight (weights + KV cache + overhead <= usable VRAM) MUST pass before launch.
- H-3: gpu_memory_utilization declared (default 0.9); leave headroom for spikes.

## Quality
- Q-1: If the app needs structured output, the spec declares the schema and the server
       enforces it (guided decoding). "Hope the model returns JSON" is forbidden.
- Q-2: Every requirement (FR/AC) maps to >= 1 automated test.
```

---

## 4. Step 2 — The Spec

`specs/spec.md` — the requirements, the **enforceable Serving Matrix**, and the **SLOs** that define "done."

```markdown
# Spec: vLLM Serving Deployment

## Functional Requirements
- FR-001: The deployment is fully determined by one config (model, engine args, GPU plan,
          structured-output, SLOs).
- FR-002: The server exposes an OpenAI-compatible API and requires an API key from an env var.
- FR-003: The config validates the Serving Matrix and FAILS before launch on any violation.
- FR-004: tensor_parallel_size MUST divide num_gpus AND be <= gpus_per_node. (H-1)
- FR-005: A GPU-fit preflight MUST pass: est_weights + est_kv_cache + overhead <= usable VRAM. (H-2)
- FR-006: FP8 weight/KV-cache quantization REQUIRES Hopper/Blackwell GPUs.
- FR-007: Multi-LoRA serving REQUIRES --enable-lora and a declared list of adapters.
- FR-008: If structured output is required, requests MUST use a guided-decoding constraint
          (guided_json / guided_choice / guided_regex / guided_grammar).
- FR-009: max_model_len and max_num_seqs are explicitly set (bound KV-cache memory).
- FR-010: /health and /metrics are reachable; readiness probe gates traffic.

## ── SERVING MATRIX (enforced by config.py) ──
| Field                  | Rule |
|------------------------|------|
| tensor_parallel_size   | divides num_gpus; <= gpus_per_node (FR-004) |
| quantization           | one of {none, awq, gptq, fp8}; fp8 → Hopper/Blackwell (FR-006) |
| kv_cache_dtype         | {auto, fp8_e4m3, fp8_e5m2}; fp8 → Hopper/Blackwell |
| enable_lora            | REQUIRED true if any lora_modules declared (FR-007) |
| max_model_len          | REQUIRED (> 0) (FR-009) |
| max_num_seqs           | REQUIRED (> 0) |
| gpu_memory_utilization | 0 < x <= 0.97 |
| structured_output      | if required → a guided constraint is set on requests (FR-008) |

## Service Level Objectives (definition of done)
- SLO-TTFT: p95 time-to-first-token < 500 ms at concurrency=32.
- SLO-TPS:  aggregate >= 1,500 output tokens/sec at concurrency=32.
- SLO-P99:  p99 end-to-end latency < 8 s for a 256-token completion.
- SLO-ERR:  error rate < 0.1% over the load window.
(Numbers are examples — set yours per workload + hardware.)

## Acceptance Criteria
- AC-1: tensor_parallel_size=4 with num_gpus=2 → rejected. (FR-004)
- AC-2: quantization=fp8 on an A100 → rejected. (FR-006)
- AC-3: lora_modules set but enable_lora=false → rejected. (FR-007)
- AC-4: GPU-fit preflight where est_vram > usable → rejected. (FR-005)
- AC-5: a structured-output request returns JSON validating against the declared schema. (FR-008)
- AC-6: load test missing any SLO → deployment NOT promoted. (R-1)
```

---

## 5. Step 3 — The Plan

`specs/plan.md`:

```markdown
# Plan: vLLM Serving
## Engine
- vLLM (pin version + image digest). OpenAI-compatible server via `vllm serve`.
## GPU & parallelism (feeds the matrix + preflight)
- Single GPU if weights + KV cache fit; else tensor_parallel_size = N within ONE node.
- Quantize (AWQ/GPTQ/FP8) to fit bigger models or raise concurrency. FP8 → H100/H200/B200.
## Memory
- max_model_len bounds per-request KV; max_num_seqs bounds concurrency; gpu_memory_utilization
  reserves headroom. Enable prefix caching for shared system prompts.
## Structured output
- Pick a guided-decoding backend (xgrammar / outlines / lm-format-enforcer) and the constraint type.
## Ops
- /health → readiness probe; /metrics → Prometheus; --api-key from env; request-id headers on.
## Constitution check: R-* (SLOs/probes), S-* (auth), RP-* (pinning/config-in-VC),
##   H-* (GPU preflight), Q-* (structured output + tests).
```

---

## 6. Step 4 — Tasks

`specs/tasks.md`:

```markdown
| ID | Task                                                   | Satisfies        | Test |
|----|--------------------------------------------------------|------------------|------|
| T1 | ServingConfig schema enforcing the Serving Matrix      | FR-003, matrix   | test_matrix.py |
| T2 | gpu_preflight: weights + KV-cache + overhead estimate  | FR-005, H-2      | test_preflight.py |
| T3 | Render `vllm serve` command/Docker from the config     | FR-001/002/009   | test_render.py |
| T4 | OpenAI client wrapper + guided-decoding helper         | FR-008           | test_structured.py |
| T5 | Health/smoke check + load test harness (SLO gate)      | R-1/2, SLO-*     | test_load_slo.py |
| T6 | Map every FR/AC to a test                              | Q-2              | test_spec_coverage.py |
```

---

## 7. Step 5 — Implement

### 7.1 Config enforcement (`src/serving/config.py`)

Rejects matrix-violating configs **before** the GPU is touched.

```python
"""ServingConfig enforces the vLLM Serving Matrix (FR-003)."""
from enum import Enum
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, model_validator

FP8_GPUS = {"H100-80", "H200-141", "B200-192"}
PHYS_VRAM = {"L4":24,"L40S":48,"A100-40":40,"A100-80":80,"H100-80":80,"H200-141":141,"B200-192":192}

class Quant(str, Enum):
    none="none"; awq="awq"; gptq="gptq"; fp8="fp8"

class GPUPlan(BaseModel):
    gpu_type: str
    num_gpus: int = Field(1, gt=0)
    num_nodes: int = Field(1, gt=0)
    tensor_parallel_size: int = Field(1, gt=0)
    gpu_memory_utilization: float = Field(0.9, gt=0, le=0.97)

class ServingConfig(BaseModel):
    model: str
    model_revision: str = "main"
    api_key_env: str = "VLLM_API_KEY"          # name of the env var, not the value (S-1)
    quantization: Quant = Quant.none
    kv_cache_dtype: str = "auto"               # auto | fp8_e4m3 | fp8_e5m2
    max_model_len: int = Field(..., gt=0)      # REQUIRED (FR-009)
    max_num_seqs: int = Field(..., gt=0)
    enable_prefix_caching: bool = True
    enable_lora: bool = False
    lora_modules: List[str] = []
    structured_output_required: bool = False
    gpu: GPUPlan

    @model_validator(mode="after")
    def _enforce(self):
        g = self.gpu
        gpus_per_node = g.num_gpus // g.num_nodes
        if g.num_gpus % g.tensor_parallel_size != 0:
            raise ValueError("tensor_parallel_size must divide num_gpus (FR-004)")
        if g.tensor_parallel_size > gpus_per_node:
            raise ValueError("tensor_parallel_size must stay within a node (AC-1/FR-004)")
        if self.quantization is Quant.fp8 and g.gpu_type not in FP8_GPUS:
            raise ValueError("fp8 quantization requires Hopper/Blackwell (AC-2/FR-006)")
        if self.kv_cache_dtype.startswith("fp8") and g.gpu_type not in FP8_GPUS:
            raise ValueError("fp8 KV cache requires Hopper/Blackwell (FR-006)")
        if self.lora_modules and not self.enable_lora:
            raise ValueError("lora_modules set but enable_lora=false (AC-3/FR-007)")
        return self

def load_config(path: str) -> ServingConfig:
    with open(path, encoding="utf-8") as fh:
        return ServingConfig.model_validate(yaml.safe_load(fh))
```

`configs/llama8b.yaml`:
```yaml
model: meta-llama/Llama-3.1-8B-Instruct
api_key_env: VLLM_API_KEY
quantization: none
max_model_len: 8192
max_num_seqs: 64
enable_prefix_caching: true
gpu: { gpu_type: H100-80, num_gpus: 1, tensor_parallel_size: 1, gpu_memory_utilization: 0.9 }
structured_output_required: true
```

### 7.2 Render & launch (`src/serving/render.py`)

```python
"""Turn a validated ServingConfig into a reproducible `vllm serve` command (FR-001)."""
import os
from .config import ServingConfig

def render_vllm_command(cfg: ServingConfig) -> list[str]:
    cmd = [
        "vllm", "serve", cfg.model,
        "--revision", cfg.model_revision,
        "--dtype", "auto",
        "--max-model-len", str(cfg.max_model_len),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--tensor-parallel-size", str(cfg.gpu.tensor_parallel_size),
        "--gpu-memory-utilization", str(cfg.gpu.gpu_memory_utilization),
        "--api-key", os.environ[cfg.api_key_env],      # value from env (S-1)
        "--enable-request-id-headers",                 # S-3
    ]
    if cfg.quantization.value != "none":
        cmd += ["--quantization", cfg.quantization.value]
    if cfg.kv_cache_dtype != "auto":
        cmd += ["--kv-cache-dtype", cfg.kv_cache_dtype]
    if cfg.enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]
    if cfg.enable_lora:
        cmd += ["--enable-lora"] + sum([["--lora-modules", m] for m in cfg.lora_modules], [])
    return cmd
```

```bash
# Equivalent one-liner the render produces:
VLLM_API_KEY=$(openssl rand -hex 16) vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 8192 --max-num-seqs 64 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 --enable-prefix-caching --api-key "$VLLM_API_KEY" \
  --enable-request-id-headers
# Server: http://localhost:8000/v1   ·   health: /health   ·   metrics: /metrics
```

### 7.3 Client + structured output (`src/serving/client.py`)

vLLM accepts guided-decoding constraints via the OpenAI client's `extra_body` (FR-008):

```python
from openai import OpenAI
import os

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["VLLM_API_KEY"])

# Guided JSON (output is forced to match the schema):
SCHEMA = {"type": "object",
          "properties": {"sentiment": {"enum": ["positive", "negative", "neutral"]}},
          "required": ["sentiment"]}

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Classify: vLLM is wonderful!"}],
    extra_body={"guided_json": SCHEMA},          # also: guided_choice / guided_regex / guided_grammar
)
print(resp.choices[0].message.content)
```

---

## 8. Serving GPU Sizing (Inference VRAM)

Inference is **far cheaper than training** — there are **no gradients or optimizer states.** The budget is:

```
inference_vram  ≈  weights  +  KV cache  +  activations  +  CUDA overhead
```

- **Weights:** ~2 GB / 1B params at FP16/BF16; **~1 GB / 1B at FP8/INT8**; ~0.5 GB / 1B at INT4.
- **KV cache** (the variable that bites): scales with **concurrent sequences × context length** — bounded by your `max_num_seqs × max_model_len`. This is why both are required fields.
- **Overhead:** ~1–2 GB CUDA context.

**Grounded sizing examples (serving):**

| Model | Precision | GPU | Approx VRAM | Flag |
|---|---|---|---|---|
| 7–8B | FP16 | 1× RTX 4090 / L4 (24 GB) | ~14 GB + KV | `--tensor-parallel-size 1` |
| 13B | FP16 | 1× A100-40 | ~26 GB + KV | `--tensor-parallel-size 1` |
| 70B | FP8 | 1× H100-80 | ~72 GB | `--quantization fp8` |
| 70B | FP16 | 2× H100-80 | ~140 GB | `--tensor-parallel-size 2` |
| 405B | FP8 | 8× H100-80 | ~405 GB | `--tensor-parallel-size 8 --quantization fp8` |

> **Preflight rule (T2):** `weights + KV_cache(max_num_seqs, max_model_len) + overhead ≤ gpu_memory_utilization × physical_VRAM × num_gpus`. If it doesn't fit, the config is rejected — before the server OOMs at peak concurrency.

---

## 9. How to Test a Serving Deployment

Most serving failures are catchable **before** traffic — and the rest with a load test against the SLOs.

| Layer | Proves | GPU? | When |
|---|---|---|---|
| **1 Matrix enforcement** | Illegal configs rejected (AC-1/2/3) | No | Every commit |
| **2 GPU-fit preflight** | weights + KV + overhead fit (AC-4) | No | Every commit + pre-launch |
| **3 Health/smoke** | server boots, `/health` 200, one completion returns | Yes | Pre-launch |
| **4 Structured-output contract** | guided request validates against schema (AC-5) | Yes | Pre-launch |
| **5 Load / SLO** | TTFT, TPS, p99, error-rate meet SLOs at target concurrency (AC-6) | Yes | Pre-promotion |
| **6 Spec coverage** | every FR/AC has a test (Q-2) | No | Every commit |

```python
# Layer 1/2 — cheap, no GPU, runs on every commit
import pytest
from pydantic import ValidationError
from serving.config import ServingConfig

def test_ac1_tp_exceeds_gpus_rejected():           # AC-1 / FR-004
    with pytest.raises(ValidationError):
        ServingConfig.model_validate({"model":"x","max_model_len":4096,"max_num_seqs":16,
            "gpu":{"gpu_type":"H100-80","num_gpus":2,"tensor_parallel_size":4}})

def test_ac2_fp8_on_a100_rejected():               # AC-2 / FR-006
    with pytest.raises(ValidationError):
        ServingConfig.model_validate({"model":"x","quantization":"fp8","max_model_len":4096,
            "max_num_seqs":16,"gpu":{"gpu_type":"A100-80","num_gpus":1,"tensor_parallel_size":1}})
```

```python
# Layer 5 — the SLO gate (load test). Promotion is blocked if any SLO misses.
def test_slo_gate(load_result):                    # AC-6 / R-1
    assert load_result.ttft_p95_ms < 500
    assert load_result.output_tps >= 1500
    assert load_result.p99_latency_s < 8
    assert load_result.error_rate < 0.001
```

> Use vLLM's own `vllm bench serve` (or a tool like `genai-perf`/`locust`) to drive Layer 5, and scrape `/metrics` (Prometheus) for TTFT/throughput during the run.

---

## 10. Files: Necessary vs. Not

### ✅ Necessary
| File | Why |
|---|---|
| `specs/spec.md` | Serving Matrix + SLOs — the source of truth. |
| `specs/constitution.md` | Reliability/security/repro/cost rules. |
| `src/serving/config.py` | Enforces the matrix before launch. |
| `src/serving/gpu_preflight.py` | weights + KV-cache fit check (no peak-OOM surprises). |
| `configs/*.yaml` | One deployment = one config (every engine arg, in VC). |
| `tests/test_matrix.py`, `test_preflight.py`, `test_load_slo.py`, `test_spec_coverage.py` | Prove the spec + SLOs hold. |
| Pinned vLLM version + **image digest** | Reproducibility (RP-1). |
| `.env.example`, `.gitignore` | API key via env, never committed. |

### ⚠️ Optional
| File | When |
|---|---|
| `Dockerfile` / k8s Deployment + HPA | Containerized / autoscaled prod. |
| Prometheus/Grafana dashboards | Production observability. |
| LoRA adapter manifest | Multi-LoRA serving. |

### ❌ Not necessary (anti-patterns)
| Anti-pattern | Why avoid |
|---|---|
| `vllm serve` with hand-typed flags in a terminal | Unreproducible; bypasses the matrix + preflight. |
| API key hard-coded or omitted | Security breach (S-1). |
| Unset `max_model_len`/`max_num_seqs` | KV cache grows unbounded → OOM under load. |
| "We'll parse the JSON the model returns" | Use guided decoding; free-text parsing is flaky (Q-1). |
| Promoting without a load/SLO test | You're shipping latency you've never measured. |

---

## 11. Pros & Cons

**Pros:** open-source (no per-token cost, full control); **best-in-class throughput** (PagedAttention + continuous batching); OpenAI-compatible (drop-in `base_url`); broad model + quantization support; multi-LoRA, speculative decoding, prefix caching, guided decoding; huge community and rapid model day-0 support.

**Cons:** **you run the infra** (GPUs, scaling, on-call); KV-cache/memory tuning has a learning curve; fast-moving project (pin versions); single-engine (no managed batch/fine-tuning platform around it); multi-node setups are non-trivial. *If you don't want GPU ops, a managed platform (Fireworks) trades control for convenience.*

**vs SGLang:** both are top open-source engines with OpenAI-compatible APIs. SGLang's **RadixAttention** gives automatic prefix-cache reuse that shines on agentic/multi-turn/RAG workloads; vLLM is the most widely adopted with the broadest ecosystem. Benchmark both on *your* workload — see the SGLang companion guide.

---

## 12. References
- **vLLM — OpenAI-Compatible Server**: <https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/>
- **vLLM — Engine Args / Quantization / LoRA / Structured Outputs**: <https://docs.vllm.ai/>
- **vLLM GitHub** (PagedAttention, continuous batching): <https://github.com/vllm-project/vllm>
- **vLLM paper** (SOSP 2023): <https://arxiv.org/abs/2309.06180>
- **Inference GPU sizing** (weights + KV cache): <https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm>
- **GitHub Spec Kit** (the SDD methodology applied here): <https://github.com/github/spec-kit>

---
*Illustrative artifact. vLLM flags evolve quickly — confirm engine-arg names and guided-decoding options against your installed version. SLO numbers are placeholders; set them from your workload + hardware and validate with a load test before promoting.*
