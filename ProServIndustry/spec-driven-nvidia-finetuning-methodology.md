# Spec-Driven Methodology for Fine-Tuning on NVIDIA Frameworks

*A spec-first methodology for post-training LLMs with NVIDIA's latest stack — **NeMo Framework 2.0** (SFT + PEFT) and **NeMo-RL** (DPO, GRPO/GSPO, RLVR, reward modeling, distillation) — plus practical guidance on **how to choose NVIDIA GPUs** and **how to calculate how many GPUs you need** for any model size and method.*

> **Companion to:** the *Spec-Driven Development (SDD)* and *Spec-Driven Fine-Tuning (Hugging Face)* tutorials. Same method — **Constitution → Spec → Plan → Tasks → Implement → Test** — now targeting NVIDIA's training frameworks and the hardware-sizing decisions they force.
>
> **Who this is for:** ML engineers, MLOps, and architects standing up enterprise fine-tuning on NVIDIA GPUs (DGX/HGX, cloud H100/H200/B200, or workstations) who want reproducible, governed, hardware-right-sized runs instead of ad-hoc notebooks.
>
> **What you'll be able to do:** pick the right NVIDIA framework + backend for a method; write a spec that *enforces* method/scope/backend/GPU rules before a job launches; size GPUs with formulas (not guesses); and run SFT **and** RL from one governed project.

---

## Table of Contents

1. [The NVIDIA Fine-Tuning Landscape (Latest)](#1-the-nvidia-fine-tuning-landscape-latest)
2. [Methods Available — and Which Framework Runs Them](#2-methods-available--and-which-framework-runs-them)
3. [The SDD Method Applied to NVIDIA Fine-Tuning](#3-the-sdd-method-applied-to-nvidia-fine-tuning)
4. [Step 1 — The Constitution](#4-step-1--the-constitution)
5. [Step 2 — The Spec (with enforceable matrices)](#5-step-2--the-spec-with-enforceable-matrices)
6. [Step 3 — The Plan](#6-step-3--the-plan)
7. [Step 4 — Tasks](#7-step-4--tasks)
8. [Step 5 — Implement](#8-step-5--implement)
   - 8.1 [Config enforcement](#81-config-enforcement-pydantic)
   - 8.2 [NeMo 2.0 SFT / PEFT](#82-nemo-20--sft--peft-lora--dora--qlora)
   - 8.3 [NeMo-RL DPO / GRPO / RLVR](#83-nemo-rl--dpo--grpo--rlvr)
9. [GPU Selection Guidance](#9-gpu-selection-guidance)
10. [How to Calculate the GPUs You Need](#10-how-to-calculate-the-gpus-you-need)
    - 10.1 [The VRAM formula](#101-the-vram-formula-where-every-gigabyte-goes)
    - 10.2 [Reference VRAM table](#102-reference-vram-table-by-model-size--method)
    - 10.3 [From VRAM to GPU count](#103-from-vram-to-gpu-count)
    - 10.4 [The `gpu_planner.py` calculator](#104-the-gpu_plannerpy-calculator)
    - 10.5 [Extra memory for RL](#105-extra-memory-for-rl)
    - 10.6 [Parallelism cheat-sheet](#106-parallelism-cheat-sheet-tp-pp-cp-sp-ep-fsdp)
11. [How to Test (Pre-Launch Gates)](#11-how-to-test-pre-launch-gates)
12. [Files: Necessary vs. Not](#12-files-necessary-vs-not)
13. [Pros & Cons](#13-pros--cons)
14. [Pitfalls & Best Practices](#14-pitfalls--best-practices)
15. [A 1-Week Skilling Plan](#15-a-1-week-skilling-plan)
16. [References](#16-references)

---

## 1. The NVIDIA Fine-Tuning Landscape (Latest)

NVIDIA's training stack reorganized in 2025–2026 around the **NeMo Framework** umbrella, which is splitting into focused repositories. The pieces that matter for LLM fine-tuning:

| Component | What it is | Use it for |
|---|---|---|
| **NeMo Framework 2.0** | The modular generative-AI training framework (Python-first, NeMo-Run recipes). Built on **Megatron Core**. | SFT and **PEFT** (LoRA, DoRA, QLoRA, P-tuning, IA3) on LLMs/multimodal. |
| **NeMo-RL** | Open-source **post-training / reinforcement-learning** library (Ray-based, scales 1 GPU → thousands). | **DPO, GRPO/GSPO, DAPO, GDPO, reward modeling (RM), RLVR, on-policy distillation, SFT.** |
| **Megatron Core / Megatron-Bridge** | High-performance kernels + 6-D parallelism (TP/PP/CP/SP/EP/FSDP). | Scaling large models (30B–405B+) across many GPUs/nodes. |
| **NeMo AutoModel** | **Day-0 Hugging Face model support** inside NeMo (DTensor path). | Fine-tuning HF checkpoints without waiting for native NeMo ports. |
| **NeMo Curator** | Large-scale data curation/cleaning. | Building/filtering training & preference datasets. |
| **NeMo Customizer** (NeMo **Microservices**) | Managed, API-driven fine-tuning microservice (enterprise/NVIDIA AI Enterprise). | Production LoRA fine-tuning behind an API, less infra to run yourself. |
| **TensorRT-LLM / NIM** | Inference optimization & serving (not training). | Deploying the tuned model. |

**Two backends you will choose between (both in NeMo-RL and NeMo 2.0):**
- **DTensor** — PyTorch-native distributed training (FSDP2, TP, SP, PP, CP). Hackable, great for prototyping and small/medium models.
- **Megatron Core** — NVIDIA's high-performance path with full 6-D parallelism (TP/PP/CP/SP/**EP** for MoE/FSDP). Best for large models and long context.

**Generation/rollout backends (RL only):** **vLLM**, **SGLang**, or **Megatron inference** — the engine that samples completions during RL. Choosing one is part of the spec.

**Decision in one line:**
- Pure **SFT/PEFT**? → **NeMo Framework 2.0** recipes.
- Any **RL / preference / RLVR / distillation**? → **NeMo-RL**.
- Want a **managed API**, not infra? → **NeMo Customizer** microservice.
- Just want an **HF checkpoint** fine-tuned fast? → **NeMo AutoModel** (DTensor).

---

## 2. Methods Available — and Which Framework Runs Them

Fine-tuning has **two axes** (carried over from the prior guides) plus a **framework/backend axis** unique to NVIDIA:

- **Parameter scope:** `full | lora | dora | qlora | ptuning | ia3`
- **Training objective:** `sft | dpo | grpo | gspo | dapo | rm | rlvr | distill`
- **Backend:** `dtensor | megatron` × generation `vllm | sglang | megatron`

| Objective | What it does | NVIDIA framework | Notes |
|---|---|---|---|
| **SFT** | Supervised next-token on (prompt→response) | NeMo 2.0 **or** NeMo-RL | Foundation step; do this first. |
| **LoRA / DoRA / QLoRA / P-tuning / IA3** | PEFT scopes (freeze base, train adapters) | NeMo 2.0 (`peft_scheme`) + NeMo-RL (LoRA SFT/GRPO/DPO) | QLoRA = 4-bit base + LoRA. |
| **DPO** | Direct Preference Optimization (pairs) | **NeMo-RL** | No reward model; needs `chosen`/`rejected`. |
| **RM (Reward Modeling)** | Train a reward model from preferences | **NeMo-RL** | Feeds classic RLHF/PPO-style loops. |
| **GRPO / GSPO** | Group-relative policy optimization (critic-free RL) | **NeMo-RL** | The popular reasoning-RL algorithm. |
| **DAPO / GDPO** | GRPO variants (dynamic sampling; multi-reward) | **NeMo-RL** | More stable / multi-reward RL. |
| **RLVR** | RL from **Verifiable Rewards** (programmatic verifiers) | **NeMo-RL** *environments* | GRPO + deterministic verifier; great for math/code. |
| **On-policy distillation** | Student matches a larger teacher's logits via KL | **NeMo-RL** | Near-teacher quality, cheaper than RL. |

> **RLVR on NVIDIA = GRPO + a NeMo-RL "environment."** Instead of a learned reward model, you register an **environment** that runs a deterministic verifier (math checker, code/unit-test runner, format validator) and returns the reward. This is exactly the RLVR pattern (DeepSeek-R1-style): sample K completions → verify → reward `1.0/0.0` → GRPO update.

---

## 3. The SDD Method Applied to NVIDIA Fine-Tuning

```
SPECIFY ───────▶ PLAN ──────────▶ TASKS ─────────▶ IMPLEMENT ───────▶ (PRE-LAUNCH GATES)
model, data,     framework +      data prep →      NeMo 2.0 recipe    config valid?
method, metric,  backend +        train → eval →   OR NeMo-RL YAML    GPU-fit preflight?
GPU budget,      parallelism +    merge → register  (config-driven)   parallelism legal?
the matrices     GPU plan                                              eval gate passed?
```

Every artifact is the same as before; the *content* is NVIDIA-specific and, critically, includes a **GPU plan** the spec enforces (a job that won't fit the requested GPUs is rejected *before* it queues — saving real cluster time and money).

---

## 4. Step 1 — The Constitution

`specs/constitution.md` — binding rules for every NVIDIA fine-tuning run.

```markdown
# Project Constitution — NVIDIA LLM Fine-Tuning

## Reproducibility & Environment
- R-1: Pin the NGC container tag (e.g., nvcr.io/nvidia/nemo:<tag>) and all library versions.
- R-2: Set a global seed; record git commit + container tag + recipe/config in every output.
- R-3: Reference base model + dataset by immutable revision. Convert HF→NeMo with import_ckpt
       and record the checksum.

## Method Integrity (enforced by config.py)
- M-1: A run declares exactly one scope (full|lora|dora|qlora|ptuning|ia3) and one
       objective (sft|dpo|grpo|gspo|dapo|rm|rlvr|distill).
- M-2: Each scope/objective's REQUIRED and FORBIDDEN fields are enforced (see spec.md matrices).
- M-3: RLVR MUST use a deterministic verifier environment and MUST NOT use a learned reward model.

## Hardware & Cost (the NVIDIA-specific core)
- H-1: Every run declares a GPU plan: gpu_type, num_gpus, num_nodes, and the parallelism
       (TP/PP/CP/SP/EP/DP). 
- H-2: A run MUST pass a GPU-fit preflight: estimated VRAM (incl. RL overhead) <= usable VRAM
       of the requested GPUs. A run that cannot fit MUST be rejected before launch.
- H-3: Tensor Parallel (TP) MUST stay within a node (NVLink/NVSwitch); Pipeline/Data Parallel
       cross nodes (InfiniBand/RoCE). Configs violating this are rejected.
- H-4: Declare a compute budget (max GPU-hours). Exceeding it stops the job.
- H-5: Prefer the cheapest method that meets the eval gate (QLoRA → LoRA → full).

## Data Governance
- DG-1: Dataset source, license, revision recorded. PII screened. (NeMo Curator where possible.)
- DG-2: Never train on the eval/test split. Splits fixed once with a seed.

## Evaluation Gate
- E-1: "Done" = beating the base model on a held-out metric by a declared threshold.
- E-2: A run failing the gate is NOT promoted/registered (saved as a marked experiment only).
- E-3 (RL): Report reward curve AND an independent held-out eval (guard against reward hacking).

## Quality
- Q-1: Public functions typed + documented.  Q-2: Every FR/AC maps to >=1 automated test.
```

---

## 5. Step 2 — The Spec (with enforceable matrices)

`specs/spec.md` — the authoritative requirements **and** the machine-enforced matrices. This is what makes the methodology *enforce* correct choices across methods, backends, and hardware.

```markdown
# Spec: NVIDIA Fine-Tuning Run

## Functional Requirements
- FR-001: A run is fully determined by one config (scope, objective, backend, model, data,
          training args, GPU plan, eval).
- FR-002: The config declares exactly one scope and one objective. (M-1)
- FR-003: The system validates the Scope, Objective, Backend, and GPU matrices and FAILS
          before launch on any violation. (M-2)
- FR-004: For qlora, the base loads in 4-bit (NF4) and adapters are attached; base frozen.
- FR-005: For RLVR, a verifier environment is registered; no learned reward_model. (M-3)
- FR-006: GRPO/RLVR set num_generations>1 and declare max_kl (stop on drift).
- FR-007: The GPU plan MUST pass the GPU-fit preflight (H-2): est_vram <= usable_vram.
- FR-008: TP_size <= gpus_per_node (TP stays intra-node). (H-3)
- FR-009: Secrets (HF_TOKEN, registry creds) come from env vars, never config/code.
- FR-010: Outputs are self-describing: config + container tag + git commit + metrics.

## ── SCOPE MATRIX ──
| Field            | full | lora | dora | qlora | ptuning | ia3 |
|------------------|------|------|------|-------|---------|-----|
| peft_scheme      | none | lora | dora | lora  | ptuning | ia3 |
| load_in_4bit     | no   | no   | no   | YES   | no      | no  |
| freeze_base      | no   | YES  | YES  | YES   | YES     | YES |
| adapter dim/alpha| n/a  | REQ  | REQ  | REQ   | n/a     | n/a |

## ── OBJECTIVE MATRIX ──
| Field           | sft | dpo            | rm           | grpo/gspo/dapo | rlvr               | distill        |
|-----------------|-----|----------------|--------------|----------------|--------------------|----------------|
| dataset         | prompt→resp | pref pairs | pref pairs   | prompts(+gt)   | prompts + ground_truth | prompts + teacher |
| reward_env      | no  | no             | no           | optional       | REQUIRED (verifier)| no             |
| reward_model    | no  | no             | (is output)  | optional       | FORBIDDEN          | no             |
| num_generations | n/a | n/a            | n/a          | >1             | >1                 | n/a            |
| max_kl          | n/a | n/a            | n/a          | REQ            | REQ                | REQ            |
| framework       | NeMo2.0/RL | NeMo-RL | NeMo-RL      | NeMo-RL        | NeMo-RL            | NeMo-RL        |

## ── BACKEND MATRIX ──
- B-1: backend ∈ {dtensor, megatron}. generation ∈ {vllm, sglang, megatron} (RL only).
- B-2: MoE / expert parallel (EP) REQUIRES backend=megatron.
- B-3: Context Parallel (CP) for very long sequences REQUIRES backend=megatron.
- B-4: dtensor is allowed for full/lora/qlora SFT + GRPO/DPO on dense models up to mid-scale.

## ── GPU MATRIX (preflight, H-1/H-2/H-3) ──
- G-1: gpu_type ∈ {L4, L40S, RTX6000Ada, A100-40, A100-80, H100-80, H200-141, B200-192}.
- G-2: est_vram_per_gpu = ceil( total_est_vram / (num_gpus) ) MUST be <= usable_vram(gpu_type)
       where usable_vram = 0.9 × physical_vram (10% reserved for fragmentation/CUDA).
- G-3: TP_size MUST divide num_gpus AND be <= gpus_per_node. PP_size × TP_size × DP_size == num_gpus.
- G-4: FP8 precision REQUIRES Hopper (H100/H200) or Blackwell (B200); FP4 REQUIRES Blackwell.

## Acceptance Criteria
- AC-1: qlora config with load_in_4bit=false → rejected. (Scope matrix)
- AC-2: rlvr config with a reward_model → rejected. (Objective matrix / M-3)
- AC-3: MoE model with backend=dtensor + EP>1 → rejected. (B-2)
- AC-4: GPU plan where est_vram_per_gpu > usable_vram → rejected. (G-2)
- AC-5: TP_size > gpus_per_node → rejected. (G-3/FR-008)
- AC-6: FP8 requested on A100 → rejected. (G-4)
- AC-7: a run beating base by < threshold → eval gate FAIL, promotion blocked. (E-2)

## Evaluation Gate
- EV-METRIC / EV-THRESHOLD / EV-DECODING declared here (e.g., +0.05 ROUGE-L or exact-match,
  greedy decoding, fixed max_new_tokens).
```

---

## 6. Step 3 — The Plan

`specs/plan.md`:

```markdown
# Plan: NVIDIA Fine-Tuning

## Framework choice (decision tree)
- Objective = sft + scope ∈ {full,lora,dora,qlora,ptuning,ia3} on a NeMo-supported model → NeMo 2.0 recipe.
- Objective ∈ {dpo,grpo,gspo,dapo,rm,rlvr,distill}                                        → NeMo-RL.
- Need an HF checkpoint not yet native to NeMo                                            → NeMo AutoModel (DTensor).
- Want a managed API (no cluster ops)                                                     → NeMo Customizer microservice.

## Backend
- Dense model, <= ~30B, prototyping        → dtensor (FSDP2/TP).
- Large (>=70B), long context, or MoE        → megatron (TP/PP/CP/SP/EP).
- RL generation engine                       → vllm (default) or sglang.

## Parallelism plan (feeds the GPU matrix)
- TP within node (<= gpus_per_node).  PP/DP across nodes.  CP for long seq (megatron).  EP for MoE.

## GPU plan
- Produced by the gpu_planner (Section 10): gpu_type, num_gpus, num_nodes, TP/PP/CP/DP, est_vram.

## Eval protocol
- Fixed decoding; held-out test split; metric + threshold from spec. RL also logs reward + KL.

## Constitution check: R-* (container/seed/manifest), M-* (config.py), H-* (gpu preflight),
##   DG-* (Curator/splits), E-* (eval gate), Q-* (tests).
```

---

## 7. Step 4 — Tasks

`specs/tasks.md`:

```markdown
| ID | Task                                                        | Satisfies        | Test |
|----|-------------------------------------------------------------|------------------|------|
| T1 | RunConfig schema enforcing Scope/Objective/Backend/GPU mats | FR-002/003, M-*  | test_matrix_enforcement.py |
| T2 | gpu_planner: VRAM estimate + GPU-fit preflight              | FR-007, H-2, G-2 | test_gpu_planner.py |
| T3 | Data prep + license/PII screen + seeded splits (Curator)    | DG-1/2           | test_data.py |
| T4 | NeMo 2.0 SFT/PEFT recipe builder (lora/dora/qlora/full)     | FR-004, Scope    | test_smoke_nemo.py (GPU) |
| T5 | NeMo-RL config builder (dpo/grpo/rlvr) + verifier env       | FR-005/006, Obj  | test_smoke_rl.py (GPU) |
| T6 | Eval gate (held-out metric + threshold)                    | E-1/2, EV-*      | test_eval_gate.py |
| T7 | Manifest writer (config + container + commit + metrics)     | FR-010, R-2      | test_manifest.py |
| T8 | Map every FR/AC to a test                                   | Q-2              | test_spec_coverage.py |
```

---

## 8. Step 5 — Implement

### 8.1 Config enforcement (pydantic)

`src/nvft/config.py` — rejects matrix-violating configs **before** a job ever reaches the cluster. (Excerpt — the GPU preflight is in `gpu_planner.py`, §10.4.)

```python
"""RunConfig enforces the Scope / Objective / Backend / GPU matrices (FR-003)."""
from enum import Enum
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, model_validator


class Scope(str, Enum):
    full="full"; lora="lora"; dora="dora"; qlora="qlora"; ptuning="ptuning"; ia3="ia3"

class Objective(str, Enum):
    sft="sft"; dpo="dpo"; rm="rm"; grpo="grpo"; gspo="gspo"; dapo="dapo"; rlvr="rlvr"; distill="distill"

class Backend(str, Enum):
    dtensor="dtensor"; megatron="megatron"

class GPUType(str, Enum):
    L4="L4"; L40S="L40S"; RTX6000Ada="RTX6000Ada"; A100_40="A100-40"; A100_80="A100-80"
    H100_80="H100-80"; H200_141="H200-141"; B200_192="B200-192"

PHYS_VRAM = {  # physical GB per GPU
    "L4":24,"L40S":48,"RTX6000Ada":48,"A100-40":40,"A100-80":80,
    "H100-80":80,"H200-141":141,"B200-192":192,
}
FP8_OK = {"H100-80","H200-141","B200-192"}    # Hopper/Blackwell
FP4_OK = {"B200-192"}                          # Blackwell only


class Adapter(BaseModel):
    dim: int = Field(16, gt=0)         # NeMo calls LoRA rank "dim"
    alpha: int = Field(32, gt=0)
    target_modules: List[str] = ["linear_qkv", "linear_proj"]

class GPUPlan(BaseModel):
    gpu_type: GPUType
    num_gpus: int = Field(..., gt=0)
    num_nodes: int = Field(1, gt=0)
    tp: int = 1; pp: int = 1; cp: int = 1; dp: int = 1; ep: int = 1
    precision: str = "bf16"            # bf16 | fp8 | fp4

class RLSpec(BaseModel):
    reward_env: Optional[str] = None   # NeMo-RL verifier environment name (RLVR)
    reward_model: Optional[str] = None
    num_generations: int = 1
    max_kl: Optional[float] = None

class RunConfig(BaseModel):
    scope: Scope
    objective: Objective
    backend: Backend = Backend.dtensor
    is_moe: bool = False
    load_in_4bit: bool = False
    freeze_base: bool = False
    adapter: Optional[Adapter] = None
    gpu: GPUPlan
    rl: RLSpec = RLSpec()

    @model_validator(mode="after")
    def _enforce(self):
        # ── SCOPE MATRIX ──
        peft = {Scope.lora, Scope.dora, Scope.qlora, Scope.ptuning, Scope.ia3}
        if self.scope in peft and not self.freeze_base:
            raise ValueError(f"scope={self.scope.value} REQUIRES freeze_base=true")
        if self.scope is Scope.qlora and not self.load_in_4bit:
            raise ValueError("scope=qlora REQUIRES load_in_4bit=true (AC-1)")
        if self.scope is Scope.full and self.load_in_4bit:
            raise ValueError("scope=full forbids load_in_4bit")
        if self.scope in {Scope.lora, Scope.dora, Scope.qlora} and self.adapter is None:
            raise ValueError(f"scope={self.scope.value} REQUIRES an adapter block")

        # ── OBJECTIVE MATRIX ──
        if self.objective in {Objective.grpo, Objective.gspo, Objective.dapo, Objective.rlvr}:
            if self.rl.num_generations <= 1:
                raise ValueError("RL objective REQUIRES num_generations>1 (FR-006)")
            if self.rl.max_kl is None:
                raise ValueError("RL objective REQUIRES max_kl (FR-006)")
        if self.objective is Objective.rlvr:
            if not self.rl.reward_env:
                raise ValueError("rlvr REQUIRES a reward_env (verifier) (FR-005)")
            if self.rl.reward_model is not None:
                raise ValueError("rlvr FORBIDS a learned reward_model (AC-2/M-3)")

        # ── BACKEND MATRIX ──
        if self.is_moe and self.gpu.ep > 1 and self.backend is not Backend.megatron:
            raise ValueError("MoE expert parallel (EP>1) REQUIRES backend=megatron (AC-3/B-2)")
        if self.gpu.cp > 1 and self.backend is not Backend.megatron:
            raise ValueError("Context Parallel (CP>1) REQUIRES backend=megatron (B-3)")

        # ── GPU MATRIX ──
        gpus_per_node = self.gpu.num_gpus // self.gpu.num_nodes
        if self.gpu.tp > gpus_per_node:
            raise ValueError("TP_size MUST stay within a node (<= gpus_per_node) (AC-5/G-3)")
        if self.gpu.tp * self.gpu.pp * self.gpu.dp * self.gpu.ep != self.gpu.num_gpus:
            raise ValueError("tp*pp*dp*ep MUST equal num_gpus (G-3)")
        if self.gpu.precision == "fp8" and self.gpu.gpu_type.value not in FP8_OK:
            raise ValueError("fp8 REQUIRES Hopper/Blackwell (AC-6/G-4)")
        if self.gpu.precision == "fp4" and self.gpu.gpu_type.value not in FP4_OK:
            raise ValueError("fp4 REQUIRES Blackwell (G-4)")
        return self


def load_config(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as fh:
        return RunConfig.model_validate(yaml.safe_load(fh))
```

### 8.2 NeMo 2.0 — SFT & PEFT (LoRA / DoRA / QLoRA)

NeMo 2.0 uses **NeMo-Run recipes**. The `peft_scheme` selects the scope; `None` = full SFT. (Convert the HF checkpoint once with `llm.import_ckpt`.)

```python
"""NeMo 2.0 PEFT/SFT recipe — driven by our validated RunConfig."""
from nemo.collections import llm
import nemo_run as run
from nvft.config import RunConfig, Scope

def build_nemo_recipe(cfg: RunConfig, ckpt_dir: str):
    # peft_scheme: "lora" | "dora" | None (full SFT). QLoRA = lora + 4-bit base load.
    scheme = {Scope.lora: "lora", Scope.dora: "dora", Scope.qlora: "lora",
              Scope.full: None}.get(cfg.scope, "lora")

    recipe = llm.llama31_8b.finetune_recipe(   # pick the model family recipe you need
        dir=ckpt_dir,
        name=f"{cfg.objective.value}_{cfg.scope.value}",
        num_nodes=cfg.gpu.num_nodes,
        num_gpus_per_node=cfg.gpu.num_gpus // cfg.gpu.num_nodes,
        peft_scheme=scheme,
    )
    # Parallelism from the GPU plan (Megatron backend):
    recipe.trainer.strategy.tensor_model_parallel_size = cfg.gpu.tp
    recipe.trainer.strategy.pipeline_model_parallel_size = cfg.gpu.pp
    recipe.trainer.strategy.context_parallel_size = cfg.gpu.cp

    if scheme:                                  # adapter hyperparams (LoRA/DoRA)
        recipe.peft.dim = cfg.adapter.dim
        recipe.peft.alpha = cfg.adapter.alpha
        recipe.peft.target_modules = cfg.adapter.target_modules
    return recipe

# run.run(build_nemo_recipe(cfg, "/checkpoints/llama31_8b"))
```

```bash
# One-time HF -> NeMo conversion, then launch:
# python convert.py    # calls llm.import_ckpt(model=llm.llama31_8b.model(), source="hf://meta-llama/Llama-3.1-8B")
# Or use the NeMo-Run CLI directly:  nemo llm finetune --factory llama31_8b peft=lora ...
```

### 8.3 NeMo-RL — DPO / GRPO / RLVR

NeMo-RL is **YAML-config driven**; the backend (DTensor vs Megatron) is auto-selected from the config. For **RLVR**, you register a verifier **environment** that returns rewards.

```yaml
# configs/grpo_rlvr_math.yaml  (NeMo-RL style — illustrative shape)
policy:
  model_name: Qwen/Qwen3-8B
  train_backend: megatron          # or dtensor
  parallelism: { tensor: 4, pipeline: 1, context: 1 }
  peft: { scheme: lora, dim: 16, alpha: 32 }   # LoRA-GRPO supported

grpo:
  num_generations: 8               # group size K (FR-006)
  max_kl: 0.05                     # stop on policy drift
  kl_beta: 0.001

generation:
  backend: vllm                    # rollout engine (or sglang)
  max_new_tokens: 1024

environment:                       # ← RLVR: the verifier is the reward
  name: math_verify                # deterministic checker vs ground_truth
  ground_truth_field: answer

data:
  train: openai/gsm8k
  license: mit
cluster: { gpus_per_node: 8, num_nodes: 1 }   # Ray places policy/gen/reward workers
```

```python
# A NeMo-RL verifier environment for RLVR (deterministic reward — no learned RM).
import re
def math_verify(prompt, completion, answer, **kw) -> float:
    m = re.search(r"####\s*(-?[\d.,]+)", completion)
    pred = m.group(1).replace(",", "") if m else None
    return 1.0 if pred is not None and pred == str(answer).replace(",", "") else 0.0
```

```bash
# NeMo-RL launches via Ray (single node to thousands of GPUs):
# uv run examples/run_grpo.py --config configs/grpo_rlvr_math.yaml
# DPO:   uv run examples/run_dpo.py  --config configs/dpo.yaml
# SFT:   uv run examples/run_sft.py  --config configs/sft.yaml
```

> Secrets (`HF_TOKEN`, NGC registry creds) come from env vars (FR-009). The NeMo-RL Ray cluster places **policy**, **generation (vLLM)**, and **reward/reference** workers on GPUs — which is *why RL needs more VRAM than SFT* (see §10.5).

---

## 9. GPU Selection Guidance

Pick the GPU from **three questions**: *How big is the model? Which method (memory multiplier)? How long is the context?*

### NVIDIA GPU cheat-sheet for fine-tuning

| GPU | VRAM | Tier | Interconnect | Best fit for fine-tuning |
|---|---|---|---|---|
| **RTX 4090 / 5090** | 24 / 32 GB | Consumer | PCIe (no NVLink) | QLoRA/LoRA on ≤8–13B; dev & learning. No ECC/NVLink → not for serious multi-GPU. |
| **L4** | 24 GB | Datacenter (low power) | PCIe | QLoRA small models; cost-efficient; inference-leaning. |
| **L40S** | 48 GB | Datacenter | PCIe | LoRA/QLoRA up to ~30B; versatile single-GPU workhorse. |
| **RTX 6000 Ada** | 48 GB | Workstation | PCIe | Local LoRA up to ~30B; desk-side dev. |
| **A100** | 40 / 80 GB | Datacenter (Ampere) | NVLink/NVSwitch | Mature multi-GPU; LoRA/full on 8–70B (multi-GPU). No FP8. |
| **H100** | 80 GB | Datacenter (Hopper) | NVLink/NVSwitch + IB | The default for serious training; **FP8** support; full FT & RL at scale. |
| **H200** | 141 GB | Datacenter (Hopper) | NVLink/NVSwitch + IB | More HBM → fit big models on **fewer** GPUs; great for 70B LoRA / RL single-node. |
| **B200 / GB200** | 192 GB | Datacenter (Blackwell) | NVLink 5 / NVSwitch + IB | Largest models; **FP8 + FP4**; best perf/GPU; 405B-class work. |

### Selection rules of thumb

- **QLoRA, ≤13B** → a **single** 24–48 GB GPU (L4/L40S/4090/RTX 6000 Ada).
- **LoRA, ≤13B** → single **A100-80 / H100-80**. **LoRA 70B** → 2–4× 80 GB, or **1× H200/B200**.
- **Full FT** → **datacenter multi-GPU/multi-node** H100/H200/B200 with **NVLink + InfiniBand**. (See the GPU-count math in §10.)
- **RL / GRPO / RLVR** → budget **~2–3× the SFT memory** (policy + reference + generation engine). Multi-node via Ray; ideally dedicate some GPUs to the **vLLM/SGLang generation** workers.
- **Long context (≥32K)** → you need **Context Parallel (CP)** → **Megatron backend** on H100/H200/B200.
- **MoE models** → **Expert Parallel (EP)** → **Megatron backend**; size for **total** params (all experts live in VRAM) even though only active experts compute.
- **Throughput / cost** → use **FP8** on Hopper/Blackwell; **FP4** on Blackwell for the largest models.

### Interconnect matters as much as VRAM

- **Within a node:** **NVLink/NVSwitch** (fast). Put **Tensor Parallel (TP)** here — it's communication-heavy.
- **Across nodes:** **InfiniBand/RoCE**. Put **Pipeline (PP)** and **Data/FSDP (DP)** parallel here.
- A DGX/HGX node = **8 GPUs** fully connected by NVSwitch. This is why **TP_size ≤ 8** (≤ gpus_per_node) is an enforced rule (Constitution H-3 / matrix G-3).

---

## 10. How to Calculate the GPUs You Need

### 10.1 The VRAM formula (where every gigabyte goes)

Training VRAM is the **sum of four (training: five) components**:

```
total_vram  =  weights  +  gradients  +  optimizer_states  +  activations  (+ KV/rollout for RL)
```

Per-component, in **bytes per parameter**:

| Component | Full FT | LoRA / QLoRA |
|---|---|---|
| **Weights (base)** | 2 (bf16) | 2 (bf16 LoRA) · **0.55 (4-bit QLoRA)** |
| **Gradients** | 2 (all params) | ~0 (only tiny adapters) |
| **Optimizer (AdamW-32)** | **8** (2 fp32 moments) + 4 (fp32 master) = up to **12** | ~0 (adapters only) |
| **Activations** | scales with batch × seq × hidden × layers; **gradient checkpointing cuts ~85%** | same, but on frozen base |

**The headline rules of thumb:**
- **Full fine-tuning (bf16 + AdamW-32):** ≈ **16 GB of VRAM per 1B parameters** (2 weights + 2 grad + 12 optimizer/master). An 8-bit optimizer cuts this to ≈ **10 GB/1B**.
- **LoRA (bf16):** ≈ **base weights + ~25%** (gradients/optimizer for adapters are negligible).
- **QLoRA (4-bit NF4):** ≈ **0.55 GB/1B for the base + ~40% overhead** — the cheapest way to train large models on one GPU.

> **Why full FT is so expensive:** the **optimizer states dominate** — an 8B model needs ~64 GB for AdamW states alone. **Why QLoRA is a big win:** the **frozen 4-bit base dominates**, and you shrink it ~4×.

### 10.2 Reference VRAM table (by model size & method)

Estimated **training** VRAM (GB), bf16, AdamW-32 for full FT; modest batch/seq with gradient checkpointing. *Planning estimates — validate with a smoke run.*

| Model | Full FT | LoRA | QLoRA (4-bit) |
|---:|---:|---:|---:|
| **1B** | ~16 | ~2 | ~1 |
| **3B** | ~48 | ~8 | ~2 |
| **8B** | ~128 | ~20 | ~6 |
| **13B** | ~208 | ~32 | ~10 |
| **30B** | ~480 | ~75 | ~23 |
| **70B** | ~1,120 | ~175 | ~54 |
| **405B** | ~6,480 | ~1,012 | ~312 |

> **Cross-check (8-bit optimizer, published):** a widely cited table puts **70B full ≈ 672 GB**, **70B LoRA ≈ 146 GB**, **70B QLoRA-4 ≈ 46 GB** — lower than the table above because it assumes an 8-bit optimizer and tighter activations. **The takeaway is the same:** full FT is ~7–25× the memory of LoRA/QLoRA. Always confirm with your framework's smoke run; numbers vary with batch size, sequence length, optimizer precision, and parallelism.

### 10.3 From VRAM to GPU count

```
usable_per_gpu  =  0.9 × physical_vram(gpu_type)          # reserve ~10% for CUDA/frag
num_gpus        =  ceil( total_vram × overhead / usable_per_gpu )
                   # overhead ≈ 1.2–1.4 for parallelism comms buffers, framework, replication
```

Then **round up to a legal parallel layout**: `num_gpus = TP × PP × DP × EP`, with `TP ≤ gpus_per_node`. In practice you round up to whole **nodes of 8** for multi-node jobs.

**Worked GPU counts (×1.3 overhead, 80 GB H100):**

| Model | Method | Est. VRAM | GPUs (H100-80) | Typical layout |
|---|---|---:|---:|---|
| **8B** | QLoRA | ~6 GB | **1** | single GPU |
| **8B** | LoRA | ~20 GB | **1** | single GPU |
| **8B** | Full FT | ~128 GB | **3 → round to 4** | 1 node, TP=4 (or 1× H200/B200) |
| **70B** | QLoRA | ~54 GB | **1** (fits 1× 80 GB) | single H100/H200 |
| **70B** | LoRA | ~175 GB | **3 → 4** | 1 node, TP=4 |
| **70B** | Full FT | ~1,120 GB | **19 → round to 24** | 3 nodes × 8, TP=8 × PP/DP |
| **405B** | QLoRA | ~312 GB | **6 → 8** | 1 node × 8, TP=8 |
| **405B** | LoRA | ~1,012 GB | **17 → 24** | 3 nodes × 8 |
| **405B** | Full FT | ~6,480 GB | **106 → 128** | 16 nodes × 8 (Megatron TP/PP/DP) |

> **Reading it:** the same 70B model goes from **1 GPU (QLoRA)** to **~24 GPUs (full FT)** — a 24× hardware swing decided entirely by the *method*. This is exactly why the spec forces you to declare scope + GPU plan together and **preflight the fit** before queuing.

### 10.4 The `gpu_planner.py` calculator

`src/nvft/gpu_planner.py` — turns a model size + method + GPU into an estimate and a **preflight pass/fail** (enforces G-2). Wire its output into the config so a non-fitting plan is rejected.

```python
"""Estimate training VRAM and the number of NVIDIA GPUs required, then preflight the plan."""
import math
from dataclasses import dataclass

PHYS_VRAM = {"L4":24,"L40S":48,"RTX6000Ada":48,"A100-40":40,"A100-80":80,
             "H100-80":80,"H200-141":141,"B200-192":192}

@dataclass
class Estimate:
    total_vram_gb: float
    num_gpus: int
    fits_one_gpu: bool

def estimate_vram_gb(params_b: float, method: str, opt_bytes: int = 12) -> float:
    """params_b in billions. method ∈ {full, lora, qlora}. opt_bytes: 12 (AdamW-32) or 4 (8-bit)."""
    if method == "full":
        bytes_per = 2 + 2 + opt_bytes          # weights + grad + optimizer/master
        return params_b * bytes_per            # GB (1e9 params × bytes = GB)
    if method == "lora":
        return params_b * 2 * 1.25             # frozen bf16 base + ~25%
    if method == "qlora":
        return params_b * 0.55 * 1.4           # 4-bit base + ~40%
    raise ValueError(method)

def plan_gpus(params_b: float, method: str, gpu_type: str,
              overhead: float = 1.3, rl: bool = False) -> Estimate:
    vram = estimate_vram_gb(params_b, method)
    if rl:
        vram *= 2.5                            # policy + reference + generation engine (§10.5)
    usable = 0.9 * PHYS_VRAM[gpu_type]
    n = math.ceil(vram * overhead / usable)
    return Estimate(round(vram, 1), n, n == 1)

def preflight(params_b, method, gpu_type, num_gpus, **kw) -> bool:
    """G-2: declared num_gpus must be enough. Returns True if the plan fits."""
    needed = plan_gpus(params_b, method, gpu_type, **kw).num_gpus
    return num_gpus >= needed

if __name__ == "__main__":
    for p, m in [(8,"qlora"),(8,"full"),(70,"lora"),(70,"full"),(405,"full")]:
        e = plan_gpus(p, m, "H100-80")
        print(f"{p}B {m:6}: ~{e.total_vram_gb:>7} GB  ->  {e.num_gpus} × H100-80")
```

### 10.5 Extra memory for RL

RL post-training holds **more than one model at once**, so it needs far more VRAM than SFT of the same model:

- **Policy** (the model being trained — full training footprint).
- **Reference policy** (frozen copy for the KL term — ~base weights).
- **Reward model** (only for learned-RM RLHF; **not** for RLVR — verifiers are free).
- **Generation/rollout engine** (vLLM/SGLang holds its **own** copy of the weights + a large **KV cache** for sampling K long completions).

**Rule of thumb:** budget **~2–3× the SFT footprint** for GRPO/RLVR of the same model+scope, and ideally place the **generation workers on separate GPUs** (NeMo-RL + Ray do this). Long rollouts (reasoning traces) make the KV cache the silent memory hog — size `num_generations × max_new_tokens` deliberately.

### 10.6 Parallelism cheat-sheet (TP, PP, CP, SP, EP, FSDP)

When one GPU isn't enough, you **shard** the work. NVIDIA (Megatron Core) supports 6-D parallelism:

| Parallelism | Splits… | Put it… | Use when |
|---|---|---|---|
| **TP** (Tensor) | each layer's matrices across GPUs | **inside a node** (NVLink) | model layer too big for one GPU |
| **PP** (Pipeline) | layers into stages | **across nodes** (IB) | model too deep/big for one node |
| **DP / FSDP** | the batch (and shards params/opt states) | **across nodes** | scale throughput; FSDP cuts memory |
| **CP** (Context) | the sequence dimension | within/across (Megatron) | very long context (≥32K) |
| **SP** (Sequence) | layernorm/dropout activations | with TP | reduce activation memory |
| **EP** (Expert) | MoE experts across GPUs | Megatron only | MoE models (Mixtral, DeepSeek, Qwen-MoE) |

**Golden rules (enforced by the GPU matrix):** `num_gpus = TP × PP × DP × EP`; `TP ≤ gpus_per_node`; TP/SP on NVLink, PP/DP on InfiniBand; EP/CP require the **Megatron** backend.

---

## 11. How to Test (Pre-Launch Gates)

The expensive part of NVIDIA fine-tuning is the **cluster time**. So the most valuable tests run on a **laptop, before the job queues**.

| Layer | Proves | GPU? | When |
|---|---|---|---|
| **1 Matrix enforcement** | Scope/Objective/Backend/GPU configs reject illegal combos (AC-1…AC-6) | No | Every commit |
| **2 GPU-fit preflight** | `est_vram ≤ usable_vram`; legal TP×PP×DP=num_gpus; FP8 only on Hopper+ | No | Every commit + pre-launch |
| **3 Data integrity** | License recorded, PII screened, no train/test leakage (DG) | No | Every commit |
| **4 RLVR verifier** | Reward env deterministic + adversarial (no reward hacking) | No | Every commit |
| **5 Smoke train (1 step)** | NeMo recipe / NeMo-RL config actually launches and saves | Yes (tiny) | Nightly / pre-run |
| **6 Eval gate** | Run beats base by threshold or promotion is blocked (E-2) | No (logic) | Every run |
| **7 Spec coverage** | Every FR/AC has a test (Q-2) | No | Every commit |

```python
# Layer 2 — the test that saves the most money: catch an un-fittable job before it queues.
from nvft.gpu_planner import preflight
def test_ac4_oversized_job_rejected():            # AC-4 / G-2
    # 70B full FT on a single H100-80 cannot fit -> preflight must say False
    assert preflight(70, "full", "H100-80", num_gpus=1) is False
def test_qlora_8b_fits_one_gpu():
    assert preflight(8, "qlora", "L40S", num_gpus=1) is True
```

```python
# Layer 1 — illegal hardware/method combos rejected before launch
import pytest
from pydantic import ValidationError
from nvft.config import RunConfig
def test_ac6_fp8_on_a100_rejected():              # AC-6 / G-4
    with pytest.raises(ValidationError):
        RunConfig.model_validate({... , "gpu": {"gpu_type": "A100-80", "num_gpus": 8,
                                                "precision": "fp8", "tp": 8}})
```

> Add the `FR-`/`AC-` IDs as comments in each test, and keep the **Layer-7 coverage gate** so a new requirement can't ship untested — same anti-drift discipline as the other guides.

---

## 12. Files: Necessary vs. Not

### ✅ Necessary
| File | Why |
|---|---|
| `specs/spec.md` | The matrices (Scope/Objective/Backend/GPU) — the source of truth. |
| `specs/constitution.md` | Reproducibility, hardware, governance, eval-gate rules. |
| `src/nvft/config.py` | Enforces every matrix before the cluster is touched. |
| `src/nvft/gpu_planner.py` | VRAM estimate + GPU-fit preflight (saves cluster $). |
| `configs/*.yaml` | One run = one config (NeMo recipe args / NeMo-RL YAML). |
| `tests/test_matrix_enforcement.py`, `test_gpu_planner.py`, `test_spec_coverage.py` | Cheap, no-GPU proof the spec holds. |
| `run_manifest.json` (generated) | Config + **container tag** + git commit + metrics (reproducibility). |
| Pinned **NGC container tag** + `requirements`/`uv.lock` | Reproducible environment. |
| `.env.example`, `.gitignore` | Secrets via env; never commit tokens. |

### ⚠️ Optional
| File | When |
|---|---|
| `cluster/*.sub` (Slurm) or K8s/KubeRay manifests | Multi-node runs. |
| `configs/full.yaml` | Only if you'll actually run full FT. |
| NeMo Curator pipelines | Large/raw datasets needing curation. |
| W&B / MLflow logging | Team experiment tracking. |

### ❌ Not necessary (anti-patterns)
| Anti-pattern | Why avoid |
|---|---|
| A single launch notebook with hard-coded parallelism + GPU counts | Unreproducible; bypasses the preflight; the classic "OOM at hour 3." |
| Hard-coded HF/NGC tokens in YAML or code | Secrets belong in env vars (FR-009). |
| One script per method (`train_lora.py`, `train_grpo.py`, …) | Path drift; use one config-driven entrypoint per framework. |
| Launching a 70B full-FT job without a VRAM estimate | Wastes a queue slot + cluster hours on a job that can't fit. |
| TP spanning nodes (TP_size > gpus_per_node) | Cripples throughput over InfiniBand; rejected by G-3. |
| Unpinned container / no seed / no manifest | The run can never be reproduced or audited. |

---

## 13. Pros & Cons

### NVIDIA stack vs. a plain PyTorch/HF stack
**Pros:** purpose-built for **scale** (Megatron 6-D parallelism, FP8/FP4, MoE); **NeMo-RL** gives production GRPO/DPO/RLVR/distillation with Ray + vLLM out of the box; tight GPU/kernel optimization (best throughput on NVIDIA hardware); enterprise path via **NeMo Customizer** + NIM serving; day-0 HF support via **AutoModel**.
**Cons:** **steeper learning curve** (NeMo-Run, Megatron configs, Ray); heavier setup (NGC containers, multi-node); checkpoint **conversion** (HF↔NeMo) adds a step; rapidly moving repos (the NeMo org is mid-split) — pin versions and expect churn; smaller single-GPU/hobbyist ergonomics than HF TRL.

### When to choose which NVIDIA tool
| If you want… | Use |
|---|---|
| Quick SFT/PEFT on a supported model | **NeMo 2.0 recipe** |
| GRPO/DPO/RLVR/RM/distillation at scale | **NeMo-RL** |
| An HF checkpoint fine-tuned without a port | **NeMo AutoModel** |
| Largest models / MoE / long context | **Megatron Core** backend |
| A managed API, minimal infra | **NeMo Customizer** microservice |
| Fastest prototyping, hackable | **DTensor** backend |

*(Per-method pros/cons — SFT vs LoRA/QLoRA vs DPO/KTO/GRPO/RLVR — are covered in the companion Hugging Face guide and apply unchanged here.)*

---

## 14. Pitfalls & Best Practices

- **Preflight before you queue.** The GPU-fit test (Layer 2) is the single highest-ROI check — it converts a 3-hour OOM into a 3-millisecond failure.
- **Declare scope + GPU plan together.** A method choice *is* a hardware choice (70B = 1 GPU as QLoRA, ~24 as full FT).
- **TP within a node, PP/DP across nodes.** Never let Tensor Parallel cross InfiniBand.
- **Size for RL's extra models.** Budget ~2.5× and isolate generation (vLLM) workers.
- **MoE = size for total params, EP needs Megatron.** All experts live in VRAM.
- **Pin the NGC container tag.** "It worked last week" dies without it; the manifest must record it.
- **Convert once, reuse.** `llm.import_ckpt` HF→NeMo, record the checksum; don't re-convert every run.
- **Use FP8 on Hopper/Blackwell** for throughput; **FP4 on Blackwell** for the largest models — but verify quality against the eval gate.
- **RLVR: test the verifier like the model.** Deterministic + adversarial, or the model learns to cheat.
- **Prefer the cheapest method that clears the gate** (QLoRA → LoRA → full) — Constitution H-5.

---

## 15. A 1-Week Skilling Plan

| Day | Focus | Outcome |
|---|---|---|
| **1** | Read this guide + NeMo 2.0 PEFT docs + NeMo-RL overview. | Map of the NVIDIA stack and which tool runs what. |
| **2** | Build `config.py` + the four matrices; make Layer-1 enforcement tests green. | The spec *enforces* method/backend/GPU choices. |
| **3** | Build `gpu_planner.py`; reproduce the §10 tables; make Layer-2 preflight tests green. | You can size GPUs from first principles. |
| **4** | Run a **NeMo 2.0 LoRA** recipe on a small model (single GPU); produce a manifest. | First reproducible NVIDIA fine-tune. |
| **5** | Run **NeMo-RL GRPO** (or DPO) on a small model; wire a verifier env for RLVR. | RL post-training end-to-end. |
| **6** | Plan a 70B run: pick GPU, TP/PP/DP, pass preflight; (dry-run the Slurm/Ray launch). | Multi-GPU planning that actually fits. |
| **7** | Add the eval gate + Layer-7 coverage; put no-GPU tests in CI; write a retro. | A governed, hardware-right-sized workflow. |

---

## 16. References

**NVIDIA frameworks**
- **NeMo Framework — SFT & PEFT (2.0)**: <https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html>
- **PEFT in NeMo 2.0** (recipes, `peft_scheme`, LoRA/DoRA): <https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/peft_nemo2.html>
- **NeMo Framework GitHub** (NVIDIA-NeMo org): <https://github.com/NVIDIA-NeMo/NeMo>
- **NeMo-RL** (GRPO/GSPO, SFT, DPO, RM, DAPO/GDPO, distillation; DTensor + Megatron): <https://github.com/NVIDIA-NeMo/RL> · docs: <https://docs.nvidia.com/nemo/rl/latest/index.html>
- **Megatron Core** (parallelism): <https://github.com/NVIDIA/Megatron-LM>
- **NeMo Customizer / Microservices**: <https://docs.nvidia.com/nemo/microservices/latest/>

**GPU sizing & VRAM math**
- *How to Calculate GPU VRAM Requirements for an LLM* (formulas: weights/grad/optimizer/activations): <https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm>
- *How much VRAM do I need for fine-tuning?* (the 16 GB/1B rule + the full/LoRA/QLoRA table): <https://modal.com/blog/how-much-vram-need-fine-tuning>
- *Fine-Tuning VRAM Calculator (Full/LoRA/QLoRA)*: <https://inventivehq.com/tools/developer/fine-tuning-vram-calculator>
- **NVIDIA H100 / H200 / B200 datasheets** (VRAM, FP8/FP4, NVLink): <https://www.nvidia.com/en-us/data-center/>

**Method background (apply unchanged on NVIDIA)**
- RLVR / GRPO (DeepSeek-R1): <https://arxiv.org/abs/2501.12948> · DPO: <https://arxiv.org/abs/2305.18290> · LoRA: <https://arxiv.org/abs/2106.09685> · QLoRA: <https://arxiv.org/abs/2305.14314>
- **GitHub Spec Kit** (the SDD methodology applied here): <https://github.com/github/spec-kit>

---

*Built as a learning artifact. NVIDIA's NeMo repos are reorganizing rapidly (the NeMo org is mid-split, NeMo-RL is pre-1.0) — confirm recipe APIs, config schemas, and container tags against the version you install. VRAM/GPU-count figures are planning estimates; the only ground truth is a smoke run on your target hardware. Pin your NGC container, start small, and preflight before you queue.*
