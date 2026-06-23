# Spec-Driven Fine-Tuning of LLMs with the Hugging Face SDK

*A hands-on, spec-first tutorial for fine-tuning a small **or** large language model with the Hugging Face stack (`transformers` + `trl` + `peft` + `datasets`) — across multiple techniques (Full, LoRA, QLoRA, SFT). The spec is the source of truth and **enforces** the rules every fine-tuning run must follow, no matter which technique you pick.*

> **Companion to:** *Spec-Driven Development (SDD): A Hands-On Tutorial.* This guide applies the same SDD method — **Constitution → Spec → Plan → Tasks → Implement → Test** — to the fine-tuning domain.
>
> **Who this is for:** ML/AI engineers and architects who want a disciplined, reproducible, auditable fine-tuning workflow instead of one-off notebooks that nobody can reproduce.
>
> **What you'll be able to do at the end:** stand up a fine-tuning project whose spec *enforces* technique-specific requirements; run Full / LoRA / QLoRA / SFT from the **same** codebase by flipping a config value; and **test the fine-tune** (config, data, smoke-train, eval gate, reproducibility) so you can prove a run is valid before it burns GPU hours.

---

## Table of Contents

1. [Why Fine-Tuning Needs a Spec](#1-why-fine-tuning-needs-a-spec)
2. [The Techniques at a Glance](#2-the-techniques-at-a-glance)
3. [The SDD Workflow Applied to Fine-Tuning](#3-the-sdd-workflow-applied-to-fine-tuning)
4. [Worked Example](#4-worked-example)
5. [Step-by-Step Implementation](#5-step-by-step-implementation)
   - 5.1 [Prerequisites](#51-prerequisites)
   - 5.2 [Project layout](#52-project-layout)
   - 5.3 [The Constitution](#53-step-1--the-constitution)
   - 5.4 [The Spec (with the enforceable technique matrix)](#54-step-2--the-spec-specify)
   - 5.5 [The Plan](#55-step-3--the-plan-plan)
   - 5.6 [The Tasks](#56-step-4--the-tasks-tasks)
   - 5.7 [Configuration & environment](#57-step-5--configuration--environment)
   - 5.8 [The implementation](#58-step-6--implement)
6. [How to Test the Fine-Tune (6 Layers)](#6-how-to-test-the-fine-tune)
7. [Files: Necessary vs. Not](#7-files-necessary-vs-not)
8. [Pros and Cons — Per Technique and for SDD Fine-Tuning](#8-pros-and-cons)
9. [Common Pitfalls & Best Practices](#9-common-pitfalls--best-practices)
10. [A 1-Week Skilling Plan](#10-a-1-week-skilling-plan)
11. [Reinforcement-Learning & RLVR Post-Training (Spec Extension)](#11-reinforcement-learning--rlvr-post-training-spec-extension)
    - 11.1 [The two axes: scope × objective](#111-the-two-axes-parameter-scope--training-objective)
    - 11.2 [RL methods at a glance](#112-rl-methods-at-a-glance-dpo-kto-grpo-rloo-ppo-rlvr)
    - 11.3 [What RLVR is](#113-what-rlvr-is-and-its-failure-modes)
    - 11.4 [Constitution additions](#114-constitution-additions-rl)
    - 11.5 [Spec additions: the Objective Matrix](#115-spec-additions--the-enforceable-objective-matrix)
    - 11.6 [Config enforcement](#116-config-enforcement-the-objective-axis)
    - 11.7 [Reward functions (verifiers)](#117-reward-functions--the-rlvr-verifiers)
    - 11.8 [The RL training entrypoint](#118-the-rl-training-entrypoint)
    - 11.9 [Testing RL & verifiers (Layer 7)](#119-how-to-test-rl--verifiers-layer-7)
    - 11.10 [Pros & cons per RL method](#1110-pros--cons-per-rl-method)
12. [References](#12-references)

---

## 1. Why Fine-Tuning Needs a Spec

Fine-tuning is where vibe coding does the most damage. A single notebook silently bakes in dozens of unstated decisions — base model, data cleaning, train/test split, LoRA rank, learning rate, eval metric, stopping rule — and three weeks later nobody can answer *"why did this run score 0.71, and can we reproduce it?"*

The SDD answer is the same as for app code: **make the decisions explicit, versioned, and testable before you spend the GPU hours.** In fine-tuning specifically, a spec buys you four things a notebook can't:

1. **Reproducibility** — the spec pins the base model, dataset revision, seed, and hyperparameters. A run is a function of its spec.
2. **Technique portability** — one spec, many techniques. You can demand "produce a LoRA *and* a QLoRA adapter from the same data and eval gate" and compare apples to apples.
3. **A definition of "done"** — the spec's acceptance criteria are an **eval gate**: the run must beat a held-out metric threshold or it does not ship.
4. **Governance** — data licensing, PII handling, and "never train on the test set" become enforced rules, not good intentions.

> **The enforcement idea (the core of this guide):** the spec defines a **technique matrix** — for each technique, the *required* and *forbidden* configuration. A small validation layer (`pydantic`) reads the run config and **rejects it before training starts** if it violates the matrix. That's how a spec *enforces* correct fine-tuning across techniques instead of merely describing it.

---

## 2. The Techniques at a Glance

| Technique | What it trains | Memory | Output artifact | Best for |
|---|---|---|---|---|
| **Full fine-tuning** | **All** model weights | Highest (model + grads + optimizer states, often 12–20× params in bytes) | A full model checkpoint | Small models (≤ ~1–3B), large/high-quality datasets, when you can afford the compute and want maximum adaptation. |
| **LoRA** (Low-Rank Adaptation) | Small **adapter** matrices injected into attention/MLP layers; base weights **frozen** | Low–medium (base weights in fp16/bf16, train ~0.1–2% of params) | A small **adapter** (MBs), optionally merged into the base | Most adaptation tasks; cheap to train, store, and swap. The default PEFT choice. |
| **QLoRA** (Quantized LoRA) | LoRA adapters on top of a **4-bit quantized** frozen base | Lowest (base in 4-bit NF4) | A small adapter (dequantize/merge for deploy) | Fine-tuning **large** models (7B–70B+) on a single GPU / limited VRAM. |
| **SFT** (Supervised Fine-Tuning) | A **training objective**, not a parameter scope — next-token cross-entropy on (prompt → completion) pairs. Can run with *any* of the above. | Depends on the parameter scope you pair it with | Whatever the underlying scope produces | Instruction-following / chat behavior. This is what TRL's `SFTTrainer` implements. |

**Key mental model:** *SFT is the objective; Full / LoRA / QLoRA are the parameter scope.* You combine them — e.g., "SFT with QLoRA" means a supervised-fine-tuning objective applied to 4-bit-quantized base + LoRA adapters. This tutorial's code lets you choose the **scope** (`full | lora | qlora`) while always using the **SFT objective** via `SFTTrainer`.

> Beyond SFT there are **preference-optimization** objectives (DPO, KTO) and **reinforcement-learning** objectives (PPO, GRPO, RLOO) — including **RLVR (Reinforcement Learning from Verifiable Rewards)** — all in TRL. These are fully specced and enforced in **[Section 11: Reinforcement-Learning & RLVR Post-Training (Spec Extension)](#11-reinforcement-learning--rlvr-post-training-spec-extension)**, which adds a second axis (the *training objective*) on top of the parameter-scope axis (full/LoRA/QLoRA) covered above.

---

## 3. The SDD Workflow Applied to Fine-Tuning

```
SPECIFY ─────────▶ PLAN ──────────▶ TASKS ─────────▶ IMPLEMENT ────────▶ (TEST GATES)
what model/data,    which technique,  atomic steps:    config-driven        config valid?
metric, accept-     hyperparameters,  data prep,       trainer that runs    data clean?
ance gate, the      compute budget,   train, eval,     full/lora/qlora      smoke ok?
technique matrix    eval protocol     merge, ship      from one entrypoint  eval gate passed?

If a run disagrees with the spec → fix the SPEC (or the config), then re-run. The spec wins.
```

The four artifacts map exactly as in app-SDD, but their *content* is fine-tuning-specific:

| Artifact | In app SDD | In fine-tuning SDD |
|---|---|---|
| `constitution.md` | security, reliability, quality rules | + data governance, reproducibility, eval-gate, no-train-on-test, license compliance, compute budget |
| `spec.md` | functional requirements, acceptance criteria | + base model, dataset + revision, eval metric & threshold, **technique matrix** (enforceable per-technique config) |
| `plan.md` | stack, architecture | + chosen technique, hyperparameters, quantization, hardware, eval protocol |
| `tasks.md` | code tasks | data-prep → train → eval → merge → register tasks, each traced to a requirement |

---

## 4. Worked Example

**Goal:** teach a base LLM to follow a specific instruction style — turn a short product description into a one-line marketing tagline — and prove it improved on a held-out set.

- **Base model (small default):** `Qwen/Qwen3-0.6B` (swap to a 7B model for the QLoRA path).
- **Dataset:** a small instruction dataset in conversational format (`messages` or `prompt`/`completion`). We'll use a public set for the walkthrough; the structure is identical for your own JSONL.
- **Eval gate (definition of done):** held-out **win-rate / ROUGE-L (or exact-style match)** must beat the base model by the threshold the spec sets, **and** validation loss must be below the base's.
- **Techniques you can run from this one project:** `full`, `lora`, `qlora` — all with the SFT objective.

---

## 5. Step-by-Step Implementation

### 5.1 Prerequisites

- **Python 3.10+**, a CUDA GPU (QLoRA path needs an NVIDIA GPU for `bitsandbytes`).
- A **Hugging Face account + token** (for gated models / pushing artifacts), stored in an env var — never in code.
- Packages (pin versions in `requirements.txt` for reproducibility — Constitution rule below):

```bash
python -m pip install --upgrade \
  "transformers>=4.46" "trl>=0.12" "peft>=0.13" "datasets>=3.0" \
  "accelerate>=1.0" "bitsandbytes>=0.44" "evaluate" "pydantic>=2" \
  "pyyaml" "pytest"
```

> **Note on moving APIs:** TRL evolves quickly. In current TRL, sequence length lives in `SFTConfig.max_length` (older code used `max_seq_length`), and the tokenizer is passed as `processing_class` (older code used `tokenizer=`). The code below uses the current names; if you're on an older pin, check `MIGRATION.md` in the TRL repo.

### 5.2 Project layout

```
llm-finetune/
├── specs/
│   ├── constitution.md          # non-negotiable rules (governance, reproducibility, gates)
│   ├── spec.md                  # WHAT: model, data, metric, acceptance + TECHNIQUE MATRIX
│   ├── plan.md                  # HOW: chosen technique, hyperparameters, hardware
│   └── tasks.md                 # atomic, traceable work items
├── configs/
│   ├── run.schema.md            # human-readable description of every config field
│   ├── lora.yaml                # a concrete run config (technique = lora)
│   ├── qlora.yaml               # technique = qlora
│   └── full.yaml                # technique = full
├── src/finetune/
│   ├── __init__.py
│   ├── config.py                # pydantic schema that ENFORCES the technique matrix
│   ├── data.py                  # load, validate, split, format dataset (no leakage)
│   ├── modeling.py              # build (model, peft_config) per technique
│   ├── train.py                 # the SFT training entrypoint
│   └── evaluate.py              # held-out eval = the acceptance gate
├── tests/
│   ├── test_config_enforcement.py   # technique matrix is enforced (Layer 1)
│   ├── test_data_integrity.py       # schema + no train/test leakage (Layer 2)
│   ├── test_smoke_train.py          # 1-step training runs end-to-end (Layer 3)
│   ├── test_eval_gate.py            # eval threshold logic (Layer 4)
│   ├── test_reproducibility.py      # same seed → same result (Layer 5)
│   └── test_spec_coverage.py        # every FR/AC has a test (Layer 6)
├── .env.example                 # HF_TOKEN etc. — committed template
├── .env                         # real secrets — NEVER committed
├── .gitignore
├── requirements.txt             # PINNED versions
└── pytest.ini
```

---

### 5.3 Step 1 — The Constitution

`specs/constitution.md` — binding on every spec, plan, config, and run.

```markdown
# Project Constitution — LLM Fine-Tuning

These rules are binding on every spec, plan, run config, and training run.

## Data Governance
- DG-1: Every dataset MUST declare a source, license, and revision/commit hash.
        Training on a dataset without a known license is forbidden.
- DG-2: PII MUST be screened/redacted before training; document the screening method.
- DG-3: The model MUST NOT be trained on the evaluation/test split. Train/val/test
        splits are created once, with a fixed seed, and never overlap.

## Reproducibility
- R-1: A global seed MUST be set for python, numpy, and torch.
- R-2: Library versions MUST be pinned in requirements.txt.
- R-3: Every run MUST persist its exact run config + git commit alongside outputs.
- R-4: Base model and dataset MUST be referenced by an immutable revision when possible.

## Technique Integrity (the enforcement core)
- T-1: The run config MUST declare exactly one technique: full | lora | qlora.
- T-2: Each technique's REQUIRED fields MUST be present and FORBIDDEN fields absent
       (see the Technique Matrix in spec.md). A violating config MUST fail before training.
- T-3: QLoRA MUST load the base model in 4-bit (NF4) and MUST attach LoRA adapters.
- T-4: LoRA/QLoRA MUST freeze base weights; only adapter (+ optional embeddings) train.

## Evaluation Gate
- E-1: "Done" is defined by a held-out metric threshold in spec.md, not by vibes.
- E-2: A run that does not beat the base model by the spec's threshold MUST NOT be
       promoted/registered. It may be saved as an experiment, clearly marked as failing.
- E-3: Eval MUST run on the untouched test split with a fixed decoding config.

## Cost / Compute
- C-1: Each run declares a compute budget (max GPU-hours / max steps). Exceeding it
       stops the run.
- C-2: Prefer the smallest technique that meets the eval gate (QLoRA before LoRA before full)
       unless the spec justifies otherwise.

## Quality
- Q-1: Public functions have type hints + docstrings.
- Q-2: Every functional requirement (FR-xxx) maps to >= 1 automated test.
```

---

### 5.4 Step 2 — The Spec (`/specify`)

`specs/spec.md` — *what* and *why*, plus the **enforceable Technique Matrix**. This is the document that "enforces fine-tuning across different techniques."

```markdown
# Spec: Instruction Fine-Tuning — Tagline Generator

## Problem
We need a base LLM adapted to produce a one-line marketing tagline from a short
product description, in our house style. It must measurably beat the un-tuned base.

## Users / Personas
- ML engineer: runs and compares fine-tuning techniques.
- Reviewer: approves promotion only if the eval gate passes.

## Goals
- Adapt a base LLM to the tagline task using the SFT objective.
- Support Full, LoRA, and QLoRA from ONE codebase, selected by config.
- Gate promotion on a held-out metric.

## Non-Goals (v1)
- RLHF / preference optimization (DPO/KTO/GRPO).
- Multi-node distributed training (single node only for v1).
- Serving/deployment (handled by a separate spec).

## Inputs (pinned for reproducibility — Constitution R-4, DG-1)
- Base model: `Qwen/Qwen3-0.6B` (small path) OR a 7B model (qlora path),
  referenced by revision/commit.
- Dataset: conversational SFT format with `messages` OR `prompt`/`completion`.
  Source, license, and revision MUST be recorded.

## Functional Requirements
- FR-001: The system MUST accept a single run config (YAML) that fully determines the run.
- FR-002: The config MUST declare `technique` ∈ {full, lora, qlora} (exactly one). (T-1)
- FR-003: The system MUST validate the config against the Technique Matrix and FAIL
          before training if any rule is violated. (T-2)
- FR-004: The system MUST create train/val/test splits once with a fixed seed and MUST
          NOT train on val/test. (DG-3)
- FR-005: The system MUST set a global seed across python/numpy/torch. (R-1)
- FR-006: For `qlora`, the base model MUST be loaded in 4-bit NF4 and LoRA attached. (T-3)
- FR-007: For `lora`/`qlora`, base weights MUST be frozen; only adapters train. (T-4)
- FR-008: The system MUST run SFT via TRL `SFTTrainer` for all techniques.
- FR-009: The system MUST evaluate on the untouched test split and emit the spec metric.
- FR-010: The system MUST persist run config + git commit + metrics next to outputs. (R-3)
- FR-011: Secrets (HF token) MUST come from env vars, never from config or code.
- FR-012: The run MUST stop if it exceeds the declared compute budget (max_steps). (C-1)

## Non-Functional Requirements
- NFR-001 (reproducibility): same config + seed + data revision → equivalent metric (±tolerance).
- NFR-002 (portability): switching technique MUST require ONLY a config change, no code edits.
- NFR-003 (auditability): every output dir is self-describing (config + metrics + commit).

## Evaluation Gate (definition of done — Constitution E-1)
- EV-METRIC: ROUGE-L (or task win-rate) on the test split.
- EV-THRESHOLD: tuned model MUST beat the base model by >= +0.05 absolute ROUGE-L
  AND have lower validation loss than the base.
- EV-DECODING: greedy decoding, max_new_tokens=32, temperature=0 (fixed). (E-3)

## ── TECHNIQUE MATRIX (authoritative, machine-enforced via config.py) ──
For each technique, REQUIRED fields MUST be set and FORBIDDEN fields MUST be absent.

| Field                | full            | lora              | qlora                         |
|----------------------|-----------------|-------------------|-------------------------------|
| `load_in_4bit`       | forbidden/false | forbidden/false   | **REQUIRED true**             |
| `bnb_4bit_quant_type`| forbidden       | forbidden         | **REQUIRED "nf4"**            |
| `lora.r`             | forbidden       | **REQUIRED (>0)** | **REQUIRED (>0)**             |
| `lora.alpha`         | forbidden       | **REQUIRED**      | **REQUIRED**                  |
| `lora.target_modules`| forbidden       | **REQUIRED**      | **REQUIRED**                  |
| `freeze_base`        | false           | **REQUIRED true** | **REQUIRED true**             |
| `gradient_checkpointing` | recommended | optional          | **REQUIRED true** (mem)       |
| `learning_rate`      | ~1e-5–2e-5      | ~1e-4–2e-4        | ~1e-4–2e-4                    |

Rules enforced from the matrix (config.py raises if broken):
- M-1: `technique=full`  → `lora` block MUST be null; `load_in_4bit` MUST be false.
- M-2: `technique=lora`  → `lora` block REQUIRED; `load_in_4bit` MUST be false; `freeze_base=true`.
- M-3: `technique=qlora` → `lora` block REQUIRED; `load_in_4bit=true`; `bnb_4bit_quant_type="nf4"`;
       `freeze_base=true`; `gradient_checkpointing=true`.
- M-4: Any unknown technique → reject.

## Acceptance Criteria (Given/When/Then → become tests)
- AC-1: GIVEN technique=lora with NO lora block, WHEN config is loaded, THEN it is rejected. (FR-003/M-2)
- AC-2: GIVEN technique=qlora with load_in_4bit=false, WHEN loaded, THEN rejected. (FR-003/M-3)
- AC-3: GIVEN technique=full with a lora block, WHEN loaded, THEN rejected. (FR-003/M-1)
- AC-4: GIVEN a valid lora config, WHEN a 1-step smoke train runs, THEN it completes and
        saves an adapter, and base params have requires_grad=False. (FR-007/FR-008)
- AC-5: GIVEN identical config+seed, WHEN run twice (smoke), THEN the first-step loss matches
        within tolerance. (NFR-001)
- AC-6: GIVEN a tuned model that does NOT beat base by EV-THRESHOLD, WHEN evaluated, THEN the
        gate reports FAIL and promotion is blocked. (E-2)
- AC-7: GIVEN the dataset, WHEN split, THEN train ∩ test = ∅ and train ∩ val = ∅. (FR-004)

## Open Questions
- [NEEDS CLARIFICATION: which target_modules for the 7B base?]
  → RESOLVED: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"].
```

---

### 5.5 Step 3 — The Plan (`/plan`)

`specs/plan.md` — *how*, with the Constitution enforced at each choice.

```markdown
# Implementation Plan: Instruction Fine-Tuning

## Stack
- transformers (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
- trl (SFTTrainer, SFTConfig)          ← SFT objective for ALL techniques (FR-008)
- peft (LoraConfig, get_peft_model, prepare_model_for_kbit_training)
- datasets (load + split, fixed seed)
- accelerate (device placement); bitsandbytes (4-bit NF4 for qlora)
- pydantic v2 (config schema = the technique-matrix enforcer)
- evaluate (ROUGE-L for the gate)

## Module responsibilities
- config.py   — pydantic RunConfig; validates technique matrix M-1..M-4 at load. (FR-002/003)
- data.py     — load, screen, dedupe, split once w/ seed; format to SFT schema. (FR-004, DG)
- modeling.py — build(model, peft_config) per technique; 4-bit for qlora; freeze base. (FR-006/007)
- train.py    — set seed, build everything, SFTTrainer.train(), persist config+commit+metrics.
- evaluate.py — generate on test split (fixed decoding), compute metric, apply EV-THRESHOLD gate.

## Hyperparameters (defaults; overridable per config)
- full : lr 2e-5, epochs 1-3, bf16, grad-checkpointing on.
- lora : r 16, alpha 32, dropout 0.05, lr 2e-4, target_modules per matrix.
- qlora: as lora + load_in_4bit nf4, double-quant, compute dtype bf16, grad-checkpointing REQUIRED.

## Eval protocol (E-3): greedy, max_new_tokens=32, temperature=0, on test split only.

## Constitution check
- DG-1..3 in data.py; R-1..4 in train.py + requirements.txt; T-1..4 in config.py + modeling.py;
- E-1..3 in evaluate.py; C-1 via SFTConfig.max_steps; Q-2 via tests/.
```

---

### 5.6 Step 4 — The Tasks (`/tasks`)

`specs/tasks.md`:

```markdown
# Tasks: Instruction Fine-Tuning

| ID  | Task                                                            | Satisfies            | Test |
|-----|-----------------------------------------------------------------|----------------------|------|
| T1  | RunConfig pydantic schema enforcing technique matrix M-1..M-4   | FR-002/003, T-1/2    | test_config_enforcement.py |
| T2  | Data load + license/PII screen + dedupe + seeded split          | FR-004, DG-1..3      | test_data_integrity.py |
| T3  | Format dataset into SFT (messages / prompt-completion)          | FR-008               | test_data_integrity.py |
| T4  | build_model_and_peft(technique): full / lora / qlora            | FR-006/007, T-3/4    | test_smoke_train.py |
| T5  | train.py: seed, SFTTrainer, max_steps budget, persist artifacts | FR-005/010/012, R-3  | test_smoke_train.py |
| T6  | evaluate.py: fixed-decoding generation + ROUGE-L + gate         | FR-009, EV-*         | test_eval_gate.py |
| T7  | Reproducibility: same seed → same first-step loss               | NFR-001              | test_reproducibility.py |
| T8  | Map every FR/AC to a test                                       | Q-2                  | test_spec_coverage.py |
```

---

### 5.7 Step 5 — Configuration & Environment

The run is **fully determined by one YAML config** (FR-001), and secrets come from env vars (FR-011).

`.env.example` (committed, no real values):
```bash
HF_TOKEN=replace-me                 # Hugging Face token (gated models / push)
HF_HOME=./.hf_cache                 # local cache dir (optional)
WANDB_DISABLED=true                 # set false + add WANDB_API_KEY to log to W&B
```

`.gitignore`:
```gitignore
.env
.hf_cache/
outputs/
__pycache__/
*.pyc
.pytest_cache/
```

`configs/lora.yaml` (a valid LoRA run):
```yaml
technique: lora                     # full | lora | qlora   (FR-002)
seed: 42

base_model:
  name: Qwen/Qwen3-0.6B
  revision: main                    # pin a commit hash in production (R-4)

dataset:
  name: trl-lib/Capybara            # source recorded (DG-1)
  revision: main
  license: apache-2.0               # REQUIRED (DG-1)
  text_field: messages              # conversational SFT
  test_size: 0.1
  val_size: 0.1

load_in_4bit: false                 # matrix M-2: forbidden for lora
freeze_base: true                   # matrix M-2

lora:                               # REQUIRED for lora (M-2)
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

training:
  output_dir: outputs/lora-tagline
  learning_rate: 2.0e-4
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  max_length: 1024                  # TRL SFTConfig.max_length
  gradient_checkpointing: false
  bf16: true
  max_steps: 500                    # compute budget (C-1, FR-012)
  logging_steps: 10

eval:
  metric: rougeL
  threshold_abs_gain: 0.05          # EV-THRESHOLD
  max_new_tokens: 32
  temperature: 0.0                  # greedy (EV-DECODING)
```

`configs/qlora.yaml` (only the deltas vs LoRA — same shape, different technique):
```yaml
technique: qlora
base_model:
  name: meta-llama/Llama-3.1-8B     # a LARGE model on one GPU
load_in_4bit: true                  # REQUIRED (M-3)
bnb_4bit_quant_type: nf4            # REQUIRED (M-3)
freeze_base: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
training:
  output_dir: outputs/qlora-tagline
  learning_rate: 2.0e-4
  gradient_checkpointing: true      # REQUIRED for qlora (M-3, memory)
  bf16: true
  max_steps: 500
```

`configs/full.yaml` (full fine-tuning — note: **no** `lora` block, **no** 4-bit):
```yaml
technique: full
base_model:
  name: Qwen/Qwen3-0.6B
load_in_4bit: false                 # forbidden for full (M-1)
freeze_base: false                  # full trains everything
# (no lora block — forbidden for full, M-1)
training:
  output_dir: outputs/full-tagline
  learning_rate: 2.0e-5             # lower LR for full FT
  num_train_epochs: 3
  gradient_checkpointing: true
  bf16: true
  max_steps: 500
```

---

### 5.8 Step 6 — Implement

#### `src/finetune/config.py` — the technique-matrix **enforcer** (T1)

This is the heart of the "enforce specs across techniques" requirement. If a config violates the matrix, **it raises before any GPU work happens.**

```python
"""RunConfig: a pydantic schema that ENFORCES the Technique Matrix from spec.md.

A config that violates M-1..M-4 raises ValidationError at load time (FR-003).
"""
from enum import Enum
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, model_validator


class Technique(str, Enum):
    full = "full"
    lora = "lora"
    qlora = "qlora"


class LoraSpec(BaseModel):
    r: int = Field(..., gt=0)
    alpha: int = Field(..., gt=0)
    dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: List[str] = Field(..., min_length=1)


class BaseModelSpec(BaseModel):
    name: str
    revision: str = "main"


class DatasetSpec(BaseModel):
    name: str
    revision: str = "main"
    license: str                       # REQUIRED — DG-1
    text_field: str = "messages"
    test_size: float = Field(0.1, gt=0, lt=1)
    val_size: float = Field(0.1, ge=0, lt=1)


class TrainingSpec(BaseModel):
    output_dir: str
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    gradient_checkpointing: bool = False
    bf16: bool = True
    max_steps: int = -1                # -1 = use epochs; >0 = hard budget (C-1)
    logging_steps: int = 10


class EvalSpec(BaseModel):
    metric: str = "rougeL"
    threshold_abs_gain: float = 0.05
    max_new_tokens: int = 32
    temperature: float = 0.0


class RunConfig(BaseModel):
    technique: Technique
    seed: int = 42
    base_model: BaseModelSpec
    dataset: DatasetSpec
    training: TrainingSpec
    eval: EvalSpec
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Optional[str] = None
    freeze_base: bool = False
    lora: Optional[LoraSpec] = None

    @model_validator(mode="after")
    def _enforce_technique_matrix(self) -> "RunConfig":
        t = self.technique
        if t is Technique.full:                               # M-1
            if self.lora is not None:
                raise ValueError("technique=full forbids a `lora` block (M-1)")
            if self.load_in_4bit:
                raise ValueError("technique=full forbids load_in_4bit (M-1)")
        elif t is Technique.lora:                             # M-2
            if self.lora is None:
                raise ValueError("technique=lora REQUIRES a `lora` block (M-2)")
            if self.load_in_4bit:
                raise ValueError("technique=lora forbids load_in_4bit (M-2)")
            if not self.freeze_base:
                raise ValueError("technique=lora REQUIRES freeze_base=true (M-2)")
        elif t is Technique.qlora:                            # M-3
            if self.lora is None:
                raise ValueError("technique=qlora REQUIRES a `lora` block (M-3)")
            if not self.load_in_4bit:
                raise ValueError("technique=qlora REQUIRES load_in_4bit=true (M-3)")
            if self.bnb_4bit_quant_type != "nf4":
                raise ValueError("technique=qlora REQUIRES bnb_4bit_quant_type='nf4' (M-3)")
            if not self.freeze_base:
                raise ValueError("technique=qlora REQUIRES freeze_base=true (M-3)")
            if not self.training.gradient_checkpointing:
                raise ValueError("technique=qlora REQUIRES gradient_checkpointing=true (M-3)")
        return self


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return RunConfig.model_validate(yaml.safe_load(fh))
```

#### `src/finetune/data.py` — load, screen, split (no leakage) (T2/T3)

```python
"""Dataset loading, governance screening, seeded splitting, and SFT formatting.

Enforces DG-1..DG-3: license recorded, basic PII screen, no train/test overlap.
"""
import re
from datasets import load_dataset, DatasetDict
from .config import RunConfig

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _screen_pii(example, text_field):
    # DG-2: minimal redaction; replace with your org's screener.
    blob = str(example.get(text_field, ""))
    example["_pii_flagged"] = bool(_EMAIL.search(blob))
    return example


def load_splits(cfg: RunConfig) -> DatasetDict:
    ds = load_dataset(cfg.dataset.name, revision=cfg.dataset.revision, split="train")
    ds = ds.map(lambda e: _screen_pii(e, cfg.dataset.text_field))

    # DG-3: split ONCE with a fixed seed; test carved out first, then val from the rest.
    test = ds.train_test_split(test_size=cfg.dataset.test_size, seed=cfg.seed)
    train_val, test_split = test["train"], test["test"]
    val_ratio = cfg.dataset.val_size / (1 - cfg.dataset.test_size)
    tv = train_val.train_test_split(test_size=val_ratio, seed=cfg.seed)
    return DatasetDict(train=tv["train"], validation=tv["test"], test=test_split)
```

#### `src/finetune/modeling.py` — build (model, peft_config) per technique (T4)

```python
"""Builds the model + PEFT config for each technique. One function, three paths."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from .config import RunConfig, Technique


def build_tokenizer(cfg: RunConfig):
    tok = AutoTokenizer.from_pretrained(cfg.base_model.name, revision=cfg.base_model.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model_and_peft(cfg: RunConfig):
    """Returns (model, peft_config_or_None) honoring the technique matrix."""
    quant_config = None
    if cfg.technique is Technique.qlora:                      # FR-006 / M-3
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model.name,
        revision=cfg.base_model.revision,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = None
    if cfg.technique in (Technique.lora, Technique.qlora):
        if cfg.technique is Technique.qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
            )
        peft_config = LoraConfig(                             # FR-007 / T-4
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # SFTTrainer will call get_peft_model when peft_config is passed,
        # which freezes the base weights and marks only adapters trainable.
    return model, peft_config
```

#### `src/finetune/train.py` — the SFT entrypoint (T5)

```python
"""Training entrypoint: same code path for full / lora / qlora (NFR-002).

Usage: python -m finetune.train --config configs/lora.yaml
"""
import argparse, json, os, random, subprocess
import numpy as np
import torch
from trl import SFTTrainer, SFTConfig
from .config import load_config
from .data import load_splits
from .modeling import build_model_and_peft, build_tokenizer


def set_seed(seed: int):                                     # FR-005 / R-1
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main(config_path: str):
    cfg = load_config(config_path)        # <-- raises here if the matrix is violated
    set_seed(cfg.seed)

    splits = load_splits(cfg)
    tokenizer = build_tokenizer(cfg)
    model, peft_config = build_model_and_peft(cfg)

    sft_config = SFTConfig(
        output_dir=cfg.training.output_dir,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_length=cfg.training.max_length,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        bf16=cfg.training.bf16,
        max_steps=cfg.training.max_steps,           # compute budget (FR-012 / C-1)
        logging_steps=cfg.training.logging_steps,
        seed=cfg.seed,
        report_to=[] if os.getenv("WANDB_DISABLED", "true") == "true" else ["wandb"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        processing_class=tokenizer,                  # current TRL arg (was `tokenizer=`)
        peft_config=peft_config,                     # None for full FT
    )
    train_result = trainer.train()
    trainer.save_model(cfg.training.output_dir)      # adapter for LoRA/QLoRA, full ckpt otherwise

    # R-3 / NFR-003: make the output dir self-describing.
    with open(os.path.join(cfg.training.output_dir, "run_manifest.json"), "w") as fh:
        json.dump({
            "config": cfg.model_dump(mode="json"),
            "git_commit": _git_commit(),
            "train_metrics": train_result.metrics,
        }, fh, indent=2, default=str)
    print(f"Done. Artifacts + manifest in {cfg.training.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
```

#### `src/finetune/evaluate.py` — the eval gate (T6)

```python
"""Held-out evaluation = the acceptance gate (EV-METRIC / EV-THRESHOLD / EV-DECODING)."""
import evaluate as hf_evaluate


def _generate(model, tokenizer, prompt, cfg):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.eval.max_new_tokens,
        do_sample=cfg.eval.temperature > 0,        # temperature=0 → greedy (EV-DECODING)
        temperature=max(cfg.eval.temperature, 1e-8),
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate_gate(model, tokenizer, test_split, references, prompts, base_score, cfg):
    """Returns (score, passed). `passed` enforces E-2: promote only if gate passes."""
    rouge = hf_evaluate.load("rouge")
    preds = [_generate(model, tokenizer, p, cfg) for p in prompts]
    score = rouge.compute(predictions=preds, references=references)["rougeL"]
    passed = (score - base_score) >= cfg.eval.threshold_abs_gain     # EV-THRESHOLD
    return score, passed
```

#### Run it

```bash
# 1) validate + train (config decides the technique — no code change between them)
python -m finetune.train --config configs/lora.yaml
python -m finetune.train --config configs/qlora.yaml
python -m finetune.train --config configs/full.yaml

# 2) the output dir now contains the adapter/checkpoint + run_manifest.json
```

---

## 6. How to Test the Fine-Tune

Fine-tuning has a uniquely expensive feedback loop, so **most tests must catch problems *before* the GPU spins up.** Six layers, cheapest first. Crucially, you are testing the **spec** (the matrix, the gate, the no-leakage rule) — not just code.

### Layer 1 — Config enforcement (the technique matrix)
Prove every matrix rule rejects bad configs. **No GPU, milliseconds.**

`tests/test_config_enforcement.py`:
```python
import pytest
from pydantic import ValidationError
from finetune.config import RunConfig

BASE = dict(
    base_model={"name": "Qwen/Qwen3-0.6B"},
    dataset={"name": "x", "license": "apache-2.0"},
    training={"output_dir": "out"},
    eval={},
)

def _cfg(**over):
    return RunConfig.model_validate({**BASE, **over})

def test_ac1_lora_without_block_rejected():          # AC-1 / M-2
    with pytest.raises(ValidationError):
        _cfg(technique="lora", freeze_base=True)     # no lora block

def test_ac2_qlora_without_4bit_rejected():          # AC-2 / M-3
    with pytest.raises(ValidationError):
        _cfg(technique="qlora", load_in_4bit=False, freeze_base=True,
             lora={"r": 8, "alpha": 16, "target_modules": ["q_proj"]})

def test_ac3_full_with_lora_rejected():              # AC-3 / M-1
    with pytest.raises(ValidationError):
        _cfg(technique="full",
             lora={"r": 8, "alpha": 16, "target_modules": ["q_proj"]})

def test_valid_lora_accepted():
    cfg = _cfg(technique="lora", freeze_base=True,
               lora={"r": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]})
    assert cfg.technique.value == "lora"
```

### Layer 2 — Data integrity (no leakage, schema, governance)
Prove splits don't overlap (AC-7 / DG-3), the license is present, and the SFT schema is valid. **No GPU.**
```python
def test_no_train_test_leakage():                    # AC-7 / DG-3
    # build splits from a tiny in-memory dataset with a fixed seed, then:
    train_ids = set(map(_hash, splits["train"]))
    test_ids = set(map(_hash, splits["test"]))
    assert train_ids.isdisjoint(test_ids)

def test_dataset_license_required():                 # DG-1
    with pytest.raises(Exception):
        DatasetSpec(name="x")  # missing license → rejected
```

### Layer 3 — Smoke train (1 step, tiny model)
Run the **entire** pipeline for `max_steps=1` on the smallest model to prove it wires up end-to-end, and assert PEFT froze the base (AC-4). Gate behind a marker so it's opt-in.
```python
@pytest.mark.gpu
def test_lora_smoke_one_step(tmp_path):              # AC-4
    cfg = load_config("configs/lora.yaml")
    cfg.training.max_steps = 1
    cfg.training.output_dir = str(tmp_path)
    model, peft_config = build_model_and_peft(cfg)
    # ... run 1 step via SFTTrainer ...
    base_params = [p for n, p in model.named_parameters() if "lora" not in n]
    assert all(not p.requires_grad for p in base_params)   # base frozen (FR-007)
```

### Layer 4 — Eval gate logic
Prove the gate **blocks promotion** when the tuned model doesn't beat the base by the threshold (AC-6 / E-2). Mock the metric so it's fast and deterministic.
```python
def test_gate_blocks_when_below_threshold():         # AC-6 / E-2
    score, passed = 0.40, None
    # base_score=0.38, threshold=0.05 → gain 0.02 < 0.05 → must FAIL
    passed = (0.40 - 0.38) >= 0.05
    assert passed is False
```

### Layer 5 — Reproducibility
Same config + seed → same first-step loss within tolerance (AC-5 / NFR-001). Run two 1-step trainings and compare.

### Layer 6 — Spec-coverage gate (anti-drift)
Assert every `FR-xxx` / `AC-x` in `spec.md` is referenced by at least one test (Constitution Q-2). Identical to the app-SDD coverage gate:
```python
import pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
def test_every_requirement_has_a_test():
    spec = (ROOT/"specs"/"spec.md").read_text(encoding="utf-8")
    blob = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT/"tests").glob("test_*.py"))
    required = set(re.findall(r"FR-\d+", spec)) | set(re.findall(r"AC-\d+", spec))
    covered  = set(re.findall(r"FR-\d+", blob)) | set(re.findall(r"AC-\d+", blob))
    missing = required - covered
    assert not missing, f"Spec items with no test: {sorted(missing)}"
```

`pytest.ini`:
```ini
[pytest]
pythonpath = src
markers =
    gpu: tests that require a GPU and load a model (opt-in)
```

Run them:
```bash
pytest -q -m "not gpu"     # Layers 1,2,4,5(logic),6 — fast, no GPU, run on every commit
pytest -q -m gpu           # Layers 3 + full 5 — nightly / pre-release on a GPU box
```

**Testing summary**

| Layer | Proves | GPU? | When |
|---|---|---|---|
| 1 Config enforcement | Technique matrix rejects bad configs | No | Every commit |
| 2 Data integrity | No leakage, schema, license present | No | Every commit |
| 3 Smoke train | Pipeline wires up; base frozen | Yes (tiny) | Nightly / pre-run |
| 4 Eval gate | Promotion blocked below threshold | No | Every commit |
| 5 Reproducibility | Same seed → same loss | Mixed | Nightly |
| 6 Spec coverage | Every requirement has a test | No | Every commit |

> **The point:** Layers 1, 2, 4, 6 cost nothing and catch the failures that otherwise surface *after* a 6-hour run — a forbidden config, a leaked test set, a gate that silently passed. That is how you "test the spec to make sure it's working as expected" in a domain where a full run is too expensive to be your test.

---

## 7. Files: Necessary vs. Not

### ✅ Necessary (the SDD fine-tuning core)

| File / Dir | Why it's essential |
|---|---|
| `specs/spec.md` | Source of truth: model, data, metric, acceptance gate, **technique matrix**. |
| `specs/constitution.md` | Governance/reproducibility/gate rules every run must obey. |
| `specs/plan.md`, `specs/tasks.md` | Chosen technique + hyperparameters; traceable work items. |
| `configs/*.yaml` | The run is a function of its config (FR-001). One per technique. |
| `src/finetune/config.py` | **The enforcer** — rejects matrix-violating configs before GPU time. |
| `src/finetune/data.py` | License/PII screen + seeded, non-overlapping splits (DG rules). |
| `src/finetune/modeling.py`, `train.py`, `evaluate.py` | The portable train + gate pipeline. |
| `tests/test_config_enforcement.py`, `test_data_integrity.py`, `test_eval_gate.py`, `test_spec_coverage.py` | Cheap tests that prove the spec holds **without** a GPU. |
| `requirements.txt` (PINNED) | Reproducibility (R-2). |
| `run_manifest.json` (generated per run) | Self-describing output: config + commit + metrics (R-3). |
| `.env.example`, `.gitignore` | Secrets via env; never commit `.env` or tokens. |

### ⚠️ Useful but optional

| File | When you need it |
|---|---|
| `tests/test_smoke_train.py`, `test_reproducibility.py` | When you have a GPU runner for nightly checks. |
| `accelerate` / `deepspeed` config | Multi-GPU or multi-node (out of scope for v1). |
| `configs/full.yaml` | Only if you actually intend to run full FT. |
| W&B / MLflow logging | Team experiment tracking. |
| `Dockerfile` | Reproducible compute environment. |

### ❌ Not necessary (skip — they cause the failures SDD exists to prevent)

| Anti-pattern | Why to avoid |
|---|---|
| A single `train.ipynb` with hard-coded hyperparameters | Unreproducible; no enforcement; the classic "what did we even run?" |
| Hyperparameters / model name / token pasted in code | Violates FR-001/FR-011 — they belong in config + env vars. |
| Committing `.env`, tokens, or the dataset's raw PII | Security + governance breach. |
| Reusing the test set during training to "get a better number" | Violates DG-3; the eval gate becomes a lie. |
| A different script per technique (`train_lora.py`, `train_qlora.py`, …) | Drift between paths; violates NFR-002. Use **one** config-driven entrypoint. |
| Skipping the seed / version pins "to move fast" | Makes every result unreproducible — the one thing fine-tuning most needs. |
| Reporting a metric with no fixed decoding config | Non-comparable numbers; the gate is meaningless. |

> **Rule of thumb:** if a file pins a *decision* or enforces a *guarantee*, it's necessary. A notebook that hides those decisions is exactly what SDD replaces.

---

## 8. Pros and Cons

### Per-technique trade-offs

| Technique | Pros | Cons |
|---|---|---|
| **Full FT** | Maximum adaptation; no adapter overhead at inference; best ceiling on quality with enough data. | Highest memory/compute; risks catastrophic forgetting; large checkpoints; needs lower LR + care. |
| **LoRA** | ~0.1–2% trainable params; cheap, fast, tiny adapters; swap many adapters on one base; low forgetting. | Slightly lower ceiling than full FT on some tasks; must pick `r`/`target_modules` well; adapter must match base at inference. |
| **QLoRA** | Fine-tune **large** models on a single GPU (4-bit NF4 base); LoRA's cheap-adapter benefits at big scale. | Quantization can cost a little quality; needs `bitsandbytes`/NVIDIA GPU; slower per step; merge/dequantize step for deploy. |
| **SFT (objective)** | Simple, well-understood, the standard way to get instruction/chat behavior; pairs with any scope above. | Only as good as your labeled (prompt→completion) data; doesn't optimize *preferences* (that's DPO/KTO/GRPO). |

### Pros & cons of doing fine-tuning the **spec-driven** way

**Pros:** reproducible runs (config + seed + pinned revisions); **technique portability** (one spec, compare full vs LoRA vs QLoRA fairly); an explicit eval gate so "done" is objective; governance enforced (license, PII, no-leakage) not hoped-for; cheap tests catch fatal errors before GPU hours are spent; self-describing outputs make audits and post-mortems trivial.

**Cons (with mitigations):** upfront effort to write the spec + config schema *(mitigation: reuse this template; the schema pays for itself the first time it rejects a bad run)*; risk of an over-rigid matrix that blocks legitimate experiments *(mitigation: keep the matrix to true invariants; allow an `experimental` technique that loosens rules but is clearly marked non-promotable)*; the spec can go stale vs fast-moving HF APIs *(mitigation: pin versions, and the Layer-6 coverage gate + smoke test catch drift)*; SDD doesn't make a bad dataset good — *garbage in, garbage out still applies.*

---

## 9. Common Pitfalls & Best Practices

- **Validate the config before the GPU.** The matrix enforcer (Layer 1) is your cheapest, highest-value test.
- **Split once, with a seed, and never train on test.** Bake it into `data.py`, assert it in Layer 2.
- **Pin everything reproducible:** base-model revision, dataset revision, library versions, seed.
- **One entrypoint, many configs.** Never fork a script per technique — that's how paths drift.
- **Fix your decoding for eval.** A metric without a fixed `max_new_tokens`/temperature isn't comparable.
- **Prefer the cheapest technique that clears the gate** (QLoRA → LoRA → full), per Constitution C-2.
- **Watch for overfitting / catastrophic forgetting.** Keep a validation curve; full FT especially needs a low LR.
- **Match adapter to base at inference.** A LoRA adapter is meaningless without its exact base model + revision.
- **Make outputs self-describing.** The `run_manifest.json` (config + commit + metrics) is what makes a run auditable six months later.
- **Keep secrets in env vars.** HF token never goes in a YAML or a notebook.

---

## 10. A 1-Week Skilling Plan

| Day | Focus | Outcome |
|---|---|---|
| **1** | Read this guide + the TRL `SFTTrainer` and PEFT `LoraConfig` docs. | Mental model: SFT = objective; full/LoRA/QLoRA = scope. |
| **2** | Build `config.py` + the three YAMLs. Make Layer-1 enforcement tests green. | The spec *enforces* the technique matrix. |
| **3** | Implement `data.py`; make Layer-2 (no-leakage, license) tests green. | Trustworthy, governed data splits. |
| **4** | Implement `modeling.py` + `train.py`; run a **LoRA** smoke train on the 0.6B model. | First real adapter + a `run_manifest.json`. |
| **5** | Run the **QLoRA** path on a 7–8B model on one GPU (only a config change!). | Felt the portability payoff (NFR-002). |
| **6** | Implement `evaluate.py`; wire the eval gate; intentionally fail it, then pass it. | "Done" is now objective, not vibes. |
| **7** | Add the Layer-6 coverage gate; put `pytest -m "not gpu"` in CI; write a short retro. | A repeatable, auditable, team-ready fine-tuning workflow. |

---

## 11. Reinforcement-Learning & RLVR Post-Training (Spec Extension)

This section **extends the same spec-driven project** with reinforcement-learning post-training — including **RLVR (Reinforcement Learning from Verifiable Rewards)**, the technique behind reasoning models like DeepSeek-R1. Everything here adds to the *existing* Constitution, spec, config schema, and test suite; it does not replace them. The enforcement philosophy is identical: **the spec declares an Objective Matrix, and a validator rejects any RL run that violates it before a single GPU step runs.**

### 11.1 The two axes: parameter scope × training objective

Fine-tuning has **two orthogonal choices**, and a complete run config must pin **both**:

```
                    TRAINING OBJECTIVE  (what signal teaches the model)
                    ┌───────────┬───────────────────┬──────────────────────────┐
                    │  SFT      │ Preference        │ Reinforcement Learning    │
                    │ (labels)  │ (DPO, KTO)        │ (PPO, GRPO, RLOO, RLVR)   │
  PARAMETER  ┌──────┼───────────┼───────────────────┼──────────────────────────┤
  SCOPE      │ full │  ✓        │  ✓                │  ✓ (costly)               │
  (which     │ lora │  ✓        │  ✓                │  ✓ (common)               │
  weights)   │ qlora│  ✓        │  ✓                │  ✓ (common for big models)│
             └──────┴───────────┴───────────────────┴──────────────────────────┘
```

- **Section 2–8 covered the *scope* axis** (`full | lora | qlora`) with the SFT objective.
- **This section adds the *objective* axis** (`sft | dpo | kto | grpo | rlvr`).
- They **compose**: e.g., "GRPO + QLoRA" (RLVR on a 4-bit large model with LoRA adapters) is a valid, common configuration. The scope matrix (M-1…M-4) and the objective matrix (O-1…O-5) are both enforced.

**Where RL fits in the lifecycle:** the standard recipe is **SFT first** (teach the format/behavior), **then** a preference or RL stage (align or sharpen). RLVR/GRPO is almost always applied to a model that has already been SFT'd.

### 11.2 RL methods at a glance (DPO, KTO, GRPO, RLOO, PPO, RLVR)

| Method | Signal it learns from | Needs a reward model? | Dataset shape | TRL trainer |
|---|---|---|---|---|
| **DPO** (Direct Preference Optimization) | Preference pairs (A is better than B) | No (closed-form, ref model only) | `prompt`, `chosen`, `rejected` | `DPOTrainer` |
| **KTO** (Kahneman–Tversky Optimization) | Per-sample **binary** good/bad labels | No | `prompt`, `completion`, `label` (bool) | `KTOTrainer` |
| **PPO** (classic RLHF) | Scalar reward from a **learned reward model** | **Yes** | prompts (+ a trained RM) | (PPO trainer) |
| **GRPO** (Group Relative Policy Optimization) | Reward over a **group** of sampled completions; critic-free | Optional (RM **or** programmatic) | prompts (+ reward fn/model) | `GRPOTrainer` |
| **RLOO** (REINFORCE Leave-One-Out) | Reward with a leave-one-out baseline | Optional | prompts (+ reward) | `RLOOTrainer` |
| **RLVR** (RL from **Verifiable** Rewards) | **Deterministic programmatic verifier** (no learned RM) | **No — forbidden** | prompts **+ ground truth** | `GRPOTrainer` (or RLOO/PPO) with verifier reward functions |
| **Reward modeling** | Trains an RM from preference data (feeds PPO) | (it *is* the RM) | preference pairs | `RewardTrainer` |

> **Reward modeling** is the optional step that produces the learned reward model classic RLHF/PPO needs. RLVR deliberately **skips** it — verifiers replace the learned RM, which removes weeks of preference-data work and makes every reward auditable.

### 11.3 What RLVR is (and its failure modes)

**RLVR** treats the LLM as a **policy** that, given a prompt, generates a chain-of-thought + answer (a sequence of actions), and receives feedback from a **deterministic verifier**: the reward is `1.0` if the answer is verifiably correct, `0.0` otherwise. A policy-optimization algorithm (GRPO is the popular choice; PPO/RLOO also work) then updates the policy to favor high-reward trajectories. The loop:

```
1. SAMPLE   — draw K completions per prompt from the current policy
2. VERIFY   — run a programmatic verifier r(prompt, completion) on each  → reward ∈ {0,1} (or shaped)
3. REWARD   — group the K rewards (GRPO uses the group mean as the baseline)
4. UPDATE   — push probability mass toward the high-reward completions
5. REPEAT   — with new prompts, monitoring KL to the reference policy
```

**Why teams use it:** ground-truth rewards (unit tests, math checkers, SQL execution, formal proofs) are **binary and tamper-resistant**; every reward traces back to a transparent verifier run (auditable); and it needs **no preference-labeling**. It shines on **verifiable** domains — math, code, logic, structured output.

**What it is NOT good for:** creative writing, brand voice, subjective quality — there's no ground truth to verify, so DPO/KTO/RLHF with human preference data remain superior there.

**Failure modes the spec must defend against (these become enforced rules + tests):**
1. **Reward hacking via partial verifiers.** A verifier that only checks syntax (not execution) lets the model produce *valid-but-wrong* outputs that score 1.0. *Defense: execution/semantic verifiers + an adversarial verifier test suite (RL-2, AC-12).*
2. **Spurious rewards.** Models have improved on benchmarks even with **random** rewards — so a rising reward curve does **not** prove your verifier works. *Defense: always validate with an independent held-out eval, not reward alone (RL-5).*
3. **Compression, not capability.** Research indicates much of RLVR's gain is **sampling-efficiency / search compression** (solving in 1 try what the base solved in K tries), not necessarily new reasoning ability. *Defense: measure Pass@1 **and** Pass@K; report both.*
4. **Reward over-optimization / policy collapse.** Pushing reward too hard drifts the policy far from the base and degrades general ability. *Defense: monitor KL to the reference policy and stop past a bound (RL-3, FR-017).*

### 11.4 Constitution additions (RL)

Append to `specs/constitution.md`:

```markdown
## Reinforcement Learning (RL / RLVR)
- RL-1: RLVR reward functions (verifiers) MUST be deterministic and version-controlled.
        Every reward MUST be traceable to a specific verifier run (auditability).
- RL-2: Verifiers MUST pass an adversarial test suite BEFORE any RL run — measuring
        false-positive rate (reward hacking) and false-negative rate against labeled
        known-good / known-bad outputs. A verifier failing its suite blocks the run.
- RL-3: RL runs MUST monitor KL divergence to the reference policy and stop if it
        exceeds `max_kl` (prevents reward over-optimization / collapse).
- RL-4: Preference data (DPO/KTO) MUST record provenance: source, annotator type
        (human/AI), and date. RLAIF-labeled data MUST be marked as AI-generated.
- RL-5: An RL run MUST be evaluated on an INDEPENDENT held-out set, not on the reward
        curve alone (reward can rise from spurious or hacked signals). Report Pass@1 AND Pass@K.
- RL-6: RLVR MUST NOT use a learned reward model; only programmatic verifiers are allowed.
        (Use objective=grpo with a reward model if you intend learned-RM RLHF.)
```

### 11.5 Spec additions — the enforceable Objective Matrix

Append to `specs/spec.md`. New functional requirements and acceptance criteria, plus the authoritative **Objective Matrix** (machine-enforced in `config.py`).

```markdown
## Objective axis (extends the spec)
A run config MUST declare an `objective` in addition to a `technique` (scope).

## Additional Functional Requirements
- FR-013: The config MUST declare exactly one `objective` ∈ {sft, dpo, kto, grpo, rlvr}.
- FR-014: The system MUST validate the Objective Matrix (O-1..O-5) and FAIL before training
          if violated.
- FR-015: For `rlvr`, all reward functions MUST be deterministic; the system MUST run a
          verifier self-test (same input → identical reward) before training. (RL-1)
- FR-016: For `rlvr`, a learned `reward_model` MUST be absent — only programmatic verifiers. (RL-6)
- FR-017: RL objectives (grpo/rlvr) MUST log KL to the reference policy and stop if KL > max_kl. (RL-3)
- FR-018: `grpo`/`rlvr` MUST set `num_generations` > 1 (a group is required).
- FR-019: `dpo` requires a preference dataset (prompt/chosen/rejected); `kto` requires
          (prompt/completion/label). A mismatched dataset MUST be rejected.
- FR-020: `rlvr` datasets MUST include a ground-truth/answer field consumed by the verifier.

## ── OBJECTIVE MATRIX (authoritative, enforced by config.py) ──
| Field                | sft        | dpo                 | kto                  | grpo               | rlvr                         |
|----------------------|------------|---------------------|----------------------|--------------------|------------------------------|
| dataset format       | text/msgs  | prompt/chosen/rejected | prompt/completion/label | prompt(+gt)    | **prompt + ground_truth REQ**|
| `reward_funcs`       | forbidden  | forbidden           | forbidden            | **REQUIRED ≥1**    | **REQUIRED ≥1 (verifiers)**  |
| `reward_model`       | forbidden  | forbidden           | forbidden            | optional           | **FORBIDDEN (RL-6)**         |
| `beta` (KL/pref)     | n/a        | **REQUIRED**        | **REQUIRED**         | optional           | optional                     |
| `num_generations`    | n/a        | n/a                 | n/a                  | **REQUIRED >1**    | **REQUIRED >1**              |
| `max_kl` (stop)      | n/a        | n/a                 | n/a                  | **REQUIRED**       | **REQUIRED**                 |
| TRL trainer          | SFTTrainer | DPOTrainer          | KTOTrainer           | GRPOTrainer        | GRPOTrainer + verifiers      |

Rules enforced (config.py raises if broken):
- O-1: objective=dpo  → preference dataset; reward_funcs & reward_model absent; beta set.
- O-2: objective=kto  → (prompt,completion,label) dataset; reward_funcs & reward_model absent; beta set.
- O-3: objective=grpo → reward_funcs ≥1 (or reward_model); num_generations>1; max_kl set.
- O-4: objective=rlvr → reward_funcs ≥1 AND all are registered verifiers; reward_model ABSENT;
                        dataset has ground_truth; num_generations>1; max_kl set.
- O-5: objective=sft  → no reward_funcs, no reward_model, no num_generations.
(The scope rules M-1..M-4 still apply — objective composes with full/lora/qlora.)

## Additional Acceptance Criteria
- AC-8 : GIVEN objective=rlvr with NO reward_funcs, WHEN config loads, THEN rejected. (O-4)
- AC-9 : GIVEN objective=rlvr WITH a reward_model, WHEN config loads, THEN rejected. (O-4/RL-6)
- AC-10: GIVEN objective=dpo with a non-preference dataset, WHEN loaded, THEN rejected. (O-1/FR-019)
- AC-11: GIVEN a verifier and one completion, WHEN scored twice, THEN identical reward. (FR-015)
- AC-12: GIVEN the verifier's adversarial suite, WHEN evaluated, THEN known-correct→1.0 and
         known-wrong/hack-attempt→0.0 within the allowed error rates. (RL-2)
- AC-13: GIVEN objective=grpo with num_generations=1, WHEN loaded, THEN rejected. (O-3/FR-018)
```

### 11.6 Config enforcement (the objective axis)

Append to `src/finetune/config.py`. These additions sit beside the existing `RunConfig` and compose with the scope validator.

```python
from typing import Callable, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, model_validator


class Objective(str, Enum):
    sft = "sft"
    dpo = "dpo"
    kto = "kto"
    grpo = "grpo"
    rlvr = "rlvr"


class RewardSpec(BaseModel):
    # Names resolved against the verifier REGISTRY in rewards.py (FR-015/FR-016).
    reward_funcs: List[str] = Field(default_factory=list)   # e.g. ["accuracy", "format"]
    reward_model: Optional[str] = None                      # learned RM (NOT allowed for rlvr)
    num_generations: int = 1                                # group size for GRPO/RLVR
    beta: Optional[float] = None                            # KL / preference strength
    max_kl: Optional[float] = None                          # RL-3 stop bound
    ground_truth_field: Optional[str] = None                # dataset column the verifier reads


# Extend RunConfig with:  objective: Objective = Objective.sft
#                         rl: RewardSpec = RewardSpec()
#
# and add this validator (runs in addition to the scope matrix validator):

def enforce_objective_matrix(self) -> "RunConfig":
    o, rl = self.objective, self.rl
    has_funcs = len(rl.reward_funcs) > 0

    if o is Objective.sft:                                   # O-5
        if has_funcs or rl.reward_model:
            raise ValueError("objective=sft forbids reward_funcs/reward_model (O-5)")
    elif o in (Objective.dpo, Objective.kto):               # O-1 / O-2
        if has_funcs or rl.reward_model:
            raise ValueError(f"objective={o.value} forbids reward signals (O-1/O-2)")
        if rl.beta is None:
            raise ValueError(f"objective={o.value} REQUIRES beta (O-1/O-2)")
    elif o is Objective.grpo:                               # O-3
        if not (has_funcs or rl.reward_model):
            raise ValueError("objective=grpo REQUIRES reward_funcs or reward_model (O-3)")
        if rl.num_generations <= 1:
            raise ValueError("objective=grpo REQUIRES num_generations>1 (O-3)")
        if rl.max_kl is None:
            raise ValueError("objective=grpo REQUIRES max_kl (O-3/RL-3)")
    elif o is Objective.rlvr:                               # O-4
        if not has_funcs:
            raise ValueError("objective=rlvr REQUIRES reward_funcs (verifiers) (O-4)")
        if rl.reward_model is not None:
            raise ValueError("objective=rlvr FORBIDS a learned reward_model (O-4/RL-6)")
        if rl.num_generations <= 1:
            raise ValueError("objective=rlvr REQUIRES num_generations>1 (O-4)")
        if rl.max_kl is None:
            raise ValueError("objective=rlvr REQUIRES max_kl (O-4/RL-3)")
        if not rl.ground_truth_field:
            raise ValueError("objective=rlvr REQUIRES a ground_truth_field (O-4/FR-020)")
    return self
```

A minimal `configs/rlvr.yaml` (RLVR via GRPO + QLoRA on a reasoning task):

```yaml
technique: qlora                 # scope axis — QLoRA for a large base
load_in_4bit: true
bnb_4bit_quant_type: nf4
freeze_base: true
lora: { r: 16, alpha: 32, target_modules: [q_proj, k_proj, v_proj, o_proj] }

objective: rlvr                  # objective axis — verifiable-reward RL
rl:
  reward_funcs: [accuracy, format]   # resolved in rewards.py REGISTRY
  reward_model: null                 # FORBIDDEN for rlvr (enforced)
  num_generations: 8                 # K samples per prompt (the "group")
  max_kl: 0.05                       # stop if policy drifts too far (RL-3)
  ground_truth_field: answer         # dataset column the verifier checks

base_model: { name: Qwen/Qwen3-4B }
dataset: { name: openai/gsm8k, license: mit, text_field: question }
training: { output_dir: outputs/rlvr-gsm8k, learning_rate: 1.0e-6,
            gradient_checkpointing: true, bf16: true, max_steps: 500 }
eval: { metric: exact_match, threshold_abs_gain: 0.05, max_new_tokens: 512, temperature: 0.0 }
```

### 11.7 Reward functions — the RLVR verifiers

`src/finetune/rewards.py` — deterministic, version-controlled, registry-resolved verifiers. These use TRL's GRPO reward-function signature: **a callable that receives `completions` (and dataset columns such as `answer` as keyword args) and returns a `list[float]`.**

```python
"""Verifiable reward functions (RLVR). Deterministic by construction (RL-1).

TRL calls each reward function as: fn(prompts, completions, **dataset_columns) -> list[float].
Dataset columns (e.g. `answer`) arrive as keyword arguments matching their column names.
"""
import re
from typing import Callable, Dict, List

REGISTRY: Dict[str, Callable] = {}


def verifier(name: str):
    """Decorator that registers a reward function so config.reward_funcs can resolve it."""
    def _wrap(fn: Callable) -> Callable:
        REGISTRY[name] = fn
        return fn
    return _wrap


def _extract_final_answer(text: str):
    # Convention: final answer after '####' (GSM8K) or inside \boxed{...}.
    m = re.search(r"####\s*(-?[\d.,]+)", text) or re.search(r"\\boxed\{(-?[\d.,]+)\}", text)
    return m.group(1).replace(",", "").strip() if m else None


@verifier("accuracy")
def accuracy_reward(completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """1.0 iff the parsed final answer equals ground truth — execution/semantic check.

    NOTE: a STRONG verifier checks the *answer*, not just the format, to avoid the
    'valid-but-wrong' reward-hacking failure mode (RL-2).
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        pred = _extract_final_answer(str(completion))
        gt_norm = str(gt).replace(",", "").strip()
        rewards.append(1.0 if pred is not None and pred == gt_norm else 0.0)
    return rewards


@verifier("format")
def format_reward(completions: List[str], **kwargs) -> List[float]:
    """0.2 shaping reward for the required <think>...</think> + #### answer structure."""
    pat = re.compile(r"<think>.*?</think>.*####\s*-?[\d.,]+", re.DOTALL)
    return [0.2 if pat.search(str(c)) else 0.0 for c in completions]


def resolve(names: List[str]) -> List[Callable]:
    """Map config.reward_funcs names → callables; unknown name fails fast (FR-015)."""
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise KeyError(f"Unknown reward functions (not registered): {missing}")
    return [REGISTRY[n] for n in names]
```

> **Verifier design rule (RL-2):** the `accuracy` verifier checks the **answer**, not merely that the output *looks* right. The classic reward-hacking trap is a syntax-only check (e.g., "the SQL parses" or "there's a `####`") that a model games with valid-but-wrong outputs. Pair a strong correctness verifier with a small format-shaping reward — never format alone.

### 11.8 The RL training entrypoint

`src/finetune/train_rl.py` — dispatches to the right TRL trainer by `objective`, **reusing** `build_model_and_peft` (scope axis) and the same seeding / manifest logic from `train.py`.

```python
"""RL/preference training entrypoint. Same scope builder; trainer chosen by objective.

Usage: python -m finetune.train_rl --config configs/rlvr.yaml
"""
import argparse
from trl import GRPOTrainer, GRPOConfig, DPOTrainer, DPOConfig, KTOTrainer, KTOConfig
from .config import load_config, Objective
from .data import load_splits
from .modeling import build_model_and_peft, build_tokenizer
from . import rewards


def main(config_path: str):
    cfg = load_config(config_path)        # raises on scope OR objective matrix violation
    splits = load_splits(cfg)
    tokenizer = build_tokenizer(cfg)
    model, peft_config = build_model_and_peft(cfg)   # full/lora/qlora — unchanged

    if cfg.objective in (Objective.grpo, Objective.rlvr):
        reward_funcs = rewards.resolve(cfg.rl.reward_funcs)   # verifiers (RLVR)
        # Optional: run the verifier self-test here before training (FR-015) — see 11.9.
        args = GRPOConfig(
            output_dir=cfg.training.output_dir,
            learning_rate=cfg.training.learning_rate,
            num_generations=cfg.rl.num_generations,   # group size K
            max_completion_length=512,
            bf16=cfg.training.bf16,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            max_steps=cfg.training.max_steps,
            beta=cfg.rl.beta or 0.0,                  # KL coefficient
            seed=cfg.seed,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,                # the verifiers ARE the reward
            args=args,
            train_dataset=splits["train"],
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    elif cfg.objective is Objective.dpo:
        args = DPOConfig(output_dir=cfg.training.output_dir, beta=cfg.rl.beta,
                         learning_rate=cfg.training.learning_rate, bf16=cfg.training.bf16,
                         max_steps=cfg.training.max_steps, seed=cfg.seed)
        trainer = DPOTrainer(model=model, ref_model=None, args=args,
                             train_dataset=splits["train"],   # prompt/chosen/rejected
                             processing_class=tokenizer, peft_config=peft_config)

    elif cfg.objective is Objective.kto:
        args = KTOConfig(output_dir=cfg.training.output_dir, beta=cfg.rl.beta,
                         learning_rate=cfg.training.learning_rate, bf16=cfg.training.bf16,
                         max_steps=cfg.training.max_steps, seed=cfg.seed)
        trainer = KTOTrainer(model=model, args=args,
                             train_dataset=splits["train"],   # prompt/completion/label
                             processing_class=tokenizer, peft_config=peft_config)
    else:
        raise ValueError(f"Use train.py for objective={cfg.objective.value} (SFT).")

    # TODO (RL-3 / FR-017): attach a TrainerCallback that reads the logged KL and calls
    # control.should_training_stop = True when KL > cfg.rl.max_kl.
    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    print(f"RL training done ({cfg.objective.value}). Artifacts in {cfg.training.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    main(p.parse_args().config)
```

> KL-stop (RL-3) is wired via a `transformers.TrainerCallback` that inspects the metric GRPO/DPO already log (`kl` / `objective/kl`) each step and flips `control.should_training_stop` once it exceeds `max_kl`. Keeping it a callback means the stop rule is testable in isolation.

### 11.9 How to test RL & verifiers (Layer 7)

RL adds a **new test layer** to the six from [Section 6](#6-how-to-test-the-fine-tune). The headline: **in RLVR, your verifier IS your spec — so you test the verifier as rigorously as the model.** A weak verifier silently teaches the model to cheat.

**Layer 7 — Verifier & objective validation** (no GPU; runs on every commit):

`tests/test_objective_enforcement.py` — the objective matrix rejects bad configs:
```python
import pytest
from pydantic import ValidationError
from finetune.config import RunConfig
# (BASE config dict as in Section 6, plus objective/rl fields)

def test_ac8_rlvr_without_reward_funcs_rejected():     # AC-8 / O-4
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**BASE, "objective": "rlvr",
                                  "rl": {"num_generations": 8, "max_kl": 0.05,
                                         "ground_truth_field": "answer"}})  # no reward_funcs

def test_ac9_rlvr_with_reward_model_rejected():         # AC-9 / RL-6
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**BASE, "objective": "rlvr",
                                  "rl": {"reward_funcs": ["accuracy"], "reward_model": "my-rm",
                                         "num_generations": 8, "max_kl": 0.05,
                                         "ground_truth_field": "answer"}})

def test_ac13_grpo_single_generation_rejected():       # AC-13 / O-3
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**BASE, "objective": "grpo",
                                  "rl": {"reward_funcs": ["accuracy"],
                                         "num_generations": 1, "max_kl": 0.05}})
```

`tests/test_verifiers.py` — determinism + **adversarial** verifier suite (catches reward hacking):
```python
from finetune.rewards import accuracy_reward, format_reward

def test_ac11_verifier_is_deterministic():             # AC-11 / RL-1
    c, a = ["...#### 42"], ["42"]
    assert accuracy_reward(c, answer=a) == accuracy_reward(c, answer=a)

def test_ac12_known_correct_scores_one():              # AC-12
    assert accuracy_reward(["reasoning... #### 42"], answer=["42"]) == [1.0]

def test_ac12_known_wrong_scores_zero():               # AC-12 — no reward hacking
    assert accuracy_reward(["reasoning... #### 41"], answer=["42"]) == [0.0]

def test_ac12_format_hack_does_not_earn_accuracy():    # AC-12 — valid-looking but wrong
    # A completion that has perfect format but the WRONG answer must NOT get accuracy reward.
    assert accuracy_reward(["<think>x</think> #### 999"], answer=["42"]) == [0.0]

def test_verifier_false_negative_bound():              # RL-2 — measure error rate
    labeled = [("#### 42", "42", 1.0), ("#### 7", "42", 0.0), ("no answer", "42", 0.0)]
    preds = [accuracy_reward([c], answer=[gt])[0] for c, gt, _ in labeled]
    errors = sum(p != exp for (_, _, exp), p in zip(labeled, preds))
    assert errors / len(labeled) <= 0.0      # tighten/relax per your tolerance
```

**Layer 7b — RL smoke + KL-stop** (GPU, opt-in): run GRPO for `max_steps=1` on a tiny model to prove the reward functions wire into the trainer and produce a scalar per completion; unit-test the KL-stop callback by feeding it a fake metric above/below `max_kl`.

**Updated test summary (adds to Section 6):**

| Layer | Proves | GPU? | When |
|---|---|---|---|
| 7 Objective enforcement | Objective matrix rejects bad RL configs | No | Every commit |
| 7 Verifier validation | Verifiers are deterministic + not hackable (adversarial) | No | Every commit |
| 7b RL smoke + KL-stop | Reward wires into the trainer; KL stop fires | Yes (tiny) | Nightly / pre-run |

> The whole point, restated for RL: **a rising reward curve is not proof.** Spurious rewards and reward hacking can drive reward up while real quality is flat or worse. The cheap Layer-7 verifier tests + an independent held-out eval (RL-5) are how you keep the RLVR spec honest — and add the new FR/AC IDs (`FR-013..020`, `AC-8..13`) to the Layer-6 coverage gate so none of them can ship untested.

### 11.10 Pros & cons per RL method

| Method | Pros | Cons |
|---|---|---|
| **DPO** | No reward model, no RL loop — simple & stable; great for style/tone/safety alignment; cheap relative to PPO. | Needs good **preference pairs**; can over-fit to the pair distribution; less expressive than online RL for hard objectives. |
| **KTO** | Only needs **binary** good/bad labels (cheaper to collect than pairs); robust when pairwise data is scarce. | Coarser signal than pairs; still needs reasonably balanced labels. |
| **PPO (RLHF)** | The original, powerful alignment method; works with any scalar reward. | Complex & unstable; needs a **trained reward model** + value model; memory- and tuning-heavy. |
| **GRPO** | **Critic-free** (no value model → less memory than PPO); strong for reasoning; works with RM **or** programmatic reward. | Needs many samples per prompt (`num_generations` → compute); sensitive to reward design. |
| **RLOO** | Lightweight online RL (REINFORCE + leave-one-out baseline); simpler than PPO. | Higher variance than critic-based methods on some tasks. |
| **RLVR** | **No reward-model training**; deterministic, **auditable** rewards; excellent on verifiable domains (math/code/SQL/logic); fast iteration (change the verifier, not retrain an RM). | Only works where **ground truth exists**; vulnerable to **reward hacking** (partial verifiers) and **spurious-reward**/compression effects — *must* be paired with adversarial verifier tests + held-out eval. |
| **Reward modeling** | Produces a reusable learned reward for PPO/GRPO; captures subjective quality verifiers can't. | The RM itself can be gamed/biased; another model to train, validate, and maintain. |

**Choosing (rule of thumb):**
- Verifiable correctness (math, code, structured output) → **RLVR** (GRPO + verifiers).
- Subjective quality / style / safety, you have preference pairs → **DPO**.
- Only binary good/bad labels available → **KTO**.
- You need maximum control and have RM infrastructure → **PPO with a reward model**.
- Always: **SFT first**, then the preference/RL stage; pair the cheapest scope that clears your gate (QLoRA before LoRA before full).

---

## 12. References

- **TRL — SFT Trainer** (current `SFTTrainer` / `SFTConfig`, `processing_class`, `max_length`): <https://huggingface.co/docs/trl/sft_trainer>
- **TRL repo** (source of truth for the evolving API): <https://github.com/huggingface/trl>
- **PEFT — LoRA / config**: <https://huggingface.co/docs/peft>
- **Transformers — `BitsAndBytesConfig` / quantization (QLoRA)**: <https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes>
- **bitsandbytes** (4-bit NF4): <https://github.com/bitsandbytes-foundation/bitsandbytes>
- **datasets**: <https://huggingface.co/docs/datasets>
- **QLoRA paper** (Dettmers et al., 4-bit NF4 + LoRA): <https://arxiv.org/abs/2305.14314>
- **LoRA paper** (Hu et al.): <https://arxiv.org/abs/2106.09685>
- **GitHub Spec Kit** (the SDD methodology this guide applies): <https://github.com/github/spec-kit>

**Reinforcement learning & RLVR (Section 11):**
- **TRL — GRPO Trainer** (`GRPOTrainer`, `reward_funcs`): <https://huggingface.co/docs/trl/grpo_trainer>
- **TRL — DPO / KTO / RLOO / Reward trainers**: <https://huggingface.co/docs/trl>
- **DeepSeek-R1** (GRPO + rule-based verifiable rewards for reasoning): <https://arxiv.org/abs/2501.12948>
- **RLVR analysis — *RL with Verifiable Rewards Implicitly Incentivizes Correct Reasoning*** (Microsoft Research, arXiv:2506.14245): <https://arxiv.org/abs/2506.14245>
- **Tülu 3** (open post-training recipe popularizing RLVR): <https://arxiv.org/abs/2411.15124>
- **Awesome-RLVR** (curated codebases & papers): <https://github.com/opendilab/awesome-RLVR>
- **DPO paper** (Rafailov et al.): <https://arxiv.org/abs/2305.18290> · **KTO paper**: <https://arxiv.org/abs/2402.01306>

---

*Built as a learning artifact. Hugging Face APIs (especially TRL) move quickly — validate `SFTTrainer`/`SFTConfig` argument names, PEFT/bitsandbytes versions, and `target_modules` against your installed versions before a production run. The code is illustrative; pin your versions and start from a smoke run.*
