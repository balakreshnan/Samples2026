# Claude Skill — `hf-finetuning-sdd` (Hugging Face)

**Install to:** `.claude/skills/hf-finetuning-sdd/SKILL.md`

---

## A. `SKILL.md` (copy verbatim)

````markdown
---
name: hf-finetuning-sdd
description: >
  Spec-driven fine-tuning with the Hugging Face stack (transformers + trl + peft + datasets).
  Use when the user asks to "fine-tune with Hugging Face / TRL / PEFT", run "SFT / LoRA / QLoRA /
  full fine-tuning", do preference or RL post-training ("DPO / KTO / GRPO / RLVR / reward model"),
  write verifiable reward functions, choose a fine-tuning technique, or size GPUs for HF training.
  Builds on spec-driven-mlops.
---

# Hugging Face Fine-Tuning (Spec-Driven)

Apply the spec-driven backbone (load spec-driven-mlops) to HF fine-tuning. Enforce TWO axes.

## Axis 1 — Parameter scope: `full | lora | qlora`  (Scope Matrix)
- `full`  → no LoRA block; `load_in_4bit=false`; trains all weights.
- `lora`  → LoRA block REQUIRED; `freeze_base=true`; `load_in_4bit=false`.
- `qlora` → LoRA block REQUIRED; `freeze_base=true`; `load_in_4bit=true`;
            `bnb_4bit_quant_type="nf4"`; `gradient_checkpointing=true`.
A config violating these MUST raise at load (before any GPU work).

## Axis 2 — Objective: `sft | dpo | kto | grpo | rlvr`  (Objective Matrix)
- `sft`  → SFTTrainer/SFTConfig (current TRL: `processing_class=`, `max_length=`).
- `dpo`  → DPOTrainer/DPOConfig; preference dataset (prompt/chosen/rejected); `beta` REQUIRED;
           reward_funcs/reward_model FORBIDDEN.
- `kto`  → KTOTrainer; (prompt/completion/label); `beta` REQUIRED.
- `grpo` → GRPOTrainer; `reward_funcs ≥1` (or reward_model); `num_generations>1`; `max_kl` set.
- `rlvr` → GRPOTrainer with **deterministic verifier** reward functions; `reward_model` FORBIDDEN;
           dataset has a ground-truth field; `num_generations>1`; `max_kl` set.
The axes COMPOSE (e.g., GRPO+QLoRA). Standard recipe: SFT first, then preference/RL.

## TRL reward-function shape (RLVR)
`def reward(completions, **dataset_columns) -> list[float]`. Register verifiers in a `REGISTRY`;
resolve names from config; a strong verifier checks the **answer**, not just format (anti reward-hacking).

## GPU sizing (rules of thumb, bf16)
- Full FT: ≈16 GB/1B (2 wts + 2 grad + 12 optimizer/master); 8-bit optimizer ≈10 GB/1B.
- LoRA: ≈ base weights + ~25%.  QLoRA(4-bit): ≈0.55 GB/1B base + ~40%.
GPUs ≈ ceil(total_vram × 1.3 / (0.9 × per-gpu VRAM)); TP within a node. Preflight before launch.

## Workflow
1. Scaffold specs + `config.py` enforcing BOTH matrices (pydantic, raise on violation).
2. `data.py` — license/PII screen, seeded non-overlapping splits, format to TRL schema.
3. `modeling.py` — build (model, peft_config) per scope; `BitsAndBytesConfig` nf4 for qlora.
4. `train.py` (SFTTrainer) and/or `train_rl.py` (GRPO/DPO/KTO); set seed; write a run manifest.
5. `rewards.py` for RLVR verifiers (deterministic + adversarially tested).

## Test gates (cheapest first)
- Contract (pydantic output schema) · matrix/objective enforcement · data no-leakage ·
  verifier determinism + adversarial (no reward hacking) · 1-step smoke (assert base frozen for
  LoRA) · eval gate (beat base by threshold) · spec coverage (every FR/AC tested).

## Guardrails
- Pin transformers/trl/peft/bitsandbytes versions (APIs move fast — confirm arg names).
- HF token from env. Never train on the test split. Reward curve ≠ proof — also run a held-out eval.
- Don't fork a script per technique — one config-driven entrypoint per objective.

Reference: spec-driven-finetuning-huggingface-tutorial.md (full code + RLVR section).
````

---

## B. `CLAUDE.md` snippet

```markdown
## Fine-tuning: Hugging Face
- Use skill `hf-finetuning-sdd`. Enforce scope (full|lora|qlora) × objective (sft|dpo|kto|grpo|rlvr)
  via a pydantic config enforcer; invalid configs fail at load.
- QLoRA ⇒ 4-bit NF4 + gradient checkpointing. RLVR ⇒ deterministic verifier rewards, no reward model.
- SFT first, then RL. Pin transformers/trl/peft versions. HF_TOKEN from env. Held-out eval gate required.
```

---

## C. Other necessary files
`requirements.txt` (pinned: `transformers trl peft datasets accelerate bitsandbytes evaluate pydantic pytest`),
`src/finetune/{config,data,modeling,train,train_rl,rewards}.py`, `configs/{lora,qlora,full,rlvr}.yaml`,
`tests/{test_config_enforcement,test_data_integrity,test_verifiers,test_eval_gate,test_spec_coverage}.py`,
`.env.example` (`HF_TOKEN`), `pytest.ini` (markers `gpu`, `live`).
