# Claude Skill — `fireworks-sdd` (Fireworks AI)

**Install to:** `.claude/skills/fireworks-sdd/SKILL.md`

---

## A. `SKILL.md` (copy verbatim)

````markdown
---
name: fireworks-sdd
description: >
  Spec-driven build on the Fireworks AI managed platform (serverless inference, dedicated GPU
  deployments, managed fine-tuning LoRA/full SFT+DPO, batch). Use when the user asks to "deploy /
  fine-tune / serve on Fireworks", choose "serverless vs dedicated", deploy a LoRA via "live-merge
  vs multi-LoRA", use firectl / the Fireworks OpenAI-compatible API, or control Fireworks cost. No
  GPU ops — the spec enforces platform rules + cost. Builds on spec-driven-mlops.
---

# Fireworks AI (Spec-Driven)

Apply the spec-driven backbone (load spec-driven-mlops) to a MANAGED platform: enforce deployment
rules + a cost ceiling (the platform runs the GPUs, so there's no GPU preflight — there's a cost gate).

## Deployment Matrix (enforce in config.py — raise before any API call)
- `deploy_mode ∈ {serverless, dedicated}`; LoRA ⇒ `lora_method ∈ {live_merge, multi_lora}`.
- **LoRA addons CANNOT be serverless** — dedicated only.  (the #1 platform rule)
- **multi_lora REQUIRES a bf16 deployment shape** — FP8/FP4 shapes reject addons.
- A fine-tuned model is referenced by its full resource name `accounts/<id>/models/<id>`.
- `dedicated` ⇒ a shutdown / autoscale-to-zero policy is REQUIRED (GPU-second billing runs 24/7).
- A fine-tune declares a dataset with license + PII screen.

## Cost gate (the managed-platform preflight)
Project spend at expected volume per mode vs a declared ceiling:
serverless (~$/M tokens) · **batch = 50% of serverless** (non-real-time bulk) · dedicated (~$/GPU-hr).
Crossover: serverless wins until sustained volume beats a dedicated GPU's hourly cost. Bulk → batch.
Flag the run for redesign if projected cost > ceiling.

## Build (from validated config)
- Fine-tune (firectl): `firectl create fine-tuning-job --base-model <m> --dataset <d> --kind sft --lora`.
- Deploy LoRA dedicated: `firectl deployment create accounts/<id>/models/<ft>` (live-merge) or
  `firectl deployment create <base> --enable-addons` (multi-LoRA, bf16).
- Inference: OpenAI SDK `base_url=https://api.fireworks.ai/inference/v1` (or `from fireworks import
  Fireworks`); structured output via `response_format`; model = full resource name.

## Test gates
- Matrix enforcement (no API) · cost gate (no API) · dataset validation (license/PII) ·
  smoke + structured output (live) · eval gate (fine-tune beats base) · spec coverage.

## Guardrails
- `FIREWORKS_API_KEY` from env. Never deploy a LoRA serverless; never multi-LoRA on FP8/FP4.
- Always put a shutdown policy on dedicated; route bulk through batch (50% off). Confirm current
  model IDs / prices / firectl syntax against live docs (they change frequently).

Reference: spec-driven-fireworks-platform.md.
````

---

## B. `CLAUDE.md` snippet

```markdown
## Platform: Fireworks AI (managed)
- Use skill `fireworks-sdd`. Enforce the Deployment Matrix: LoRA ⇒ dedicated only; multi-LoRA ⇒ bf16
  shape; dedicated ⇒ shutdown/autoscale policy. Run a cost gate (serverless vs dedicated vs 50% batch)
  before deploying. FIREWORKS_API_KEY from env. Fine-tunes pass an eval gate.
```

---

## C. Other necessary files
`src/fw/{config,cost,client}.py`, `configs/*.yaml` (model, deploy_mode, lora_method, dataset, ceiling),
`firectl` scripts / IaC, `tests/{test_matrix,test_cost,test_eval_gate,test_spec_coverage}.py`,
run manifest (model resource name + job ID + dataset + metrics), `.env.example` (`FIREWORKS_API_KEY`),
`pytest.ini` (markers `live`).
