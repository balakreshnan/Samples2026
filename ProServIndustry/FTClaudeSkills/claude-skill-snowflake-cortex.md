# Claude Skill — `snowflake-cortex-sdd` (Snowflake Cortex AI)

**Install to:** `.claude/skills/snowflake-cortex-sdd/SKILL.md`

---

## A. `SKILL.md` (copy verbatim)

````markdown
---
name: snowflake-cortex-sdd
description: >
  Spec-driven fine-tuning and inference on Snowflake Cortex AI (AI_COMPLETE / AISQL, REST
  Complete/Embed/Agents, Cortex Fine-tuning PEFT/LoRA, Cortex Training full+RL via ArcticTraining).
  Use when the user asks to "fine-tune / inference on Snowflake Cortex", call "AI_COMPLETE /
  FINETUNE / AI_CLASSIFY", choose "AISQL vs REST", run "Cortex Training / RL", or build GenAI that
  keeps data inside the Snowflake perimeter. Builds on spec-driven-mlops.
---

# Snowflake Cortex AI (Spec-Driven)

Apply the spec-driven backbone (load spec-driven-mlops) to a serverless, in-perimeter platform. The
spec enforces **method + inference path + RBAC + row limits + credit cost** (no GPU preflight).

## Method & Inference Matrices (enforce in config.py — raise before any FINETUNE/AI_COMPLETE)
- `method ∈ {cortex_finetune (PEFT/LoRA, GA), cortex_training (full FT + RL, preview/ArcticTraining)}`.
  **RL or full fine-tuning ⇒ cortex_training**, never cortex_finetune.
- `cortex_finetune` base model must be in the fine-tunable list (mistral-7b, mixtral-8x7b,
  llama3-8b/70b, llama3.1-8b/70b) AND `training_rows ≤ effective_row_limit(model, epochs)`
  (e.g., mistral-7b @3 epochs = 15k; llama3.1-70b @3 = 4.5k).
- `inference_path ∈ {aisql, rest}`: bulk/table ⇒ AISQL (`AI_COMPLETE` over rows); interactive ⇒ REST.
- RBAC: calling role has `SNOWFLAKE.CORTEX_USER` + `USE AI FUNCTIONS`; fine-tune needs `CREATE MODEL`.

## Governance (in-perimeter — the Snowflake core)
Training + inference data NEVER leaves Snowflake. Screen/redact PII with `AI_REDACT`. Track spend via
`SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FINE_TUNING_USAGE_HISTORY`. Snapshot the training table; record job ID.

## Cost gate (credits)
FT credits = input_tokens × epochs · inference = input+output tokens · AISQL adds **warehouse** runtime.
Use `AI_COUNT_TOKENS` to estimate volume; size the warehouse to the workload (oversized = wasted credits).

## Build (from validated config)
- Fine-tune: `SELECT SNOWFLAKE.CORTEX.FINETUNE('CREATE', 'db.schema.model_ft', 'mistral-7b',
  '<train query>', '<eval query>')`; track with `FINETUNE('DESCRIBE', <job_id>)`.
- Full FT / RL: ArcticTraining YAML (`type: sft|grpo|dpo`, `model: Qwen/...`) on managed GPU pools.
- Inference: `AI_COMPLETE('db.schema.model_ft', col)` over a table (AISQL); `response_format` JSON
  schema for structured output; REST Complete API for low latency; Python `snowflake.cortex.complete`.

## Test gates
- Matrix enforcement (no compute) · row-limit preflight · RBAC preflight (SHOW GRANTS) · cost gate
  (AI_COUNT_TOKENS) · smoke + structured output · eval gate (beat base) · spec coverage.

## Guardrails
- Never copy data out to an external API (defeats the residency benefit). Don't exceed model/epoch
  row limits or request RL via Cortex Fine-tuning. Redact PII before model calls. Right-size the
  warehouse for AISQL batch. Cortex Training is **preview** — confirm models/limits/rates per region.

Reference: spec-driven-snowflake-cortex-ai.md.
````

---

## B. `CLAUDE.md` snippet

```markdown
## Platform: Snowflake Cortex AI
- Use skill `snowflake-cortex-sdd`. Enforce method (cortex_finetune PEFT vs cortex_training full/RL),
  the per-model/epoch row limit, inference path (AISQL bulk vs REST interactive), and RBAC
  (CORTEX_USER + CREATE MODEL). Data stays in-perimeter; AI_REDACT PII. Credit cost gate via
  AI_COUNT_TOKENS; right-size the warehouse. RL ⇒ Cortex Training, not Cortex Fine-tuning.
```

---

## C. Other necessary files
`src/cortex/{config,cost}.py`, `sql/{finetune,inference}.sql` (in VC), `configs/*.yaml` (method,
base_model, epochs, training_rows, path, ceiling), ArcticTraining YAML (for Cortex Training),
`tests/{test_matrix,test_rowlimit,test_rbac,test_cost,test_spec_coverage}.py`, training-table snapshot
+ FINETUNE job ID + model name (manifest), key-pair/OAuth via secrets, `pytest.ini` (markers `live`).
