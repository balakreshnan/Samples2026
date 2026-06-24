# Claude Skill — `databricks-mosaic-sdd` (Databricks Mosaic AI)

**Install to:** `.claude/skills/databricks-mosaic-sdd/SKILL.md`

---

## A. `SKILL.md` (copy verbatim)

````markdown
---
name: databricks-mosaic-sdd
description: >
  Spec-driven fine-tuning and serving on Databricks Mosaic AI (Foundation Model APIs, AI Functions,
  Model Serving, Unity Catalog, MLflow). Use when the user asks to "serve / fine-tune / deploy on
  Databricks", set up "provisioned throughput / pay-per-token / AI Functions batch", govern an
  endpoint with "Unity Catalog / Unity AI Gateway / guardrails / rate limits", or build production
  GenAI on the lakehouse. Builds on spec-driven-mlops.
---

# Databricks Mosaic AI (Spec-Driven)

Apply the spec-driven backbone (load spec-driven-mlops) to a lakehouse-native platform. Databricks
manages the GPUs — the spec enforces **serving-mode + governance + DBU cost**, not a GPU preflight.

## Serve & Governance Matrix (enforce in config.py — raise before creating an endpoint)
- `serve_mode ∈ {pay_per_token, ai_functions, provisioned_throughput, external_models}`.
- **Fine-tuned / custom models ⇒ provisioned_throughput (or custom serving), NEVER pay-per-token.**
- `provisioned_throughput` ⇒ model must be **optimizable** (eligible); set `min/max` throughput in
  the allowed **chunk increments**; `min ≤ max`.
- Every model is **registered in Unity Catalog** (`catalog.schema.model[@version]`).
- A **production** endpoint REQUIRES Unity AI Gateway: usage_tracking + inference_tables + guardrails
  + at least one rate_limit. CREATE/CAN MANAGE restricted to admins.
- Bulk/ETL inference ⇒ AI Functions (`ai_query`/`ai_classify`/…); interactive ⇒ provisioned throughput.

## Cost gate (DBUs)
Project DBU spend per surface vs a ceiling: FMAPI per-token · AI Functions (warehouse DBUs) ·
provisioned throughput (serving DBUs at tiers) · Vector Search (query + storage + reindex).
Crossover: pay-per-token until sustained volume beats a provisioned tier. Watch idle provisioned
capacity (set autoscale min) and Vector Search reindex on fast-changing corpora.

## Build (from validated config)
- Provisioned endpoint (REST): GET `get-model-optimization-info` (confirm optimizable + chunk size),
  then POST `/api/2.0/serving-endpoints` with `served_entities` (min/max throughput) + an `ai_gateway`
  block (usage tracking, inference tables, guardrails, rate limits).
- Fine-tune via Mosaic AI Model Training → register in UC → MLflow run + eval → deploy provisioned.
- Inference: OpenAI SDK to the serving endpoint (`response_format` for JSON), or `ai_query` in SQL for batch.

## Test gates
- Matrix enforcement (no cluster) · eligibility preflight (optimizable + chunks) · cost gate ·
  governance check (UC registration + gateway) · smoke + structured output · eval gate · spec coverage.

## Guardrails
- Never serve an un-registered (non-UC) model. Production endpoints REQUIRE gateway guardrails +
  rate limits. Tokens from Databricks secrets. Confirm serving payloads / fine-tuning APIs per release.

Reference: spec-driven-databricks-mosaic-ai.md.
````

---

## B. `CLAUDE.md` snippet

```markdown
## Platform: Databricks Mosaic AI
- Use skill `databricks-mosaic-sdd`. Register every model in Unity Catalog; fine-tuned/custom models
  ⇒ provisioned throughput (not pay-per-token). Production endpoints REQUIRE Unity AI Gateway
  (usage tracking + inference tables + guardrails + rate limits). Run a DBU cost gate. Tokens from secrets.
```

---

## C. Other necessary files
`src/dbx/{config,cost}.py`, `configs/*.yaml` (UC model name, serve_mode, throughput, gateway),
`tests/{test_matrix,test_cost,test_eval_gate,test_spec_coverage}.py`, MLflow run + UC model version +
deployment manifest, Databricks Asset Bundles / Terraform (optional IaC), Databricks secrets +
`.gitignore`, `pytest.ini` (markers `live`).
