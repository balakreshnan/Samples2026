# Claude Skill — `spec-driven-mlops` (Base)

*The shared spec-driven backbone every vendor skill builds on. Install this first.*

**Install to:** `.claude/skills/spec-driven-mlops/SKILL.md` (project) or `~/.claude/skills/spec-driven-mlops/SKILL.md` (personal).

---

## A. `SKILL.md` (copy verbatim)

````markdown
---
name: spec-driven-mlops
description: >
  Apply Spec-Driven Development to any LLM fine-tuning or serving task. Use when the user
  asks to "use spec-driven development", "scaffold a spec / constitution / matrix", "make this
  run config-driven", "enforce rules before training/serving", "add an eval gate / cost gate /
  SLO gate", "set up specs/ for this project", or starts any fine-tuning or model-serving work
  that should be governed, reproducible, and tested before GPUs/credits are spent. Also fires as
  the foundation for the vendor skills (hf-finetuning-sdd, nvidia-nemo-sdd, vllm-serving-sdd,
  sglang-serving-sdd, fireworks-sdd, databricks-mosaic-sdd, snowflake-cortex-sdd).
---

# Spec-Driven MLOps

You help users build fine-tuning and serving systems where a **versioned spec — not the code —
is the source of truth**, and an invalid run **fails before** any GPU hour or credit is spent.

## When this fires
Any fine-tuning, post-training (RL/RLVR/DPO), or model-serving task that should be reproducible,
governed, and tested. If a specific vendor is named (vLLM, SGLang, Fireworks, Databricks,
Snowflake, NeMo, Hugging Face/TRL), also load that vendor's `*-sdd` skill and apply its matrix.

## The workflow — always produce these artifacts
1. **`specs/constitution.md`** — non-negotiable rules: security (secrets via env), reproducibility
   (pin versions/revisions/seed; record a manifest), data governance (license + PII + no
   train/test leakage), an **eval gate**, a **cost/GPU ceiling**, and quality (typed, tested).
2. **`specs/spec.md`** — WHAT & WHY: functional requirements (`FR-001`…), an authoritative
   **enforceable matrix** (required/forbidden fields per technique/mode), SLOs or an eval
   threshold, and acceptance criteria (`AC-1`…) written Given/When/Then.
3. **`specs/plan.md`** — HOW: stack, technique/mode choice, hyperparameters, hardware/parallelism.
4. **`specs/tasks.md`** — atomic, testable items, each traced to an `FR`/`AC`.
5. **`src/.../config.py`** — a **pydantic enforcer**: load the run config and RAISE on any matrix
   violation. This is the heart of the method — invalid runs never reach the GPU/platform.
6. **`tests/`** — six+ layers (cheapest first):
   - **Matrix enforcement** (no GPU): illegal configs rejected.
   - **Preflight** (no GPU): GPU-fit / row-limit / RBAC / cost — whatever the vendor bounds.
   - **Data integrity** (no GPU): license, PII screen, no train/test leakage.
   - **Smoke** (small/GPU): 1-step train or 1 completion actually runs.
   - **Eval / SLO gate**: fine-tune beats base by threshold OR serving meets SLOs; else block.
   - **Spec coverage** (no GPU): every `FR`/`AC` in spec.md is referenced by ≥1 test.

## Core principles
- **Two axes for fine-tuning:** *scope* (full/LoRA/QLoRA/…) × *objective* (SFT/DPO/GRPO/RLVR/…).
  They compose; enforce both.
- **Enforce before spend:** the config validator is the highest-ROI artifact — it turns a
  multi-hour OOM or a surprise invoice into a millisecond CI failure.
- **Fix the spec, then the code.** If reality disagrees with the spec, update the spec first.
- **Single-source invariants** (label sets, schemas, limits) so prose, prompt, and validator
  can't drift. Reference requirement IDs (`# FR-006`) inside tests for the coverage gate.
- **Secrets via env/secret-store. Never fabricate** model names, limits, prices, or APIs — leave a
  clearly marked placeholder and tell the user to confirm against current vendor docs.

## What to deliver
Scaffold the five spec/config files + a starter test suite, wired so `pytest -q -m "not gpu and
not live"` runs the cheap gates. Keep specs lightweight (spec the slice being built now), and add
the vendor matrix from the matching `*-sdd` skill.
````

---

## B. `CLAUDE.md` snippet (add to the repo)

```markdown
## Spec-Driven MLOps
- All fine-tuning/serving is spec-driven: scaffold specs/{constitution,spec,plan,tasks}.md +
  a pydantic config enforcer before writing run code (skill: spec-driven-mlops).
- An invalid run must fail at config-load, not at deploy time.
- Gates in CI: matrix enforcement, preflight, data integrity, eval/SLO, spec-coverage.
- Cheap gates: `pytest -q -m "not gpu and not live"`.
```

---

## C. Other necessary files (the skill scaffolds these)

- `specs/constitution.md`, `specs/spec.md`, `specs/plan.md`, `specs/tasks.md`
- `src/<pkg>/config.py` — the pydantic enforcer (vendor skills provide the concrete matrix)
- `tests/test_matrix.py`, `tests/test_preflight.py`, `tests/test_eval_gate.py`, `tests/test_spec_coverage.py`
- `pytest.ini` with markers: `gpu`, `live`
- `.env.example`, `.gitignore` (never commit secrets)

`pytest.ini`:
```ini
[pytest]
pythonpath = src
markers =
    gpu: requires a GPU / model load (opt-in)
    live: hits the real platform / needs credentials (opt-in)
```

---

*This base skill is referenced by all seven vendor skills. Install it first, then add the vendor(s) you deploy on. Full rationale lives in `spec-driven-development-tutorial.md`.*
