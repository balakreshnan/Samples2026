# Spec-Driven MLOps — Claude Skills Pack (README & Install Guide)

*A set of **Claude Code skills** that teach Claude to apply the spec-driven methodology from the companion handbook to each fine-tuning / serving vendor. Drop them into a repo (or your personal config) and Claude will scaffold specs, enforce each vendor's hard rules, and gate deployments — automatically, whenever you ask.*

> **Companion to:** the eight spec-driven guides (SDD methodology, Hugging Face, NVIDIA, vLLM, SGLang, Fireworks, Databricks, Snowflake). Those are *documentation for humans*; **these skills are instructions for Claude.**

---

## 1. What's in this pack

| File (this folder) | Becomes the skill… | Triggers when you ask Claude to… |
|---|---|---|
| `claude-skill-spec-driven-base.md` | `spec-driven-mlops` | "apply spec-driven dev", "scaffold a spec", "write a constitution + matrix", "add an enforce-before-run gate" |
| `claude-skill-huggingface-finetuning.md` | `hf-finetuning-sdd` | "fine-tune with Hugging Face / TRL / PEFT", "LoRA/QLoRA/SFT/DPO/GRPO/RLVR", "verifier rewards" |
| `claude-skill-nvidia-nemo.md` | `nvidia-nemo-sdd` | "fine-tune with NeMo / NeMo-RL", "GRPO on NeMo", "size GPUs / parallelism for training" |
| `claude-skill-vllm-serving.md` | `vllm-serving-sdd` | "serve/deploy on vLLM", "vLLM endpoint config", "structured output / GPU-fit for vLLM" |
| `claude-skill-sglang-serving.md` | `sglang-serving-sdd` | "serve on SGLang", "RadixAttention / prefix-cache", "SGLang launch config" |
| `claude-skill-fireworks.md` | `fireworks-sdd` | "deploy/fine-tune on Fireworks", "serverless vs dedicated", "live-merge vs multi-LoRA" |
| `claude-skill-databricks-mosaic-ai.md` | `databricks-mosaic-sdd` | "serve/fine-tune on Databricks", "provisioned throughput", "Unity Catalog + AI Gateway" |
| `claude-skill-snowflake-cortex.md` | `snowflake-cortex-sdd` | "fine-tune/inference on Snowflake Cortex", "AI_COMPLETE / FINETUNE", "Cortex Training / RL" |

Each vendor file contains a ready-to-paste **`SKILL.md`**, a **`CLAUDE.md`** snippet, and any **supporting files** (e.g., the `config.py` enforcer the skill references).

---

## 2. How Claude skills work (30-second primer)

A **skill** is a folder containing a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: my-skill
description: One line that tells Claude WHEN to use this skill (include trigger phrases).
---

# Body: the instructions, workflow, and rules Claude follows when the skill fires.
```

- **`name`** — unique, kebab-case.
- **`description`** — the most important field: Claude reads it to decide whether to invoke the skill, so it must name the **triggers** ("use when the user asks to…").
- **Body** — the operational playbook. Keep it focused; link to reference files for depth (progressive disclosure).
- A skill folder may also hold `reference.md`, `scripts/`, or templates the skill points to.

When a user request matches a skill's description, Claude loads the body and follows it.

---

## 3. Where to install

Pick one location depending on scope:

| Location | Scope | Use when |
|---|---|---|
| `<your-repo>/.claude/skills/<name>/SKILL.md` | This project, shared with the team via git | Most cases — the skill ships with the codebase it governs. |
| `~/.claude/skills/<name>/SKILL.md` | All your projects (personal) | You want the skill available everywhere you work. |
| Cowork personal skills: `Documents/Cowork` skills folder | Your Cowork assistant | You want Cowork (this assistant) to use it in chat. |

**Install steps (project example):**
```bash
mkdir -p .claude/skills/vllm-serving-sdd
# Copy the SKILL.md block from claude-skill-vllm-serving.md into:
#   .claude/skills/vllm-serving-sdd/SKILL.md
# Repeat per vendor you use. Commit to git so the team shares them.
```

> In **Cowork**, personal skills live under your OneDrive `Documents/Cowork` skills folder — say "create a skill" and the `skills` capability will scaffold it there. I can also install any of these as live Cowork skills for you on request.

---

## 4. The `CLAUDE.md` you should add to a repo

`CLAUDE.md` is project memory Claude reads at the start of every session. Add a block like this so Claude knows the project is spec-driven and which vendor skill to use:

```markdown
# CLAUDE.md — Project AI/MLOps Conventions

## Methodology
- This repo uses **Spec-Driven Development** for all fine-tuning and serving work.
  Before writing training/serving code, scaffold `specs/{constitution,spec,plan,tasks}.md`
  and a pydantic **config enforcer** that rejects invalid runs BEFORE GPUs/credits are spent.
  Use the `spec-driven-mlops` skill.

## Vendor
- Primary platform: **<vLLM | SGLang | Fireworks | Databricks | Snowflake | NeMo | Hugging Face>**.
  Use the matching `*-sdd` skill. Honor its enforceable matrix — a run that violates the
  matrix must fail at config-load, not at deploy time.

## Non-negotiables
- Secrets come from env vars / the platform secret store — never hard-coded or committed.
- Every fine-tune passes an **eval gate** (beat base by a declared threshold) before promotion.
- Every serving deploy declares **SLOs** and a **cost ceiling**; both are gated in CI.
- Every requirement (FR-xxx / AC-x) in spec.md maps to ≥1 test (the spec-coverage gate).

## Commands
- `pytest -q -m "not gpu and not live"`   # cheap gates: matrix, preflight, cost, coverage
- `pytest -q -m gpu`                       # smoke/load (needs hardware)
- `pytest -q -m live`                      # hits the real platform (needs creds)
```

---

## 5. Optional: bundle them as one installable plugin

To ship the whole pack as a single Claude Code **plugin**, lay it out like this and add a manifest:

```
spec-driven-mlops-plugin/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    ├── spec-driven-mlops/SKILL.md
    ├── hf-finetuning-sdd/SKILL.md
    ├── nvidia-nemo-sdd/SKILL.md
    ├── vllm-serving-sdd/SKILL.md
    ├── sglang-serving-sdd/SKILL.md
    ├── fireworks-sdd/SKILL.md
    ├── databricks-mosaic-sdd/SKILL.md
    └── snowflake-cortex-sdd/SKILL.md
```

`.claude-plugin/plugin.json`:
```json
{
  "name": "spec-driven-mlops",
  "version": "1.0.0",
  "description": "Spec-driven fine-tuning & serving skills for HF, NVIDIA, vLLM, SGLang, Fireworks, Databricks, and Snowflake.",
  "author": "Your Team"
}
```

Then install the plugin folder per the Claude Code plugin docs (or point a marketplace at the repo).

---

## 6. How to validate a skill works

1. **Frontmatter check** — `name` is unique kebab-case; `description` names the triggers.
2. **Trigger test** — start a session and phrase a request the way a teammate would ("help me serve Llama-3-8B on vLLM with structured output"). Confirm Claude invokes the right skill.
3. **Conflict check** — if two skills could fire for the same phrase (e.g., vLLM vs SGLang), make each `description` name its distinctive triggers (engine name, RadixAttention vs PagedAttention, etc.).
4. In Cowork, the `skills` capability can **validate, audit, and test triggering** for you — say "validate my skills" / "do my skills conflict".

---

## 7. The shared rule every skill enforces

All eight skills implement the same backbone so they compose cleanly:

> **Constitution → Spec (+ enforceable matrix) → Plan → Tasks → Implement → Test.**
> A run config is validated against the vendor's matrix at load time; an invalid run **fails before** any GPU hour or credit is spent. Fine-tunes pass an eval gate; serving meets SLOs under a cost ceiling; every FR/AC has a test.

Start with **`spec-driven-mlops`** (the base), then add the vendor skill(s) you deploy on.

---

*Each vendor markdown file in this folder is self-contained: copy its `SKILL.md` block to `.claude/skills/<name>/SKILL.md`, paste its `CLAUDE.md` block into your repo's `CLAUDE.md`, and add any supporting files it lists. Confirm API/flag names against the vendor's current docs before relying on generated code.*
