# Claude Skill — `nvidia-nemo-sdd` (NVIDIA NeMo / NeMo-RL)

**Install to:** `.claude/skills/nvidia-nemo-sdd/SKILL.md`

---

## A. `SKILL.md` (copy verbatim)

````markdown
---
name: nvidia-nemo-sdd
description: >
  Spec-driven fine-tuning on NVIDIA's stack — NeMo Framework 2.0 (SFT + PEFT) and NeMo-RL
  (DPO, GRPO/GSPO, RLVR, reward modeling, distillation). Use when the user asks to "fine-tune
  with NeMo / NeMo-RL", run "GRPO/DPO/SFT on NeMo", use "Megatron / DTensor backend", pick
  NVIDIA GPUs, calculate how many GPUs are needed, or plan tensor/pipeline/context/expert
  parallelism for training. Builds on spec-driven-mlops.
---

# NVIDIA NeMo Fine-Tuning (Spec-Driven)

Apply the spec-driven backbone (load spec-driven-mlops). Enforce scope × objective × backend × GPU.

## Framework choice
- SFT / PEFT on a supported model → **NeMo Framework 2.0** recipes (`finetune_recipe`, `peft_scheme`).
- DPO / GRPO / GSPO / RLVR / reward modeling / distillation → **NeMo-RL** (YAML config, Ray).
- HF checkpoint not natively ported → **NeMo AutoModel** (DTensor).  Managed API → NeMo Customizer.

## Matrices (enforce in config.py — raise before the cluster is touched)
- **Scope:** `full | lora | dora | qlora | ptuning | ia3`. PEFT ⇒ `freeze_base=true`; qlora ⇒ 4-bit.
- **Objective:** `sft | dpo | grpo | gspo | dapo | rm | rlvr | distill`. GRPO/RLVR ⇒ `num_generations>1`
  + `max_kl`; RLVR ⇒ deterministic verifier **environment**, NO learned reward_model.
- **Backend:** `dtensor | megatron`. MoE/Expert-Parallel (EP) and Context-Parallel (CP) ⇒ **megatron**.
- **GPU:** `tp*pp*dp*ep == num_gpus`; `tp ≤ gpus_per_node` (TP on NVLink, PP/DP on InfiniBand);
  FP8 ⇒ Hopper/Blackwell; FP4 ⇒ Blackwell.

## GPU sizing & count (the load-bearing preflight)
- Full FT ≈16 GB/1B; LoRA ≈ base+25%; QLoRA(4-bit) ≈0.55 GB/1B+40%.
  RL/GRPO ≈ ~2.5× the SFT footprint (policy + reference + vLLM/SGLang generation engine).
- `num_gpus = ceil(total_vram × 1.3 / (0.9 × per-gpu VRAM))`, rounded to a legal `tp×pp×dp×ep`
  and to whole 8-GPU nodes for multi-node. Reject a plan that cannot fit BEFORE it queues.
- GPU picks: QLoRA≤13B → 1 GPU; LoRA 70B → ~4×80 GB or 1× H200/B200; full 70B → ~24 GPUs (3 nodes).

## Workflow
1. Scaffold specs + `config.py` enforcing all four matrices + a `gpu_planner.py` (VRAM + fit preflight).
2. NeMo 2.0: build a recipe from config (`peft_scheme`, parallelism from the GPU plan); `import_ckpt` HF→NeMo.
3. NeMo-RL: render a YAML (policy backend, `num_generations`, `max_kl`, generation=vllm/sglang) + a
   verifier environment for RLVR.
4. Persist a manifest: NGC container tag + git commit + config + metrics.

## Test gates
- Matrix enforcement · GPU-fit preflight (the highest-ROI test) · data integrity · verifier
  determinism/adversarial · 1-step smoke (base frozen for PEFT) · eval gate · spec coverage.

## Guardrails
- Pin the **NGC container tag**; NeMo repos are reorganizing — confirm recipe/config APIs per release.
- TP must stay within a node. Size MoE for **total** params (EP needs Megatron). Long context ⇒ CP ⇒ Megatron.
- Convert HF→NeMo once (`import_ckpt`), record checksum. Prefer cheapest method that clears the gate.

Reference: spec-driven-nvidia-finetuning-methodology.md (matrices, gpu_planner, parallelism cheat-sheet).
````

---

## B. `CLAUDE.md` snippet

```markdown
## Fine-tuning: NVIDIA NeMo
- Use skill `nvidia-nemo-sdd`. SFT/PEFT → NeMo 2.0 recipes; RL/DPO/GRPO/RLVR → NeMo-RL.
- Enforce scope × objective × backend × GPU matrices + a GPU-fit preflight BEFORE the job queues.
- TP within a node; EP/CP/MoE ⇒ Megatron backend. Pin the NGC container tag. RL footprint ≈2.5× SFT.
```

---

## C. Other necessary files
`src/nvft/{config,gpu_planner}.py`, `configs/{lora,qlora,full,grpo,rlvr}.yaml` (NeMo-RL) or NeMo-Run
recipe scripts, `tests/{test_matrix_enforcement,test_gpu_planner,test_spec_coverage}.py`,
a pinned **NGC container** reference, `run_manifest.json` writer, `cluster/*.sub` (Slurm) or KubeRay
manifests for multi-node, `.env.example` (`HF_TOKEN`, NGC creds), `pytest.ini` (markers `gpu`, `live`).
