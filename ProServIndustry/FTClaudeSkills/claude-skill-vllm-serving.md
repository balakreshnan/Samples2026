# Claude Skill — `vllm-serving-sdd` (vLLM)

**Install to:** `.claude/skills/vllm-serving-sdd/SKILL.md`

---

## A. `SKILL.md` (copy verbatim)

````markdown
---
name: vllm-serving-sdd
description: >
  Spec-driven self-hosted LLM serving with vLLM (PagedAttention, continuous batching,
  OpenAI-compatible). Use when the user asks to "serve / deploy a model on vLLM", "set up a
  vLLM endpoint / vllm serve config", configure "tensor parallelism / quantization (AWQ/GPTQ/FP8)
  / multi-LoRA / structured (guided) output / prefix caching" on vLLM, size GPUs for vLLM
  inference, or add SLO/governance gates to a vLLM deployment. Builds on spec-driven-mlops.
---

# vLLM Serving (Spec-Driven)

Apply the spec-driven backbone (load spec-driven-mlops) to a self-hosted vLLM endpoint. "Done" =
SLOs met under a GPU-fit + cost ceiling.

## Serving Matrix (enforce in config.py — raise before launch)
- `tensor_parallel_size` divides `num_gpus` AND ≤ `gpus_per_node` (TP on NVLink).
- `quantization ∈ {none, awq, gptq, fp8}`; `fp8` (weights or KV cache) ⇒ Hopper/Blackwell.
- `max_model_len` and `max_num_seqs` REQUIRED (they bound KV-cache memory).
- `enable_lora=true` REQUIRED if any `lora_modules` declared.
- `gpu_memory_utilization` in (0, 0.97].
- If structured output is required → requests carry a guided constraint
  (`guided_json` / `guided_choice` / `guided_regex` / `guided_grammar`).

## GPU-fit preflight (no peak-OOM surprises)
`weights + KV_cache(max_num_seqs, max_model_len) + overhead ≤ gpu_memory_utilization × VRAM × num_gpus`.
Weights ≈2 GB/1B (FP16), ≈1 GB/1B (FP8). KV cache scales with concurrency × context — bound it.

## Launch (render from validated config; never hand-type flags)
`vllm serve <model> --max-model-len N --max-num-seqs N --tensor-parallel-size T
  --gpu-memory-utilization 0.9 --enable-prefix-caching --api-key $VLLM_API_KEY
  --enable-request-id-headers [--quantization fp8] [--kv-cache-dtype fp8_e4m3]
  [--enable-lora --lora-modules ...]`
Server at `:8000/v1`; health `/health`; Prometheus `/metrics`. Client = OpenAI SDK with that base_url;
pass vLLM-only params (guided decoding) via `extra_body`.

## SLOs (declare + load-test; block promotion on miss)
TTFT p95, output tokens/sec, p99 latency, error rate — at a target concurrency. Drive with
`vllm bench serve` / genai-perf / locust; scrape `/metrics`.

## Test gates
- Matrix enforcement (no GPU) · GPU-fit preflight (no GPU) · render check · health/smoke (GPU) ·
  structured-output contract (GPU) · load/SLO gate · spec coverage.

## Guardrails
- Pin the vLLM version + image digest; confirm engine-arg / guided-decoding names per release.
- API key from env; front with an auth gateway; enable request-id headers.
- Never leave `max_model_len`/`max_num_seqs` unset (unbounded KV cache → OOM). Use guided decoding,
  not free-text JSON parsing. Don't promote without a load test.

Reference: spec-driven-vllm-serving.md.
````

---

## B. `CLAUDE.md` snippet

```markdown
## Serving: vLLM (self-hosted)
- Use skill `vllm-serving-sdd`. Render `vllm serve` from a validated pydantic config; never hand-type flags.
- Enforce the Serving Matrix + GPU-fit preflight; set max_model_len & max_num_seqs; structured output
  via guided decoding. Declare SLOs (TTFT/TPS/p99) and block promotion on a missed load test.
- Pin vLLM version + image digest. API key from env. /health + /metrics wired before traffic.
```

---

## C. Other necessary files
`src/serving/{config,gpu_preflight,render,client}.py`, `configs/*.yaml` (every engine arg, in VC),
`tests/{test_matrix,test_preflight,test_structured,test_load_slo,test_spec_coverage}.py`,
`Dockerfile`/k8s Deployment+HPA (optional), pinned vLLM **image digest**, `.env.example` (`VLLM_API_KEY`),
`pytest.ini` (markers `gpu`, `live`).
