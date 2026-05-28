# LLM Inference Optimization and Scaling to High-Concurrency Serving

> **Audience:** Principal Cloud Solution Architect (Microsoft). Assumes fluency with transformer internals, GPU memory hierarchies, NCCL/RDMA, and distributed systems.
> **Scope:** Pipeline-level structure of LLM inference; algorithmic, systems, and kernel-level optimizations; serving stack architecture and capacity planning for millions of concurrent users.
> **Date:** 2026-05-28

---

## Executive Summary

Modern LLM inference is the textbook case of a workload that breaks every assumption baked into "request-response" serving infrastructure. A single user "request" is two qualitatively different jobs glued together: a **prefill** that is dense matmul on the entire prompt (compute-bound) and a **decode** loop of hundreds of low-arithmetic-intensity steps (memory-bandwidth-bound). The 2023-2026 systems literature is converging on a single architectural conclusion: stop running them on the same GPU.

Six structural moves dominate the state of the art:

1. **Disaggregation of prefill and decode pools** (Splitwise, DistServe, Mooncake) — separate SLOs (TTFT vs TPOT/TBT), separate hardware sizing, KV cache transferred over RDMA.
2. **Block-paged KV management with cross-request prefix reuse** (vLLM PagedAttention, SGLang RadixAttention, Mooncake Store) — eliminates internal fragmentation and amortizes prefill across the long-tail of shared system prompts.
3. **Iteration-level continuous batching with chunked prefill** (Orca, Sarathi-Serve) — both eliminate head-of-line blocking and keep the GPU compute-bound during decode by piggy-backing prefill chunks.
4. **FP8/FP4 quantization plus weight-only INT4** (FlashAttention-3 FP8, AWQ, GPTQ, SmoothQuant) — doubles or quadruples effective compute and shrinks the bandwidth burden on decode.
5. **MoE + Expert Parallelism + MLA** (DeepSeek-V3, Kimi K2) — sparse activation amortizes parameter cost; latent attention shrinks the KV-cache that dominates decode bandwidth.
6. **Kernel co-design with the hardware** (FlashAttention-1/2/3, CUDA Graphs, TensorRT-LLM custom kernels, torch.compile, NVIDIA Dynamo orchestration) — closes the gap to the roofline.

At million-user scale, the serving stack becomes a *KVCache-centric distributed system*. Mooncake (FAST 2025 best paper) is the most explicit articulation of this thesis: a multi-tier KV cache pool (HBM → DRAM → SSD) interconnected by RDMA up to 8×400 Gbps drives Kimi to handle "100 billion tokens daily" with 115% and 107% more requests than the prior systems on A800 and H800 clusters respectively [3]. The same pattern, with Microsoft-specific framing, appears in Splitwise [7] and is now landing in Azure AI Foundry via NIM, TensorRT-LLM, and Dynamo integrations [28][29].

This report walks the stack bottom-up: pipeline structure, optimization techniques per layer, then horizontal scaling architecture.

---

## PART 1: Inference Pipeline Stages

### 1.1 The Two-Phase Anatomy of a Generative Request

Every decoder-only transformer inference goes through two phases [3][7]:

```
                ┌─────────────────────────────────────────────┐
 User prompt    │  PREFILL (1 forward pass over N tokens)    │
 N tokens ─────▶│  GEMM-heavy. Compute-bound.                │──┐
                │  Output: KV cache (2 × L × H × d × N bytes)│  │
                └─────────────────────────────────────────────┘  │
                                                                 ▼
                ┌─────────────────────────────────────────────┐
 First token    │  DECODE (T iterations, 1 token each)        │
 …  ◀───────────│  GEMV per token. Memory-bandwidth bound.   │
 Final token    │  Each step: read KV cache + new Q          │
                │  Output: append K_new, V_new to cache       │
                └─────────────────────────────────────────────┘
```

Two SLOs govern user-visible latency [1][3]:
- **TTFT** (Time To First Token): duration of prefill plus queue wait.
- **TPOT** / **TBT** (Time Per Output Token / Time Between Tokens): the inter-token interval during decode.

#### Prefill vs Decode: Side-by-Side

| Dimension | **Prefill** | **Decode** |
|-----------|-------------|------------|
| **What it does** | Processes the full prompt in one forward pass; emits the first output token and the KV cache | Generates output tokens one at a time, autoregressively, reusing the KV cache |
| **Tokens processed per step** | `N` (entire prompt, e.g., 2K–128K) | 1 per request |
| **Number of steps per request** | 1 | `T` (output length, e.g., 100–2000+) |
| **Dominant op shape** | GEMM (matrix × matrix) | GEMV (matrix × vector) |
| **Bottleneck** | **Compute** — Tensor Core FLOPs | **Memory bandwidth** — HBM reads |
| **Arithmetic intensity** | High (`∝ N`); easily exceeds H100 machine balance (~295 FLOPs/byte) | Very low (`≈ B/2` for FP16 batch `B`); needs B≈590 to saturate H100 |
| **GPU utilization** | 70–75% of peak FLOPs (FlashAttention-3) | Tensor Cores mostly idle; HBM saturated |
| **Scaling with input length** | Latency `O(N²)` (attention) + `O(N)` (FFN) | KV cache grows `O(context)`; per-step latency grows with context |
| **Scaling with batch size** | Marginal speedup once one prefill saturates the GPU | Near-linear throughput gain — batching is critical |
| **Latency SLO** | **TTFT** (Time To First Token), e.g., <500 ms | **TPOT / TBT**, e.g., 20–100 ms/token |
| **KV cache role** | **Produces** the cache (writes K, V for all `N` tokens) | **Consumes + appends** to the cache (reads all, writes 1) |
| **Memory pressure** | Activations + weights | KV cache + weights (KV often >> activations) |
| **Preferred hardware** | High-FLOP GPUs (H100, H200, B200) | High-bandwidth GPUs; can use cheaper/older silicon cost-effectively (per Splitwise [9]) |
| **Preferred parallelism** | Tensor Parallelism (max FLOPs on one device) | Pipeline Parallelism or smaller TP (more KV headroom, more batch slots) |
| **Quantization that helps most** | Weight + activation (FP8, INT8 SmoothQuant) — doubles Tensor Core throughput | Weight-only (INT4 AWQ/GPTQ) + KV cache quantization (FP8/INT4) — cuts bandwidth |
| **Batching strategy** | Chunked prefill (Sarathi-Serve) to bound TBT spikes | Continuous batching (Orca) — many sequences sharing weight reads |
| **What makes it faster** | More FLOPs/sec, FP8 Tensor Cores, FlashAttention | Larger batch, smaller KV cache (MLA, GQA, quantization), faster HBM, speculative decoding |
| **Real-world time share** | Spiky; can be seconds for long prompts | Dominates wall-clock — up to 70% of GPU time in production GPT-4 traces [9] |

**The takeaway:** Prefill and decode are *qualitatively different workloads* that happen to share weights. They want different hardware, different parallelism, different quantization, different batching, and they have different SLOs. Colocating them on the same GPU is the root cause of most serving inefficiency, which is why **disaggregation** (§1.4) has become the dominant architectural pattern at scale.

### 1.2 Prefill: Why It Is Compute-Bound

Prefill takes an `N`-token prompt and runs **one** forward pass to produce the first output token and the full KV cache. For a transformer with hidden size `d`, head dimension `d_h`, `L` layers, the dominant operations per layer are:

| Operation | FLOPs | Bytes loaded (weights only) |
|-----------|-------|-----------------------------|
| QKV projection | `6·N·d²` | `3·d²·s` (s = bytes per weight) |
| Attention (Q·Kᵀ, softmax·V) | `4·N²·d` | (KV streamed) |
| Output projection | `2·N·d²` | `d²·s` |
| FFN (w/ GLU, 2.66× expansion typical) | `~16·N·d²` | `~3·d²·s` |

The **arithmetic intensity** (FLOPs/byte) per layer is approximately `FLOPs / weight_bytes ≈ N · (constant)`. For `N` of even a few hundred tokens this comfortably exceeds the H100's machine balance of ~295 FLOPs/byte (989 TF FP16 ÷ 3.35 TB/s HBM3). The kernel saturates Tensor Cores. As the General Compute analysis frames it: *"On an H100, a prefill pass on a 2K prompt runs at something like 800 tokens per millisecond of compute, because the work is dense. … It is compute-bound, and the bottleneck is FLOPs"* [33].

Empirically, FlashAttention-3 reports H100 FP16 utilization of **75% (~740 TFLOPs/s)** and FP8 utilization approaching **1.2 PFLOPs/s** [16] — i.e., a well-tuned prefill kernel is GEMM-limited, not memory-limited.

**Implication for serving:** Prefill latency scales O(N²) in compute but only O(N) in memory. Prefill on a long prompt is a bursty, multi-second compute spike. Batching prefills with each other yields little marginal speedup once a single prefill already saturates the device.

### 1.3 Decode: Why It Is Memory-Bandwidth Bound

Each decode step processes exactly **one** new token. The QKV projections for that single token are tiny GEMVs. The attention step is `q · K_cache^T` and `softmax · V_cache` — both must load the *entire* KV cache from HBM. For a batched decode step of batch size `B`, the arithmetic intensity is approximately:

```
AI ≈ (2 · B · L · cache_tokens · d_h) FLOPs
    / (2 · L · cache_tokens · d_h · s) bytes   ≈  B / s
```

For `s = 2` bytes (FP16), AI ≈ B/2 FLOPs/byte. To saturate H100 tensor cores (AI ≈ 295), you would need batch size ~590 — typically infeasible because KV cache memory caps the live batch. As the Hao Zhang OSDI 2024 tutorial puts it explicitly: *"Decode: s = 1, GEMM degenerates to GEMV"* [5]. Decode throughput is therefore **HBM-bandwidth bound**; it scales nearly linearly with batch size up to the KV-cache capacity ceiling [33].

A concrete corollary, from Splitwise's characterization on production GPT-4 traces: *"Up to 70% of the time is spent processing less than 4 active tokens. … Upto 70% of the time is spent exclusively in token generation phase"* [9]. Decode dominates wall-clock GPU time in real workloads, yet leaves Tensor Cores nearly idle.

### 1.4 Why Disaggregating Prefill and Decode Matters

Colocating both phases on the same GPU creates four pathologies, exhaustively characterized in DistServe and Splitwise:

1. **Prefill-decode interference.** A burst of prefills monopolizes Tensor Cores, spiking TBT for in-flight decodes — *"strong prefill-decoding interferences"* [1].
2. **Coupled parallelism plans.** Prefill prefers tensor parallelism (one device, max FLOPs); decode prefers pipeline or smaller TP (more KV-cache headroom). A colocated deployment cannot satisfy both [1].
3. **Hardware-cost mismatch.** Splitwise's A100 vs H100 measurement: H100/A100 ratios are 3.43× TFLOPs, 1.64× bandwidth, 1.75× power, 2.16× cost. Prompt throughput improves 2.2× on H100; token throughput only 1.42×, *and energy per token is 1.2× worse* [9]. The conclusion: decode on cheaper, lower-power, older silicon is *more cost-efficient*.
4. **Conflicting SLOs.** TTFT wants small prefill batches; TPOT wants large decode batches. The colocated scheduler cannot satisfy both simultaneously [1][3].

The disaggregation answer (DistServe, Splitwise, Mooncake): split the cluster into a **prefill pool** and a **decode pool**, transfer KV cache over the back-plane interconnect (NVLink within node; InfiniBand/RoCE across nodes), and size each pool independently against its SLO.

Reported results:

| System | Hardware | Headline number |
|--------|----------|-----------------|
| **DistServe** [1] | A100-80G | **7.4× more requests** or **12.6× tighter SLO** vs colocated baselines, >90% requests within SLO |
| **Splitwise** [7][9] | A100 / H100 mix | **1.4× throughput at 20% lower cost**, OR **2.35× throughput at same cost+power** |
| **Mooncake** [3] | A800 / H800 / H200 | **115% (A800) and 107% (H800) more requests** in production; **224k tok/s prefill, 288k tok/s decode** for Kimi K2 on 128 H200 with PD + large-scale EP [5] |

The KV transfer is the only real cost. For Llama-3 70B with GQA, a 32K-prompt KV cache is roughly 10 GB FP16; at 400 Gbps RDMA that is ~200 ms one-way — too slow to be hidden behind prefill. Splitwise transfers **layer-wise** (overlapping transfer of layer L's KV with computation of layer L+1) using MSCCL++ over InfiniBand [9]; Mooncake takes the same idea further with GPUDirect RDMA up to **8×400 Gbps** and zero-copy multi-NIC striping [3][6].

---

## PART 2: Optimization Techniques

### 2.1 KV Cache Management

The KV cache is the dominant memory consumer in decode. For Llama-2 70B (FP16, GQA-8) with batch 32 at 8K context, the cache is roughly `2 · 80 layers · 8 KV-heads · 128 dim · 32 batch · 8192 tokens · 2 bytes ≈ 86 GB` — more than a single H100. Every saving here is multiplicative with batch size and context length.

#### 2.1.1 PagedAttention (vLLM)

vLLM's PagedAttention (Kwon et al., SOSP 2023) [10] borrows OS virtual memory: divide the logical KV cache into fixed-size **blocks** (typically 16 tokens), and maintain a block table mapping logical-to-physical addresses. New tokens allocate blocks on demand. The result:
- **No internal fragmentation.** Pre-allocating `max_tokens` per request (the FasterTransformer approach) wastes most of HBM. vLLM reports **24× higher throughput than HuggingFace Transformers** on the same hardware [10].
- **Copy-on-write for shared prefixes.** Beam search, parallel sampling, and shared system prompts can share physical blocks until they diverge.

Block size is a tuning lever: smaller blocks reduce fragmentation but increase per-block bookkeeping cost. 16 is the production default in most engines.

#### 2.1.2 RadixAttention (SGLang)

PagedAttention reuses *within* a request. **RadixAttention** (Zheng et al., NeurIPS 2024) [11][12] reuses *across* requests. KV blocks are indexed in a **radix tree** keyed on token sequences. On a new request, the scheduler walks the tree, identifies the longest matching prefix (system prompt + few-shot exemplars + RAG context), and reuses those blocks — paying prefill cost only for the suffix:

> *"When picking the next request from the queue, prefer requests rooted at the same branch as the current running set. This keeps the hot branch pinned."* [4]

Eviction is **LRU at the branch level** rather than block level, so the tree shape and the cache shape stay aligned. Reported impact on Llama-3.1 8B with ShareGPT 1K prompts: SGLang ~16,200 tok/s vs vLLM ~12,500 (~29% edge). On prefix-heavy RAG workloads the advantage reaches **6.4×**; on voice-cloning workloads cache hit rate cleared **86.4%** [4][11][12].

The gotcha worth knowing: **prefix ordering must be consistent**. If the upstream agent layer randomizes the order of "system prompt | tools | few-shot | question", the radix structure cannot find common prefixes. Production deployments enforce canonical prompt ordering.

#### 2.1.3 KV Cache Quantization

KV cache dominates HBM in decode. Quantizing it from FP16 to FP8/INT8/INT4 directly buys bandwidth and capacity:
- **FP8 KV** (Hopper, Blackwell): straightforward — same numerical envelope as FP16 with half the bandwidth. Used by TensorRT-LLM and FlashAttention-3 [16][24].
- **INT8 KV** with per-token / per-channel scales: standard in TensorRT-LLM. Typically <0.5 perplexity delta.
- **INT4 KV** (e.g., Atom, KIVI): aggressive; requires per-group scales and outlier protection. Usable on long contexts where 2× more cache often outweighs small accuracy loss.

#### 2.1.4 Eviction: H2O, StreamingLLM

When context can grow without bound (agentic loops, streaming chat), you cannot store everything. Two techniques solve this:

- **H2O (Heavy Hitter Oracle)** (Zhang et al., NeurIPS 2023) [22]: observes that *"a small portion of tokens contributes most of the value when computing attention scores"*. These "heavy hitters" emerge naturally from token co-occurrence. H2O keeps the heavy hitters and a sliding window of recent tokens; reported: **20% heavy hitters reproduce baseline quality** while reducing memory dramatically, yielding up to **29× throughput vs DeepSpeed Zero-Inference, 3× vs FlexGen, 1.9× lower latency** [22].
- **StreamingLLM**: the discovery of **attention sinks** — the first 4 tokens of any sequence absorb large attention mass simply as a softmax numerical anchor. Naïve sliding-window eviction collapses perplexity because evicting position 0 breaks softmax stability. StreamingLLM keeps the first ~4 tokens (sinks) + the last `W` tokens, enabling "infinite" streaming [23].

Pick H2O for RAG/long-document workloads where the relevant content is buried mid-context; pick StreamingLLM for chat/agent streams where only recency matters [23].

#### 2.1.5 Cross-instance KV reuse (Mooncake Store, LMCache)

Mooncake elevates the KV cache to a first-class **distributed store**. CPU DRAM, NVMe SSDs, and RDMA NICs across the GPU cluster are pooled into a multi-tier KV cache (HBM → DRAM → SSD), with the Transfer Engine moving KV blocks between tiers via zero-copy GPUDirect RDMA and multi-NIC striping [3][6]:

> *"Mooncake utilizes (GPUDirect) RDMA technology to transfer data directly from the initiator's DRAM/VRAM to the target's DRAM/VRAM in a zero-copy manner, while maximizing the use of multi-NIC resources on a single machine."* [6]

The principle is stated bluntly in the FAST 2025 paper title: *"Trading More Storage for Less Computation."* The cache hit on a shared prefix avoids re-running prefill — a cubic-savings trade for tokens that would otherwise be recomputed across many requests [3].

### 2.2 Batching Strategies

#### 2.2.1 From Static to Continuous Batching (Orca)

Pre-Orca serving (Triton+FasterTransformer style) used **request-level static batching**: the scheduler grouped requests, fixed the batch, ran until all finished, then returned. Two pathologies:
- Faster requests waited for slower ones (head-of-line blocking).
- Incoming requests waited for the next batch boundary.

**Orca** (Yu et al., OSDI 2022) [14] introduced **iteration-level scheduling**: the scheduler interacts with the engine at the granularity of *one decode iteration*. New requests can join mid-batch; completed requests leave immediately. The engine also performs **selective batching**: ops with batch-independent semantics (LayerNorm, FFN) are batched normally; attention is unbatched (per-request) because each request has a different cache length. Reported: **36.9× higher throughput than NVIDIA FasterTransformer** at equivalent latency on GPT-3 175B [14].

This pattern — also called *continuous batching* or *in-flight batching* — is the foundation of every modern engine: vLLM, TensorRT-LLM ("inflight batching" [24][25]), SGLang, TGI, Friendli, LMDeploy.

#### 2.2.2 Chunked Prefill (Sarathi-Serve)

Continuous batching fixes the *who's in the batch* problem but not the *prefill stalls decode* problem. A long prefill, when scheduled, blocks all decodes in the batch for its duration — causing tail-latency spikes for TBT.

**Sarathi-Serve** (Agrawal et al., OSDI 2024) [13] introduces two co-designed techniques:
- **Chunked prefill**: split a long prefill into fixed-size chunks (e.g., 512 tokens). One chunk is processed per iteration.
- **Stall-free batching**: at every iteration, pack the available compute budget with *both* decode tokens and one prefill chunk. The decode tokens piggyback on the prefill compute almost for free, because the prefill chunk already needs to load the same weights.

The combined effect is that *every* iteration's arithmetic intensity stays high (compute-bound, like prefill) while none of the in-flight decodes stalls for long. Sarathi-Serve thus closes the gap between maximum throughput and predictable TBT — *"taming the throughput-latency tradeoff."* It is the production scheduler in TensorRT-LLM, vLLM (>= v0.5), and SGLang [13][24].

### 2.3 Quantization

#### 2.3.1 Weight-only vs Weight+Activation

| Class | Examples | What's quantized | What's not | Typical bit width | Use case |
|-------|----------|------------------|------------|-------------------|----------|
| **Weight-only** | GPTQ, AWQ, GGUF, Marlin | Weights only | Activations stay FP16/BF16 | INT4 (W4A16), INT3 | Decode (bandwidth-bound; loading weights is the cost) |
| **Weight + Activation** | SmoothQuant, FP8, INT8 SQ | Both | — | FP8 (E4M3/E5M2), INT8 | Prefill (compute-bound; tensor-core INT8/FP8 wins) |

The asymmetry is structural. Decode loads each weight once and computes very little — so shrinking *weights* (W4) buys 4× bandwidth and is essentially free in compute. Prefill is compute-bound — so shrinking *activations* too (A8/FP8) doubles Tensor Core throughput.

#### 2.3.2 GPTQ, AWQ, SmoothQuant

- **GPTQ** (Frantar et al.): layer-wise weight quantization that compensates for rounding error using approximate second-order (Hessian-based) information. Strong at INT4 weight-only but expensive to calibrate.
- **AWQ** (Lin et al., MLSys 2024 Best Paper) [30]: *"protecting only 1% salient weights can greatly reduce quantization error. To identify salient weight channels, we should refer to the activation distribution, not weights."* AWQ scales salient channels up before quantization (an equivalent transformation), avoiding mixed-precision storage. Reported: **1.45× faster than GPTQ kernels, 1.85× faster than cuBLAS FP16** at INT4 [30]. Now ubiquitous in TensorRT-LLM, vLLM, TGI, LMDeploy.
- **SmoothQuant**: handles the W8A8 problem by migrating activation outlier magnitudes into weight scales. Critical because activation outliers in attention projections (a few channels carry 100× the magnitude of the median) defeat per-tensor quantization otherwise.

#### 2.3.3 FP8 and FP4 on Hopper/Blackwell

Hopper introduced FP8 with E4M3 (range) and E5M2 (precision) variants; Blackwell adds FP4. Native Tensor Core support gives FP8 ~2× the throughput of FP16 and FP4 ~4×. The challenge is keeping numerical error bounded:
- DeepSeek-V3 trains end-to-end in FP8 with per-128-channel scaling, accumulation in BF16, and statistics-driven scale calibration [17].
- FlashAttention-3 uses **block quantization** (per-block scale) plus **incoherent processing** (Hadamard rotation) to break up outlier patterns, achieving FP8 attention with *"2.6× lower numerical error than a baseline FP8 attention"* [16].

#### 2.3.4 Granularity

| Granularity | Definition | Pros | Cons |
|-------------|-----------|------|------|
| Per-tensor | One scale per tensor | Cheapest, fastest | Outliers ruin it |
| Per-channel | One scale per row/col | Better for outliers in one dim | Modest cost |
| Per-group | One scale per (e.g.) 128 contiguous elements | Robust to scattered outliers | More metadata, more scale loads |

INT4 W4A16 deployments universally use per-group (group sizes 64-128) because INT4's dynamic range cannot tolerate per-tensor.

#### 2.3.5 Microsoft contributions

DeepSpeed-Inference (ZeRO-Inference) pioneered offloading + tensor parallelism for inference; **ZeroQuant** introduced fine-grained per-group quantization with knowledge distillation recovery; later **ZeroQuant-FP** and **FP6-LLM** showed FP6 (six-bit floating) hits a sweet spot between INT4 quality loss and FP8 cost. These flow into Azure AI Foundry's hosted endpoints.

### 2.4 Speculative Decoding

The original insight (Leviathan et al., ICML 2023) [31]: decode is memory-bound, not compute-bound, so the H100 has spare FLOPs every step. Use those FLOPs to verify a *batch* of candidate tokens drafted by a cheap model:

```
1. Draft model proposes k tokens (cheap, sequential).
2. Target model runs ONE forward pass over those k tokens (parallel, expensive).
3. Verify each draft token via rejection sampling that preserves the target distribution.
4. Accepted tokens are free; the first rejected token is regenerated by the target.
```

Reported 2-3× wall-clock speedup on T5-XXL [31]. Crucially, **the target distribution is unchanged** — outputs are bit-identical (in expectation) to standard decoding.

Variants:
- **Medusa**: adds multiple "Medusa heads" on top of the target model that predict tokens at positions +1, +2, +3 in parallel. Removes the need for a separate draft model. Tree-based verification.
- **EAGLE** (Li et al.) [32]: drafts in *feature space* (the second-to-top-layer hidden states), which is more predictable than token space. Reported **2.7–3.5× speedup on LLaMA2-Chat 70B with doubled throughput** while preserving distribution.
- **Lookahead Decoding** (Fu et al.): no draft model at all; uses Jacobi iteration on n-grams.
- **MTP (Multi-Token Prediction)**: DeepSeek-V3 trains the model to predict multiple tokens per step. Reported 85–90% acceptance rate on the second token [17].

**When it helps / hurts:**
- *Helps:* low batch size (memory-bound regime), predictable code/math/structured outputs, high draft acceptance rate.
- *Hurts:* large batch (already compute-bound — extra verification work is not free), highly diverse outputs, where draft acceptance falls below ~50%.

The breakeven is approximately when `(acceptance_rate · target_step_time) / (draft_step_time + target_step_time / k)` exceeds 1. At batch 1 this is almost always satisfied; at batch 256 rarely is.

### 2.5 Attention Optimizations

#### 2.5.1 FlashAttention 1 → 2 → 3

The bottleneck in classical attention is *materializing* the `N×N` attention matrix in HBM. **FlashAttention-1** (Dao et al., NeurIPS 2022) [15] eliminates this with **IO-aware tiling** plus **online softmax**:
- Tile Q, K, V into blocks sized for SRAM (~20 MB on A100/H100).
- Compute attention per block-pair in SRAM.
- Use the online softmax algorithm (Milakov & Gimelshein) to compose softmax across blocks via running max/sum statistics — mathematically equivalent to a full softmax.
- HBM accesses drop from `O(N² + Nd)` to `O(N²d²/M)` ≈ `O(Nd)` when SRAM is large enough.

Reported: 3× speedup on GPT-2 attention, enabled 16K and 64K context Path-X / Path-256 benchmarks for the first time [15].

**FlashAttention-2** improved parallelism (split work along Q rather than K) and reduced non-matmul FLOPs, reaching ~50–73% utilization on A100.

**FlashAttention-3** (Shah et al., NeurIPS 2024) [16] is rewritten for Hopper specifically:
1. **Producer-consumer warp specialization**: separate warps issue TMA loads (producer) and consume them with WGMMA (consumer). Hides memory latency under compute.
2. **GEMM-softmax pipelining**: 2-stage pipeline overlaps softmax (low-throughput non-matmul) with the next block's matmul.
3. **FP8 block quantization + incoherent processing**: enables FP8 attention with 2.6× lower error than per-tensor FP8.

Reported: **1.5–2.0× speedup over FA2, ~740 TFLOPs/s FP16 (75% utilization), close to 1.2 PFLOPs/s in FP8** on H100 [16].

#### 2.5.2 MQA → GQA → MLA

The KV cache scales with the number of KV heads. Reducing them is the most direct way to shrink the cache:

- **MQA (Multi-Query Attention)**: one KV head shared across all Q heads. Maximum cache reduction (`H_q ×`) but quality drops on some tasks.
- **GQA (Grouped-Query Attention)**: groups of Q heads share one KV head. Llama 2/3 use GQA-8 (8 Q heads per KV head). KV cache reduced 8× with minimal quality loss. The dominant production choice.
- **MLA (Multi-Head Latent Attention)** [17][18]: introduced in DeepSeek-V2/V3. Instead of reducing KV-head count, project K/V down into a *latent* of dimension `kv_lora_rank=512` (for the 671B model, vs full `n_heads × qk_head_dim = 128 × 192 = 24,576` per token) and cache only the latent. On the decode path, an **"absorb"** mode fuses the up-projection into Q, so the cached latent is consumed directly. Reported: **4–8× less KV memory vs standard MHA** with quality comparable to or slightly above full MHA [18]. RoPE is handled by reserving a small slice (`qk_rope_head_dim=64`) that is rotated normally; the rest is rotation-invariant ("NoPE").

MLA is the reason DeepSeek-V3 and Kimi-K2 can fit long contexts on commodity-ish HBM. The cost is implementation complexity — the attention kernel is no longer a drop-in for FlashAttention's prefill kernel and must be written specifically.

#### 2.5.3 Sliding-Window and Sparse Attention

For very long contexts, full attention is `O(N²)`. Mistral popularized **sliding-window attention** (each token attends only to the previous `W=4096`). Layered stacks effectively gain a much larger receptive field (`L × W`). Combined with attention sinks, this gives bounded-memory streaming. TensorRT-LLM and vLLM both support sliding-window + chunked attention natively [24].

Recent **sparse attention** approaches (DeepSeek Sparse Attention, Rocket Sparse Attention) selectively prune attention to top-k key tokens at decode time, trading quality for very long contexts. Production use is still emerging.

### 2.6 Parallelism Strategies

A 671B-parameter MoE model does not fit on any single GPU. Distributed parallelism slices the model multiple ways:

#### 2.6.1 Tensor Parallelism (Megatron)

Megatron-LM (Shoeybi et al., 2019) [20] partitions each layer's matmul across `TP` GPUs:
- MLP: shard A column-wise, B row-wise. Two matmuls per layer; only one `all-reduce` after the second.
- Attention: shard QKV heads across GPUs; output projection row-sharded; one `all-reduce` per attention block.

The communication cost is **2 all-reduces per layer per forward pass**, each on the hidden dimension. For TP=8 on NVLink (900 GB/s NVL4, 1.8 TB/s NVL5) this is nearly free *inside a node*. **Across nodes**, even 400 Gbps InfiniBand is ~20× slower than NVLink — so TP rarely scales past one node in production.

Reported: 76% scaling efficiency at 512 GPUs (2019 paper); 502 PFLOP/s on 3072 GPUs for a 1T-param model in the 2021 follow-up [21].

#### 2.6.2 Pipeline Parallelism

Pipeline parallelism (PP) partitions *layers* across GPUs. Microbatches stream through the pipeline. The classical issue is the **pipeline bubble**: the fill-and-drain phases where some GPUs are idle. Interleaved 1F1B schedules (GPipe, PipeDream, Megatron interleaved) shrink the bubble to `~(p-1)/m` where `p` is pipeline depth and `m` is the microbatch count.

For inference, pipeline parallelism trades latency for capacity: each request must traverse the entire pipeline serially, so TPOT is worse than TP. PP is mostly used for *very* large models that don't fit in any single-node TP arrangement (e.g., DeepSeek-V3 inference uses TP within node + PP across).

#### 2.6.3 Expert Parallelism (MoE)

Mixture-of-Experts models activate only `k` (typically 1–8) of `E` (32–256) experts per token. **Expert parallelism** shards the experts across GPUs. Each token must be routed to its assigned expert's GPU — an `all-to-all` communication on the token dimension.

DeepSeek-V3's inference architecture is the most extreme published example [19]:

> *"Our solution employs cross-node Expert Parallelism (EP). First, EP significantly scales the batch size, enhancing GPU matrix computation efficiency and boosting throughput. Second, EP distributes experts across GPUs, with each GPU processing only a small subset of experts (reducing memory access demands)."*

Kimi K2 (1T-parameter MoE) is deployed on 128 H200s with PD disaggregation + large-scale EP, achieving 224k tokens/sec prefill and 288k tokens/sec decode [5]. The all-to-all is the bottleneck; DeepSeek developed **DeepEP** to overlap dispatching and compute.

#### 2.6.4 Sequence / Context Parallelism (Ring Attention)

For million-token contexts, the activations of even a single layer don't fit on one GPU. **Ring Attention** (Liu, Zaharia, Abbeel) [26] partitions the *sequence dimension* across GPUs and computes blockwise attention in a ring:

> *"By performing self-attention and feedforward network computations in a blockwise fashion, we can distribute sequence dimensions across multiple devices, allowing concurrent computation and communication. … The results are invariant to the ordering of these blockwise computations."* [26]

Each GPU holds one slice of Q/K/V; KV blocks circulate around the ring while each GPU computes the local QK and accumulates online-softmax statistics. The KV communication overlaps perfectly with compute, so it adds no end-to-end latency. Enables training/inference of sequences `device_count` times longer than any single device can hold — million-token contexts on practical hardware [26].

In production, "context parallelism" (CP) in Megatron and "Helix Parallelism" in TensorRT-LLM are direct descendants [20][24].

### 2.7 Kernel-Level Optimizations

| Technique | What it does | Production impact |
|-----------|--------------|-------------------|
| **CUDA Graphs** | Record an iteration's launch sequence once; replay with one CPU-side launch | Removes per-kernel launch overhead (~10 µs each × 100s of kernels per layer). Essential for decode |
| **Fused kernels** | LayerNorm + QKV proj, RMSNorm + RoPE, attention output + residual, MLP gate + up + SiLU + down | 1.3–2× per-block speedup; reduces SMEM/HBM traffic |
| **Custom attention kernels** | FlashAttention variants for MQA/GQA/MLA, sliding window, paged KV | Engine-specific; vLLM has FlashInfer; TensorRT-LLM has trtllm-attention |
| **torch.compile** | TorchInductor traces + fuses + emits Triton | Production decode at PyTorch-eager parity often within 10% of hand-tuned. Now default in vLLM v1 |
| **TensorRT engine build** | Ahead-of-time graph compilation, kernel autotuning per GPU/SM count, weight stripping | Best raw throughput on NVIDIA; cost is a longer build step and per-shape rebuild |
| **CUTLASS / cuBLASLt custom GEMMs** | SM-tile-aware GEMM autotuned for batch/seq/hidden shape | Especially important for small-batch decode shapes where stock GEMMs leave 50% on the table |

The Marlin kernel (W4A16 INT4) family is a representative example: hand-written Hopper SASS that achieves ~95% of memory bandwidth on INT4 GEMM — the foundation of fast AWQ/GPTQ decode in vLLM, TensorRT-LLM, and SGLang [30].

---

## PART 3: Scaling to Millions of Concurrent Users

### 3.1 Serving Stack Architecture

| Engine | Strengths | Weaknesses | Best for |
|--------|-----------|-----------|----------|
| **vLLM** | PagedAttention reference impl; widest model coverage; fastest community velocity; v1 architecture has near-zero scheduler overhead | Smaller absolute throughput ceiling than TRT-LLM; fewer enterprise features (multi-tenant LoRA, auth) | Generic high-throughput serving; multi-model fleets; FP8/INT4 quants |
| **TensorRT-LLM** | Hand-tuned Hopper/Blackwell kernels; AOT engine builds; richest feature set (MTP, EAGLE3, MLA, Helix, sparse attention, MoE EP) [24] | Build step required per (model, GPU SKU, max-shape); steeper ops curve | NVIDIA-only deployments where absolute throughput is the goal (NIM, Triton-LLM) |
| **SGLang** | RadixAttention (prefix cache wins ~6.4× on RAG); programmatic frontend for structured LLM programs; native PD disagg | Newer codebase; some models lag vLLM | Agentic / RAG workloads with heavy prefix sharing |
| **TGI** (HF) | First-class HF model integration; production-grade auth/metrics | Throughput now trails vLLM/SGLang on most benchmarks | HF-native deployments |
| **NVIDIA Dynamo** | KV-cache-aware *router* + scheduler over heterogeneous worker pools; native PD disaggregation; backend-agnostic (TRT-LLM, vLLM) [27] | Orchestration layer, not an engine | Multi-replica clusters needing prefix-aware routing |
| **NVIDIA Triton Inference Server** | Generic model serving; ensembles; gRPC/HTTP/Kubernetes-native | LLM-specific features layered on TRT-LLM backend | Mixed AI workloads beyond LLMs |
| **Ray Serve** | Python-native orchestration; autoscaling; placement groups | Engine-agnostic; you bring vLLM or TRT-LLM | Multi-tenant ML platforms with Python control plane |

**Azure-specific patterns** [28][29]: Azure AI Foundry now ships **NVIDIA NIM** microservices natively. NIM internally bundles TensorRT-LLM (and vLLM where TRT support lags), CUDA graphs, FP8 quants, and PagedAttention/in-flight batching. On Azure managed compute, NIM is deployed as an Online Endpoint — sized per the model card's VM SKU. The new **ND GB200 v6** SKU (Grace+Blackwell) is the high-end tier, paired with Azure Quantum InfiniBand for cross-node KV transfer. For DIY workloads, **Azure Container Apps Serverless GPUs** with NIM provide scale-to-zero. The **Azure OpenAI Service** itself is built on Microsoft Research's Splitwise [7][9] (phase-split prefill/decode pools) plus continuous batching plus FP8 — these are the publicly disclosed building blocks; the exact production topology is not public.

### 3.2 Capacity Planning: Roofline-First Sizing

The first step in sizing a fleet is the **single-GPU roofline** for the target model:

```
Compute-bound regime: latency = FLOPs / device_TFLOPs
Memory-bound regime:  latency = bytes_loaded / device_HBM_BW
Crossover at AI = device_TFLOPs / device_HBM_BW  (~295 for H100 FP16)
```

For a request with prompt length `N` and completion length `T`:

```
Prefill time      ≈ N · (model FLOPs/token) / device_TFLOPs
Decode time/token ≈ (weight_bytes + KV_bytes_in_cache) / device_HBM_BW
End-to-end        ≈ Prefill_time + T · Decode_time_per_token
TPOT_at_batch_B   ≈ Decode_time_per_token · max(1, B / B_saturate)
```

where `B_saturate ≈ device_TFLOPs · 2 / device_HBM_BW` (the batch at which decode becomes compute-bound again — for H100 FP16 this is ~150–300; FP8 doubles it).

**Sizing a fleet to QPS with SLO:**

1. Pick model, GPU SKU, quantization → derive single-instance prefill_time(N), decode_time_per_token(B).
2. Choose target batch `B*` such that `decode_time_per_token(B*) ≤ TPOT_SLO`.
3. Calculate KV cache budget: per-request KV bytes × `B*` must fit in available HBM after weights. This caps `B*`.
4. Single-instance throughput (tok/s) = `B* / decode_time_per_token(B*)`.
5. Required instances ≈ `QPS · (N + T_avg) / single_instance_throughput`, with **20–40% overhead** for traffic burstiness, prefill-decode interference (if colocated), and rolling restarts.
6. Apply Little's law to validate: `Concurrency = QPS · E2E_latency`. The fleet must hold `Concurrency` in-flight requests across all instances.

**Example: serve Llama-3 70B at 10 QPS, 4K prompt, 1K completion, TPOT < 50 ms, TTFT < 1.5 s on H100s.**
- Weights at FP8: 70 GB. Fits on 1×H100 (80 GB) with ~10 GB headroom for KV cache. GQA-8 KV per 4K tokens ≈ 1.3 GB → effective batch ~7. Too small.
- Move to TP=2 (NVLink): weights 35 GB/GPU, KV headroom 45 GB/GPU → batch ~30. Single decode step ~30 ms at batch 30. ✓ TPOT.
- Prefill on 4K tokens at FP8: ~600 ms on TP=2. ✓ TTFT.
- Single instance throughput: 30 / 0.030 = 1000 tok/s ≈ 0.8 requests/s end-to-end (with 5s per request).
- 10 QPS × 5s = 50 concurrent → ~13 instances (with 30% overhead). 26 H100s. Disaggregating prefill/decode (Splitwise-style) typically cuts this by 30–50% on real traces [7].

### 3.3 Multi-Replica and Load Balancing

Stateless LB (random / least-connections) is **always wrong** for LLM serving because of the prefix cache. The optimal strategy is **KV-cache-aware routing**:

> *"The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation."* [27]

Two scoring inputs:
- **Decoding cost**: number of active KV blocks (queue load proxy).
- **Prefill cost**: number of *new* tokens that would have to be prefilled on this worker given its current radix tree contents — i.e., `prompt_tokens - cache_hit_tokens`.

The router picks the worker minimizing a weighted sum. This is the same pattern as SGLang's cache-aware scheduler [4][12] and the Mooncake KVCache-centric scheduler [3]. Production realization: route system-prompt-X traffic to instances I₁, I₂; route system-prompt-Y to I₃, I₄. The hot prefix stays resident.

**Token-budget-aware load balancing**: instead of counting requests, count `prompt_tokens + estimated_completion_tokens`. Long-context requests consume disproportionate KV memory and time, and request-count routing oversaturates instances that get unlucky with long traffic.

### 3.4 Autoscaling

LLM autoscaling has three modes:

- **Reactive (queue + GPU util)**: scale up when scheduler queue depth × estimated TTFT exceeds threshold; scale down when GPU memory or compute utilization stays below threshold. Stock Kubernetes HPA or KEDA suffice with custom metrics.
- **Predictive (forecast-driven)**: time-series forecasts (Prophet, recent LSTMs) anticipate diurnal patterns. Useful for ~1–10 min ahead.
- **Cold start mitigation**: LLM model load times are minutes (70B FP8 ≈ 70 GB to copy from object storage, then engine warm-up). Production mitigations:
  - **Warm pools** of fully loaded but idle replicas.
  - **Model-streaming** (vLLM 0.9+, Run.ai Model Streamer): kernel-bypass parallel reads to overlap weight download with VRAM upload.
  - **Persistent shared volumes** (Azure Premium SSD v2, Lustre, Filestore) holding pre-converted engine artifacts.
  - **Scale-to-zero** is feasible only when cold start < SLO; otherwise keep a floor of `N≥1` warm replicas. Azure Container Apps serverless GPUs support this pattern but cold start is typically 30–90s [28].

### 3.5 Disaggregated Serving at Scale

The reference architecture is a three-tier control/data plane:

```
                            ┌──────────────────┐
                            │  Global Router    │
                            │  (KV-aware,       │
                            │   prefix-aware,   │
                            │   token-budget)   │
                            └─────────┬─────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
       ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
       │  Prefill Pool   │   │  Decode Pool    │   │ KVCache Store   │
       │  (H100/B200)    │   │  (A100/L40S)    │   │  HBM↔DRAM↔SSD   │
       │  Large TP       │   │  Smaller TP     │   │  RDMA fabric    │
       │  Big prefill    │   │  Big batches    │   │  Multi-NIC      │
       │  batches        │   │                 │   │  Mooncake Store │
       └────────┬────────┘   └────────▲────────┘   └────────▲────────┘
                │                     │                     │
                └─────── RDMA KV transfer (zero-copy) ──────┘
                         (NVLink intra-node, IB/RoCE inter)
```

Mooncake (Kimi production) [3] is the canonical example: a thousands-of-nodes cluster processing >100 billion tokens/day. Key engineering details:

- **Layer-wise KV transfer**: emit KV for layer `L` over the network while computing layer `L+1` on the prefill node. Splitwise does the same via MSCCL++ [9].
- **8 × 400 Gbps RDMA per node**: striped across multiple NICs, NUMA-aware. Without this bandwidth, the transfer would dominate TTFT [3].
- **Multi-tier KV cache** (HBM → DRAM → SSD): hot prefixes in HBM, warm in DRAM, cold serialized to SSD. The Transfer Engine [6] arbitrates promotion/eviction. Reported: petabyte-scale across a single Kimi cluster.
- **Prediction-based early rejection**: in overload, instead of accepting and timing out, Mooncake predicts at admission whether the SLO is achievable and rejects early [3]. This is the only published large-scale LLM serving system that admits and engineers around overload — an important pattern for any SaaS deployment.
- **xPyD**: Mooncake supports `x` prefill nodes feeding `y` decode nodes arbitrarily, not just 1:1. Decode pools can sit behind multiple prefill pools, and a single prefill can fan out KV to multiple decode nodes for replicated low-latency serving.

NVIDIA's productized version is **Dynamo + TensorRT-LLM PD-disagg** with Mooncake Transfer Engine as one of the supported transports. As of Dec 2025, Mooncake's Transfer Engine is integrated natively into both vLLM v1 (as a KV Connector) and TensorRT-LLM for PD-disaggregated inference [5].

### 3.6 Caching Layers

| Layer | What's cached | Hit-rate driver | Latency saved |
|-------|---------------|-----------------|---------------|
| **Semantic response cache** | (prompt-embedding, response) pairs in vector store | Repeat questions; canonical FAQs | Full end-to-end (10s → 50ms) |
| **Prefix KV cache (cross-user)** | System prompt + tool schemas + few-shot blocks | System-prompt churn rate | Prefill-only |
| **Embedding cache** | `(text → embedding)` for RAG | Document corpus stability | Embedding latency (~ms) |
| **Retrieval cache** | `(query → retrieved docs)` | Query repetition | Vector search + I/O |

For Microsoft customers, **Azure Cache for Redis** (with vector similarity) is the typical semantic-cache substrate; for prefix KV cache, Mooncake Store or LMCache running on the GPU cluster's CPU DRAM/SSD.

A subtle production point: semantic caching at the application layer (above the LLM) trades freshness for cost; for non-deterministic responses (creative tasks, agentic outputs) you typically *don't* cache. For structured QA over stable knowledge bases, hit rates of 30–60% are achievable, dwarfing all GPU-level optimizations.

### 3.7 Multi-Region and Geographic Distribution

Three constraints conflict: **data residency**, **TTFT latency** (cross-region adds 50–150 ms), and **GPU capacity availability**. Patterns:

- **Active-active per region**: replicate model artifacts to each region (Azure ND GB200 v6 regions today: a small subset). Azure Front Door / Traffic Manager routes by client geo. KV caches are *not* cross-region — each region has its own RadixAttention store.
- **Hub-and-spoke**: a few "hub" regions host the full disaggregated serving stack; spokes terminate clients and forward over Azure ExpressRoute. TTFT suffers; cost falls.
- **Data-residency tiers**: EU traffic to EU regions only (GDPR), with model artifacts deployed only where compliance allows.

**Cost-latency tradeoff**: H100 capacity is ~30–60% cheaper in less-constrained regions (US South Central, North Europe vs East US 2). Routing non-latency-sensitive batch traffic to cheaper regions while keeping real-time chat in expensive regions is a real cost lever — Azure customers commonly run two endpoints with different SLO tiers.

### 3.8 Real-World Reference Architectures

| Operator | Public details |
|----------|----------------|
| **OpenAI / Azure OpenAI Service** | Built on Triton + TensorRT-LLM (per public NVIDIA case studies). The Splitwise paper [7] explicitly characterizes "production clusters running GPT-4" traces (Bing, GitHub) and validates phase-split deployments on H100/A100. Splitwise is "implemented on vLLM" per the OCP slides [9] — i.e., the Azure OpenAI fleet currently runs phase-split disaggregation with KV transfer over MSCCL++ on InfiniBand back-planes. |
| **Anthropic** | Limited public detail. Public engineering posts confirm proprietary inference engine, FlashAttention-derived kernels, custom batching scheduler. Cross-region active-active with prefix routing for system-prompt and tool-spec reuse. |
| **Google (Gemini)** | TPU-based, custom XLA compiler stack. Speculative decoding (Leviathan et al. [31] originated at Google), MoE, MQA/GQA. Pathways execution layer handles disaggregation at the TPU pod level. |
| **Moonshot (Kimi K1.5/K2)** | Mooncake [3], published in full. PD disagg, KVCache-centric scheduler, 100B+ tokens/day. Kimi K2 (1T MoE) on 128 H200s = 224k/288k tok/s prefill/decode [5]. |
| **DeepSeek (V3, R1)** | Open-sourced infrastructure (Open Infra Index Day 6) [19]. Cross-node Expert Parallelism, MLA, FP8 training and inference, DeepEP for all-to-all overlap. 671B/37B MoE. |
| **xAI (Grok)** | Reportedly SGLang-based [4]; 100K+ H100 cluster ("Colossus"). Limited details public. |

The convergence across operators is striking: by 2026, every frontier serving stack is some combination of **PD disaggregation + RadixAttention-style prefix reuse + chunked/continuous batching + FP8 + GQA-or-MLA + speculative decoding** running over **RDMA fabrics with multi-tier KV cache**.

---

## PART 4: Worked Example — Serving a 405B Model for 1,000 Concurrent Users on Kubernetes

This section walks the full sizing exercise for **Llama 3.1 405B** (the canonical open 405B-class model; the same math applies to DeepSeek-V3 671B at ~37B active, Mistral Large 2, or a hypothetical Phi-4 405B) serving **1,000 concurrent users** on a Kubernetes-managed GPU cluster.

### 4.1 Workload Definition

"1,000 concurrent users" is ambiguous; the sizing changes by >5× depending on which interpretation. Pick one explicitly:

| Interpretation | Active inflight requests at peak | Typical use case |
|----------------|-----------------------------------|------------------|
| 1,000 users *signed in* | ~50–100 inflight | Internal copilot (1 message every 30–60s) |
| 1,000 users *actively chatting* | ~150–250 inflight | Customer support, coding assistant |
| 1,000 *concurrent inflight requests* | 1,000 inflight | Bulk RAG, batch summarization |

For this worked example we assume the middle case — **a chat-style copilot with ~200 average inflight requests at peak** (think load + 25–30% headroom). Add the workload contract:

| Parameter | Value |
|-----------|-------|
| Avg prompt length | 1,500 tokens (system + RAG + user) |
| Avg output length | 400 tokens |
| Peak inflight requests | 200 |
| Required aggregate decode throughput | 200 × 30 tok/s = **6,000 tok/s** |
| TTFT SLO (p95) | <2.0 s |
| TPOT SLO (p95) | <50 ms (i.e., ≥20 tok/s per user) |
| Availability SLO | 99.9% |

### 4.2 Model Memory Footprint

Llama 3.1 405B uses **GQA-8** (128 query heads, 16 KV heads), `d_head = 128`, `L = 126` layers, hidden dim 16,384.

**Weight footprint by precision:**

| Precision | Weights | Notes |
|-----------|---------|-------|
| BF16 / FP16 | **810 GB** | Reference quality; baseline |
| FP8 (E4M3) | **405 GB** | ~0.1–0.3 pp accuracy delta; Hopper/Blackwell native |
| INT8 (SmoothQuant) | ~405 GB | Comparable to FP8 |
| INT4 (AWQ / GPTQ) | **~210 GB** | ~0.3–0.8 pp delta; weight-only, FP16 activations |

**KV cache per token (FP16):**
`2 × L × n_kv_heads × d_head × bytes = 2 × 126 × 16 × 128 × 2 = 1,032 KB ≈ 1 MB per token.`

Quantized: FP8 KV → 512 KB/token; INT4 KV → 256 KB/token.

For our workload, a request at peak holds (1500 + 400) × 1 MB ≈ **1.9 GB of KV cache**. With 200 inflight requests across the fleet, total cache pressure is **~380 GB FP16** or **~190 GB FP8**.

### 4.3 Hardware Selection

A 405B model in FP8 (405 GB weights) does **not fit on a single 8×H100-80GB node** once KV cache and activations are added — 640 GB total HBM minus 405 GB weights leaves only 235 GB for KV + activations + workspace, capping live batch hard. Practical choices:

| Per-replica config | Aggregate HBM | Headroom after FP8 weights | Verdict for 405B |
|---|---|---|---|
| 8 × H100 80GB | 640 GB | 235 GB | Tight; small batch only |
| 8 × H200 141GB | **1,128 GB** | 723 GB | **Recommended (FP8)** |
| 8 × B200 192GB | 1,536 GB | 1,131 GB | Future-proof; lower TP latency |
| 8 × MI300X 192GB | 1,536 GB | 1,131 GB | AMD path; ROCm + vLLM |
| 16 × H100 80GB (2-node TP) | 1,280 GB | 875 GB | Works in BF16 with cross-node NVLink/IB |

**Pick: 8 × H200 with TP=8 per replica, FP8 weights, FP8 KV cache.** This avoids cross-node tensor parallelism (which adds 2–4× per-layer all-reduce latency over IB vs intra-node NVLink/NVSwitch) while giving ~720 GB of HBM for KV cache — enough for ~700 concurrent decode slots per replica.

### 4.4 Throughput per Replica

Published numbers (vLLM, TensorRT-LLM benchmarks, and Mooncake's reported figures for similar-scale models, extrapolated):

| Phase | Per-replica capacity (8×H200, FP8, TP=8) |
|-------|------------------------------------------|
| Prefill (1.5K-token prompts) | ~25,000–35,000 prefill tok/s |
| Decode (continuous batching, batch ~200) | ~4,000–6,000 decode tok/s aggregate |
| Per-user decode rate at batch=200 | ~25–35 tok/s |

These are realistic ballparks for a tuned vLLM v1 / TensorRT-LLM deployment with FP8 + chunked prefill + PagedAttention. The decode number is the binding constraint.

### 4.5 Cluster Sizing — Number of Replicas

For 6,000 tok/s aggregate decode demand and ~5,000 tok/s per replica → **2 replicas minimum**. Add headroom:

| Factor | Multiplier | Cumulative replicas |
|--------|-----------|---------------------|
| Base demand (200 inflight × 30 tok/s) | 2 | 2 |
| Burst headroom (1.3×) | 1.3× | 3 |
| N+1 redundancy per AZ | +1 | 4 |
| Rolling upgrades (drain 1 at a time) | +1 | 5 |
| **Total per region** | | **5 replicas** |

**Per region: 5 × 8 H200 = 40 H200 GPUs.** For two-region active-active (typical 99.9% availability target): **80 H200 GPUs total**.

#### Optional: Disaggregated PD topology

If TTFT is tight (Splitwise-style separation):

| Pool | Replicas | Hardware | Purpose |
|------|---------|----------|---------|
| Prefill pool | 2 × 8×B200 | High FLOPs, FP8 | Long-prompt burst absorption |
| Decode pool | 4 × 8×H200 | High HBM bandwidth | Sustained TPOT |
| KV transfer | 8×400 Gbps IB | GPUDirect RDMA | Layer-wise overlap |

Splitwise reports **1.4× throughput at 20% lower cost** vs colocated on production Azure GPT-4 traces [9]; for a 1,000-user workload this saves roughly the cost of 1–2 replicas.

### 4.6 Software Stack

| Layer | Recommended | Alternatives | Why |
|-------|-------------|--------------|-----|
| **Inference engine** | **vLLM v1** or **TensorRT-LLM** | SGLang, TGI, LMDeploy | PagedAttention + chunked prefill + FP8 + continuous batching; vLLM v1 has best community support, TRT-LLM has best raw throughput on NVIDIA |
| **K8s orchestration** | **NVIDIA Dynamo** or **KServe + vLLM Production Stack** | Ray Serve, BentoML, Seldon | Dynamo handles PD-disagg + KV-aware routing natively; KServe is more vendor-neutral |
| **Multi-node serving primitive** | **LeaderWorkerSet (LWS)** | StatefulSet + custom | LWS (sig-apps) is the K8s-native primitive for gang-scheduled multi-GPU TP groups |
| **GPU runtime** | **NVIDIA GPU Operator** | Manual nvidia-container-toolkit | Manages drivers, CDI, MIG, DCGM exporter, fabric manager |
| **Network** | **NVIDIA Network Operator** + **InfiniBand** (or RoCE v2) + **GPUDirect RDMA** | Plain Ethernet (kills perf) | Required for cross-node TP and PD KV transfer |
| **Topology-aware scheduling** | **Topology Manager (single-numa-node policy)** + **NRI plugins** | Manual node labels | Aligns GPU/NIC/CPU on same NUMA domain |
| **Service mesh / gateway** | **Envoy** (vLLM router) or **Dynamo Router** | Istio, NGINX | Need token-budget-aware + prefix-aware routing, not naive round-robin |
| **Model loading** | **Run:ai Model Streamer**, **vLLM model loader with safetensors on local NVMe**, or **CSI driver to S3/Blob** | HF download at boot (terrible) | 405 GB model at 1 GB/s = 7 min cold start; streamer brings it under 2 min |
| **Autoscaling** | **KEDA** + Prometheus adapter on **queue depth, TBT p95, KV-cache utilization** | HPA on GPU util alone (insufficient) | GPU util saturates at small batch; queue depth is the right signal |
| **KV cache tier** | **LMCache** (DRAM/SSD tier) | Mooncake Store (if you can self-host) | Cross-instance prefix reuse for RAG + system prompts |
| **Observability** | **Prometheus + Grafana + DCGM exporter** + **OpenTelemetry traces** | Datadog, New Relic | Need per-request prefill/decode breakdown, KV-cache hit rate, batch size histogram |
| **Image registry** | **Harbor** or **ACR** with **Dragonfly** P2P distribution | Plain registry | 30 GB CUDA images × 80 nodes = registry meltdown without P2P |
| **Storage for weights** | **Local NVMe per node** + **shared object store** (Azure Blob / S3) for cold copy | Network FS (NFS/Lustre) for the hot path | NVMe local cache makes pod restarts <2 min after first warm |
| **Secrets / auth** | Azure Workload Identity + Key Vault CSI driver | Vault Agent | Standard pattern; required if model is gated |
| **CI/CD** | **Argo CD** + Helm charts (NVIDIA NIM, vLLM Production Stack) | Flux, Spinnaker | GitOps for cluster + model rollouts |

### 4.7 Kubernetes Architecture

```
                    ┌────────────────────────────────────┐
                    │  Ingress (Application Gateway /    │
                    │  Azure Front Door, multi-region)   │
                    └─────────────────┬──────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │  Dynamo / vLLM Router              │
                    │  - Prefix-aware (radix tree)       │
                    │  - Token-budget-aware              │
                    │  - KV-cache affinity               │
                    └────────┬──────────────┬────────────┘
                             │              │
              ┌──────────────▼──┐      ┌────▼─────────────┐
              │ Prefill Pool    │      │ Decode Pool      │
              │ Replica×2       │      │ Replica×4        │
              │ LWS (Leader+7)  │      │ LWS (Leader+7)   │
              │ 8×B200 / node   │      │ 8×H200 / node    │
              │ TP=8, FP8       │      │ TP=8, FP8 + KV8  │
              └────────┬────────┘      └────┬──────────────┘
                       │                    │
                       │ KV via             │
                       │ GPUDirect RDMA     │
                       │ over IB 8×400 Gbps │
                       └────────┬───────────┘
                                │
                    ┌───────────▼────────────┐
                    │  LMCache / Mooncake    │
                    │  KV store (DRAM+NVMe)  │
                    └────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │ Cluster services:              │
                │ - GPU Operator (drivers, DCGM) │
                │ - Network Operator (RDMA CNI)  │
                │ - KEDA (autoscaling)           │
                │ - Prometheus + Grafana         │
                │ - Argo CD (GitOps)             │
                └────────────────────────────────┘
```

**Key K8s objects per replica (decode pool example):**

```yaml
# LeaderWorkerSet: gang-schedules a TP=8 group as one logical replica
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata: { name: llama-405b-decode }
spec:
  replicas: 4                  # 4 decode replicas
  leaderWorkerTemplate:
    size: 1                    # 1 leader (single-node TP=8 on 8xH200)
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      spec:
        containers:
        - name: vllm
          image: vllm/vllm-openai:v0.x.x
          args:
          - --model=meta-llama/Llama-3.1-405B-Instruct
          - --tensor-parallel-size=8
          - --quantization=fp8
          - --kv-cache-dtype=fp8
          - --enable-chunked-prefill
          - --max-num-batched-tokens=8192
          - --enable-prefix-caching
          resources:
            limits:
              nvidia.com/gpu: 8
              rdma/ib: 1       # InfiniBand HCA
          volumeMounts:
          - { name: model-cache, mountPath: /models }   # local NVMe
          - { name: shm, mountPath: /dev/shm }          # NCCL needs large shm
```

Pair with a **PodDisruptionBudget** (`minAvailable: 3` of 4), **PriorityClass** (system-cluster-critical-ish), and **NodeAffinity** pinning to a node pool labelled `accelerator=h200, network=ib-400`.

### 4.8 Autoscaling Signals

Naive HPA on `nvidia_gpu_utilization` fails: GPU util saturates at modest batch but throughput keeps climbing. Use composite signals via KEDA + Prometheus adapter:

| Signal | Source | Action |
|--------|--------|--------|
| Pending queue depth > 50 for 30s | vLLM /metrics | Scale out by 1 replica |
| Decode TPOT p95 > 60 ms for 60s | vLLM /metrics | Scale out (SLO violation) |
| KV cache utilization > 85% for 60s | vLLM /metrics | Scale out (next batch will be admission-blocked) |
| All three < 40% for 10 min | vLLM /metrics | Scale in by 1 replica |
| Time-of-day predictive (Prophet) | Forecasting service | Pre-warm before peak |

**Cold start mitigation:** 405 GB weights from Blob at 2 GB/s = ~3 min download; vLLM model loader from local NVMe ≈ 90 s; CUDA graph capture + warmup ≈ 60 s. **Total ~5 min cold start** — keep at least N+1 warm at all times; never scale-to-zero on customer-facing tiers.

### 4.9 Networking — Non-Negotiable Specifics

- **Intra-node**: NVLink + NVSwitch (H200: 900 GB/s GPU-to-GPU). Tensor Parallelism over plain PCIe is a non-starter — TP=8 all-reduces happen per-layer × 126 layers per token.
- **Inter-node** (for PD disagg KV transfer): **InfiniBand NDR 8×400 Gbps** or RoCE v2 with PFC/ECN tuned. Plain TCP loses ~70% of throughput vs RDMA on cross-node transfers.
- **GPUDirect RDMA**: HCA must be on the same PCIe root complex / NUMA node as the GPU it serves; the Network Operator + Topology Manager enforce this.
- **MTU 9000** (jumbo frames) end-to-end on the storage and KV-transfer fabrics.

### 4.10 Cost Estimate (Indicative)

| Item | Quantity | Unit price (Azure ND H200 v5 on-demand, 2026 est.) | Monthly |
|------|----------|----------------------------------------------------|---------|
| 8×H200 nodes (region A) | 5 | ~$40/hr × 730 hr | $146,000 |
| 8×H200 nodes (region B) | 5 | ~$40/hr × 730 hr | $146,000 |
| IB switches, storage, networking | — | ~10% overhead | $30,000 |
| **Total (on-demand)** | | | **~$320,000/mo** |
| With 1-year reserved (~40% discount) | | | **~$190,000/mo** |
| Add disagg-PD savings (~20%) | | | **~$150,000/mo** |

For frame of reference: a 405B model serving 1,000 active users at ~$150–200K/mo (~$150–200/user/mo) is roughly in line with what the public-cloud AI providers price their frontier-model tier at — confirming the sizing is realistic.

### 4.11 What to Build First

A defensible rollout order, minimizing risk:

1. **Single-replica 8×H200 vLLM, FP8, TP=8** behind a simple K8s Service. Hit it with synthetic load (k6 / locust). Measure prefill tok/s, decode tok/s, TTFT, TPOT.
2. **Scale to 2 replicas** behind the vLLM router / Dynamo. Validate prefix caching across replicas reduces TTFT on repeated system prompts.
3. **Add chunked prefill + continuous batching tuning** (`max-num-batched-tokens`, `max-num-seqs`). Re-measure.
4. **Add KEDA autoscaling** on queue depth + TPOT. Soak-test for 24 hours.
5. **Add second region** active-active. Validate failover.
6. **Only then**: consider PD disaggregation. The complexity tax is real; only worth it once colocated throughput is provably the bottleneck.
7. **Add LMCache / Mooncake-style KV tier** when prefix-cache hit rate on shared system prompts exceeds 40% — below that, the engineering cost outweighs the savings.

---

## Conclusion: Why the Stack Looks the Way It Does

The systems literature of the last three years has converged on a specific stack not by fashion but because the *physics* forces the answer. Three facts about transformer decoding drive everything:

1. **Decode is bandwidth-bound, prefill is compute-bound.** No batching strategy can change this — it is a property of the GEMV-vs-GEMM shape. The optimal response is to stop running them on the same hardware [1][7][9].
2. **The KV cache is the single most expensive resource.** It dominates HBM in decode and grows linearly with both batch size and context length. Every architectural decision in the modern stack — PagedAttention, RadixAttention, MLA, multi-tier KV stores, FP8/INT4 KV quant, H2O eviction, attention sinks — is some form of *making the cache smaller, more shared, or more cheaply replicated* [3][10][11][17][22].
3. **The arithmetic intensity is wildly different across a request's lifetime.** From `O(N²) compute` at prefill to `O(N) bandwidth` per decode step, no single hardware configuration is optimal for both. Hence disaggregation, hence heterogeneous fleets (B200 prefill + L40S decode), hence kernel-level co-design with the hardware's asymmetries (FlashAttention-3's producer/consumer warps, FP8 outlier protection) [16].

For a Microsoft architect designing a public-cloud-scale LLM service, the *minimum viable* stack in 2026 is:
- vLLM v1 or TensorRT-LLM with **chunked prefill + continuous batching + PagedAttention + FP8 weights and KV**;
- **RadixAttention / prefix-aware routing** at the gateway (Dynamo or SGLang router);
- **Disaggregated prefill/decode pools** at >100 QPS — Splitwise/Mooncake patterns recoup 30–50% cost vs colocated, validated on Azure's own GPT-4 traces [7][9];
- **Multi-tier KV store** (HBM/DRAM/SSD) over RDMA back-plane once cross-instance prefix reuse becomes meaningful;
- **Speculative decoding** (EAGLE / Medusa / MTP) for any latency-sensitive non-batch workload.

### Pre-Mortem and Residual Risks

If a deployment built on this blueprint were to fail in 6 months, the most likely causes:

1. **Underestimating cross-region KV non-shareability**: prefix caches are per-region, so a poorly chosen routing policy can collapse hit rates and double cost. Mitigate by sticky-routing system-prompt traffic.
2. **Hot-shard collapse on prefix-aware routing**: if one system prompt dominates traffic, all replicas serving it get crushed while others sit idle. Mitigate with hybrid policy: prefix-aware *within* a constraint of max load, falling back to least-loaded above threshold.
3. **Quantization quality drift at the long tail**: FP8 / INT4 quants pass standard benchmarks but fail on out-of-distribution domain-specific inputs. Mitigate with continuous shadow-evaluation in production.
4. **Speculative decoding inverting at high batch**: as load grows, average batch grows, and speculative decoding shifts from helpful to harmful. Mitigate by dynamic on/off based on current batch size.
5. **MoE all-to-all becoming the wall**: at the EP scales Kimi K2 / DeepSeek-V3 run, all-to-all is the bottleneck. If RoCE fabric isn't sized correctly (low bisection, asymmetric topology), throughput collapses non-linearly.

The good news: all five failure modes are *observable* at runtime with prefix-cache hit-rate, batch-size, and per-instance utilization metrics. The serving stack remains an engineering problem, not a research problem — the playbook now exists.

---

## Sources

1. Zhong, Y. *et al.* **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.** OSDI 2024. https://arxiv.org/abs/2401.09670 / USENIX PDF: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
2. *(DistServe USENIX paper.)* https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
3. Qin, R. *et al.* **Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot.** FAST 2025 Best Paper. https://www.usenix.org/system/files/fast25-qin.pdf
4. Qin, R. *et al.* **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.** arXiv:2407.00079 (2024–2025). https://arxiv.org/abs/2407.00079
5. Mooncake project page (Moonshot AI / Tsinghua / community). https://kvcache-ai.github.io/Mooncake/ — Kimi K2 on 128 H200 deployment numbers.
6. Mooncake Architecture documentation. https://kvcache-ai.github.io/Mooncake/design/architecture.html
7. Patel, P. *et al.* **Splitwise: Efficient Generative LLM Inference Using Phase Splitting.** ISCA 2024. https://arxiv.org/abs/2311.18677
8. Splitwise — Microsoft Research publication page. https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/
9. Choukse, E. **Splitwise presentation** (OCP). https://ocp-all.groups.io/g/OCP-CMS/attachment/241/0/Splitwise%20external%20longer%20%20presentation.pdf — A100/H100 ratios, MSCCL++ KV transfer.
10. Kwon, W. *et al.* **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention.** SOSP 2023 / vLLM blog. https://vllm.ai/blog/2023-06-20-vllm
11. Zheng, L. *et al.* **SGLang: Efficient Execution of Structured Language Model Programs.** NeurIPS 2024. https://arxiv.org/abs/2312.07104
12. Zheng, L. *et al.* **Fast and Expressive LLM Inference with RadixAttention and SGLang.** LMSYS blog, Jan 2024. https://www.lmsys.org/blog/2024-01-17-sglang/
13. Agrawal, A. *et al.* **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.** OSDI 2024. https://www.usenix.org/system/files/osdi24-agrawal.pdf
14. Yu, G.-I. *et al.* **Orca: A Distributed Serving System for Transformer-Based Generative Models.** OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu
15. Dao, T. *et al.* **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** NeurIPS 2022. https://arxiv.org/abs/2205.14135
16. Shah, J. *et al.* **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.** NeurIPS 2024. https://arxiv.org/abs/2407.08608
17. DeepSeek-AI. **DeepSeek-V3 Technical Report.** arXiv:2412.19437. https://arxiv.org/pdf/2412.19437
18. **Multi-head Latent Attention (MLA) in DeepSeek-V3** — DeepWiki. https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.2-multi-head-latent-attention-%28mla%29
19. DeepSeek-AI. **Day 6: DeepSeek-V3/R1 Inference System Overview.** Open Infra Index. https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
20. Shoeybi, M. *et al.* **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.** arXiv:1909.08053. https://arxiv.org/abs/1909.08053
21. Narayanan, D. *et al.* **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.** SC '21. https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf
22. Zhang, Z. *et al.* **H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.** NeurIPS 2023. https://arxiv.org/abs/2306.14048
23. **StreamingLLM (attention sinks) vs H2O analysis.** https://cyberinsist.com/blog/streamingllm-vs-h2o-kv-cache-management
24. NVIDIA. **TensorRT-LLM documentation** (in-flight batching, MLA, MTP, EAGLE3, Helix Parallelism, sparse attention). https://nvidia.github.io/TensorRT-LLM/
25. **TensorRT-LLM Inflight Batching internals.** DeepWiki. https://deepwiki.com/NVIDIA/TensorRT-LLM/20.1-inflight-batching-and-decoder
26. Liu, H., Zaharia, M., Abbeel, P. **Ring Attention with Blockwise Transformers for Near-Infinite Context.** arXiv:2310.01889. https://arxiv.org/abs/2310.01889
27. NVIDIA. **Dynamo KV-Cache-Aware Routing.** https://docs.dynamo.nvidia.com/dynamo/user-guides/kv-cache-aware-routing
28. Microsoft Azure Blog. **Microsoft and NVIDIA accelerate AI development and performance** (NIM on Azure AI Foundry; ND GB200 v6). https://azure.microsoft.com/en-us/blog/microsoft-and-nvidia-accelerate-ai-development-and-performance/
29. NVIDIA Developer Blog. **Accelerated AI Inference with NVIDIA NIM on Azure AI Foundry.** https://developer.nvidia.com/blog/accelerated-ai-inference-with-nvidia-nim-on-azure-ai-foundry/
30. Lin, J. *et al.* **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** MLSys 2024 Best Paper. https://arxiv.org/abs/2306.00978
31. Leviathan, Y., Kalman, M., Matias, Y. **Fast Inference from Transformers via Speculative Decoding.** ICML 2023. https://arxiv.org/abs/2211.17192
32. Li, Y. *et al.* **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.** arXiv:2401.15077. https://arxiv.org/html/2401.15077v3
33. **Disaggregated Prefill and Decode (Splitwise / DistServe)** — General Compute blog, Apr 2026. Useful for H100 prefill/decode throughput numbers. https://www.generalcompute.com/blog/disaggregated-prefill-and-decode
