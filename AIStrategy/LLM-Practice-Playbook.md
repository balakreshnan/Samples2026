# LLM Practice Playbook: Fine-Tuning, Pretraining & Training at Scale

**Prepared for:** Balamurugan Balakreshnan, Principal Cloud Solution Architect  
**Date:** June 2026  
**Version:** 1.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Training Phases & Unique Techniques](#training-phases--unique-techniques)
   - 2.1 Pretraining
   - 2.2 Fine-Tuning
   - 2.3 Post-Training Alignment
   - 2.4 Inference Optimization
3. [GPU & Compute Providers](#gpu--compute-providers)
   - 3.1 Hyperscalers
   - 3.2 AI Neo-Clouds
   - 3.3 On-Premises Options
   - 3.4 Provider Comparison Matrix
4. [Organizational Structure](#organizational-structure)
   - 4.1 Org Chart
   - 4.2 RACI Matrix
   - 4.3 Team Sizing Guide
5. [Skills & Competency Framework](#skills--competency-framework)
6. [Cost Modeling & Financial Planning](#cost-modeling--financial-planning)
7. [Training & Upskilling Curriculum](#training--upskilling-curriculum)
8. [Change Management](#change-management)
9. [Reinforcement Learning for LLMs](#reinforcement-learning-for-llms)
10. [Evaluation Frameworks](#evaluation-frameworks)
11. [Responsible AI](#responsible-ai)
12. [MLOps Maturity Model](#mlops-maturity-model)
13. [Technology Stack Reference](#technology-stack-reference)
14. [Roadmap & Quick Wins](#roadmap--quick-wins)

---

## 1. Executive Summary

Building a world-class LLM practice requires mastery across three interlocking domains: **compute infrastructure**, **model science**, and **organizational capability**. This playbook provides a comprehensive blueprint for standing up a center of excellence (CoE) capable of pretraining, fine-tuning, and deploying both small language models (SLMs, 1B–13B parameters) and large language models (LLMs, 30B–405B+ parameters).

**Key Investment Thesis:**
- Fine-tuning open-source frontier models (Llama 3.x, Mistral, Phi-4, Falcon) costs 5–50× less than training from scratch and often achieves 90%+ of task-specific performance
- Reinforcement learning alignment (DPO, GRPO) is now accessible without massive compute overhead
- A lean, specialized team of 8–15 people can run production-grade LLM pipelines with the right toolchain

**Strategic Imperatives:**
1. Build compute-provider agnosticism (multi-cloud GPU strategy)
2. Invest in data curation and evaluation infrastructure before model training
3. Embed Responsible AI practices at every stage of the training lifecycle
4. Establish MLOps Level 2 capabilities for continuous model improvement

---

## 2. Training Phases & Unique Techniques

### 2.1 Pretraining

Pretraining is the most compute-intensive phase — teaching the model language understanding from raw text at scale.

#### Data Curation (The Most Underrated Phase)

| Technique | Description | Tools |
|---|---|---|
| **Deduplication** | MinHash LSH or exact-match dedup to remove near-duplicates | DataTrove, Dolma, RedPajama pipeline |
| **Quality filtering** | Perplexity scoring, heuristic filters (line length, punctuation ratios) | CCNet, Gopher rules, FineWeb filters |
| **Domain weighting** | Upsampling high-quality domains (books, academic papers, code) | Cerebras' Slimpajama methodology |
| **Curriculum learning** | Present data in increasing difficulty order — improves convergence | Custom data schedulers |
| **Multilingual balancing** | ISO 639 language detection + weighted sampling for multilingual pretraining | FastText lid.176, cld3 |
| **Synthetic data injection** | Add instruction-following, reasoning, and code synthetic data during pretraining | Phi-4 "textbook quality" approach |
| **Data flywheel** | Use model generations + human feedback to continuously improve the training corpus | Production inference logs → filtered dataset |

#### Tokenization Strategies

| Strategy | Use Case | Example |
|---|---|---|
| **BPE (Byte-Pair Encoding)** | General purpose — GPT family | tiktoken (GPT-4), LLaMA tokenizer |
| **SentencePiece (Unigram)** | Multilingual, handles morphologically rich languages | T5, Gemma |
| **BBPE (Byte-level BPE)** | No unknown tokens, handles all Unicode | GPT-2, RoBERTa |
| **Custom domain vocab** | Add domain-specific tokens (medical, legal, code) pre-training | BioMedLM, StarCoder |

#### Architecture Choices

| Component | Options | Current Best Practice (2026) |
|---|---|---|
| **Attention** | MHA, GQA, MQA, MLA | GQA (Grouped Query Attention) — LLaMA 3.x standard |
| **Normalization** | LayerNorm, RMSNorm, Pre-Norm | Pre-Norm RMSNorm (more stable) |
| **Position Encoding** | Sinusoidal, RoPE, ALiBi, NoPE | RoPE with YaRN extension for long context |
| **Activation** | GeLU, SwiGLU, GeGLU | SwiGLU (used in LLaMA, PaLM) |
| **MoE (Mixture of Experts)** | Dense vs sparse routing | Mixtral-style top-2 routing for efficiency |

#### Pretraining Optimization Techniques

| Technique | What It Does | When to Use |
|---|---|---|
| **Flash Attention 2/3** | Recomputes attention in tiles — 2–4× memory reduction, 2× speed | Always — near-zero cost to enable |
| **ZeRO Stage 1/2/3** | Shards optimizer state, gradients, parameters across GPUs | ZeRO-2 for most cases; ZeRO-3 for 70B+ |
| **Tensor Parallelism** | Splits individual layers across GPUs | Required for 70B+ on a single node |
| **Pipeline Parallelism** | Splits model layers across nodes | Required for 100B+ across multiple nodes |
| **Gradient Checkpointing** | Recompute activations on backward pass — 60% memory savings | Always for long context windows |
| **Mixed Precision (BF16)** | 16-bit training with float32 master weights | Standard for all H100/A100 training |
| **Gradient Accumulation** | Simulate large batch with small GPU memory | When batch size is limited |
| **Activation Offloading** | Move activations to CPU RAM during forward pass | For extreme memory constraints |

#### Continued Pretraining vs Domain-Adaptive Pretraining (DAPT)

```
Scenario A: Start from scratch
  → Use case: Novel architecture, proprietary dataset, unique domain (medical imaging + text)
  → Cost: $500K–$50M+ depending on model size
  → Risk: High — requires deep expertise in distributed training

Scenario B: Continued Pretraining (CPT)
  → Use case: Extend a public base model with domain-specific corpus
  → Example: Take Llama 3.1 8B, continue training on 50B tokens of legal documents
  → Cost: $10K–$500K depending on tokens and model size
  → Risk: Medium — catastrophic forgetting must be managed via data mixing

Scenario C: Domain-Adaptive Pretraining (DAPT)
  → Use case: Inject specialized knowledge before fine-tuning
  → Example: BioMedLM — pretrain on PubMed before medical Q&A fine-tuning
  → Cost: $5K–$100K
  → Best practice: Mix 10–20% general text with domain text to prevent forgetting
```

---

### 2.2 Fine-Tuning

Fine-tuning adapts a pretrained model to specific tasks, formats, or domains.

#### Parameter-Efficient Fine-Tuning (PEFT) Techniques

| Technique | Description | Memory Savings | Performance |
|---|---|---|---|
| **LoRA** (Low-Rank Adaptation) | Injects trainable rank-decomposition matrices into attention layers | 60–80% vs full FT | 95–98% of full FT |
| **QLoRA** | LoRA + 4-bit NF4 quantization of base model | 90%+ vs full FT | 90–95% of full FT |
| **DoRA** (Weight-Decomposed LoRA) | Decomposes weights into magnitude and direction | Similar to LoRA | Often outperforms LoRA |
| **LoRA+** | Asymmetric learning rates for A and B matrices | Same as LoRA | ~2% better than LoRA |
| **Prefix Tuning** | Prepend learnable tokens to input | Minimal | Task-specific only |
| **Prompt Tuning** | Learn soft prompts in embedding space | Minimal | Limited generalization |
| **Adapter Layers** | Insert small trainable bottleneck layers | 40–60% | Good for multi-task |
| **Full Fine-Tuning** | Update all parameters | None | Best performance |

#### LoRA Hyperparameter Guide

```yaml
# Recommended LoRA config for instruction fine-tuning
lora_r: 16-64           # Rank: higher = more capacity, more memory
lora_alpha: 32-128      # Scale: typically 2x rank
lora_dropout: 0.05-0.1  # Regularization
target_modules:          # Which layers to adapt
  - q_proj
  - v_proj
  - k_proj               # Add for better performance
  - o_proj               # Add for best results
  - gate_proj
  - up_proj
  - down_proj            # MLP layers for domain adaptation
bias: "none"             # Don't train bias terms
task_type: "CAUSAL_LM"
```

#### Fine-Tuning Data Formats

| Format | Best For | Example |
|---|---|---|
| **Alpaca** | Instruction following | `{instruction, input, output}` |
| **ShareGPT** | Multi-turn conversation | `{conversations: [{from, value}]}` |
| **OpenAI Chat** | Chat models | `{messages: [{role, content}]}` |
| **Custom** | Domain-specific tasks | Depends on task structure |

**Data Quality > Data Quantity:**
- 1,000 high-quality, diverse examples often outperform 100,000 noisy ones
- Use LLM-as-judge to filter synthetic data before training
- LIMA paper showed 1,000 carefully curated samples match RLHF-aligned models

#### Supervised Fine-Tuning (SFT) Best Practices

1. **Data mix**: 70–80% task-specific, 20–30% general instruction following (to preserve capabilities)
2. **Learning rate**: Start at 1e-4 for LoRA, 1e-5 to 2e-5 for full fine-tuning
3. **Epochs**: 1–3 epochs (overfitting is common with small datasets)
4. **Packing**: Pack multiple short examples into one sequence for efficiency
5. **Evaluation**: Hold out 5–10% for validation; monitor training loss AND eval metrics

---

### 2.3 Post-Training Alignment

Alignment techniques teach the model to be helpful, harmless, and honest.

#### Alignment Technique Decision Tree

```
Do you have human preference data?
├── YES → Use DPO or IPO (most efficient)
│         Do you have process-level feedback?
│         ├── YES → Process Reward Model (PRM) + GRPO
│         └── NO  → Outcome-level DPO is sufficient
└── NO  → Options:
          ├── Can you use an AI judge? → RLAIF / Self-Reward
          ├── Have strong few-shot examples? → Constitutional AI
          └── Starting fresh? → PPO with reward model (expensive but gold standard)
```

#### Technique Comparison

| Technique | Compute Cost | Human Labels Needed | Stability | When to Use |
|---|---|---|---|---|
| **PPO (RLHF)** | Very High | 50K–500K | Unstable | Gold standard, big teams |
| **DPO** | Low | 5K–50K | Stable | Best default choice |
| **IPO** | Low | 5K–50K | More stable than DPO | When DPO over-optimizes |
| **KTO** | Low | Unpaired feedback OK | Stable | When you lack paired data |
| **SimPO** | Low | 5K–50K | Very stable | Reference-free alternative |
| **GRPO** | Medium | 5K–50K | Stable | Reasoning models |
| **Constitutional AI** | Medium | Minimal | Stable | Principled safety alignment |
| **RLAIF** | Medium | Minimal | Moderate | Scaling labeling |

---

### 2.4 Inference Optimization

| Technique | Speedup | Memory Savings | Quality Loss |
|---|---|---|---|
| **INT8 Quantization** | 1.5–2× | 50% | Minimal |
| **INT4/NF4 Quantization** | 2–4× | 75% | Small |
| **GPTQ** | 3–4× | 75% | Minimal |
| **AWQ** | 3–4× | 75% | Minimal (better than GPTQ) |
| **Speculative Decoding** | 2–3× | None | Zero |
| **vLLM (PagedAttention)** | 3–24× throughput | Efficient KV cache | Zero |
| **Continuous Batching** | 10–50× throughput | N/A | Zero |
| **Flash Decoding** | 2–5× for long seq | N/A | Zero |

---

## 3. GPU & Compute Providers

### 3.1 Hyperscalers

#### Microsoft Azure

| Instance | GPU | GPUs | vCPUs | RAM | Price/hr (On-Demand) | Price/hr (Spot) |
|---|---|---|---|---|---|---|
| **NC40ads A100v4** | A100 80GB | 1 | 40 | 320GB | ~$3.67 | ~$1.10 |
| **NC80adis A100v4** | A100 80GB | 2 | 80 | 640GB | ~$7.35 | ~$2.20 |
| **ND96asr v4** | A100 80GB | 8 | 96 | 900GB | ~$27.20 | ~$8.16 |
| **ND96amsr A100v4** | A100 80GB SXM | 8 | 96 | 1.9TB | ~$32.77 | ~$9.83 |
| **ND H100 v5** | H100 80GB | 8 | 96 | 1.9TB | ~$98.32 | ~$29.50 |
| **NDm H100 v5** | H100 80GB SXM | 8 | 96 | 2TB | ~$108.00 | ~$32.40 |

**Azure-Specific Advantages:**
- Azure OpenAI Service integration for model deployment
- Azure ML for MLOps pipelines
- Reserved Capacity Blocks for predictable GPU access (3-month, 1-year)
- CycleCloud for HPC cluster management
- InfiniBand networking for multi-node training (3.2 Tbps)

#### AWS

| Instance | GPU | GPUs | Price/hr | Notes |
|---|---|---|---|---|
| **p4d.24xlarge** | A100 40GB | 8 | ~$32.77 | 400 Gbps EFA networking |
| **p4de.24xlarge** | A100 80GB | 8 | ~$40.96 | Higher memory |
| **p5.48xlarge** | H100 80GB SXM | 8 | ~$98.32 | EFA v2, 3.2 Tbps |
| **p5e.48xlarge** | H100 80GB SXM | 8 | ~$113.44 | Latest generation |
| **Trn1.32xlarge** | Trainium | 16 | ~$21.50 | Purpose-built for training |
| **Trn2.48xlarge** | Trainium2 | 16 | ~$28.00 | Next-gen, cheaper than H100 |

**AWS Capacity Blocks:** Reserve H100 clusters (1–64 instances) for 1–56 days, 1–60 days ahead. More reliable than spot for training jobs.

#### Google Cloud Platform

| Instance | Accelerator | Count | Price/hr | Notes |
|---|---|---|---|---|
| **a2-highgpu-8g** | A100 40GB | 8 | ~$29.39 | Standard A100 |
| **a2-ultragpu-8g** | A100 80GB | 8 | ~$40.24 | 80GB variant |
| **a3-highgpu-8g** | H100 80GB | 8 | ~$98.32 | Latest H100 |
| **a3-megagpu-8g** | H100 Mega 80GB | 8 | ~$112.00 | Higher NVLink bandwidth |
| **TPU v5e-8** | TPUv5e | 8 | ~$12.88 | Best for Google models |
| **TPU v5p-8** | TPUv5p | 8 | ~$42.24 | Highest-end TPU |

**GCP Advantages:**
- TPUs are unmatched for JAX/PyTorch XLA training at scale
- Vertex AI for MLOps
- 3600 Tbps bisectional bandwidth in TPU pods

---

### 3.2 AI Neo-Clouds & Specialized GPU Providers

These providers often offer significantly lower costs than hyperscalers, with trade-offs in support, compliance, and ecosystem.

#### Tier 1 — Enterprise-Grade Neo-Clouds

| Provider | Key GPUs | H100 Price/hr | Specialty | Compliance |
|---|---|---|---|---|
| **CoreWeave** | H100, A100, H200, GB200 | ~$2.49–$2.89 (H100 SXM) | Best networking, RDMA | SOC 2, HIPAA |
| **Lambda Labs** | H100, A100, A10 | ~$2.49 (H100 SXM) | Simple API, Jupyter | SOC 2 |
| **Together AI** | H100, A100 | ~$2.50 | Fine-tuning APIs | SOC 2 |
| **Voltage Park** | H100 | ~$2.40–$2.60 | Large H100 clusters | Emerging |
| **Nebius** | H100 | ~$2.80 | European data residency | GDPR |

#### Tier 2 — Cost-Optimized Providers

| Provider | Key GPUs | Price Range | Specialty |
|---|---|---|---|
| **Runpod** | H100, A100, A40 | $1.49–$2.89/hr | On-demand + spot, marketplace |
| **Vast.ai** | H100, A100, RTX | $0.75–$2.50/hr | Peer-to-peer marketplace |
| **Crusoe Energy** | H100, A100 | ~$2.39/hr | Stranded natural gas, low carbon |
| **Fluidstack** | H100, A100 | ~$2.29–$2.69/hr | European presence |
| **Massed Compute** | H100, A100 | ~$2.30/hr | US-based, reliable |
| **Lepton AI** | H100 | ~$2.45/hr | Developer-friendly APIs |

#### Tier 3 — Developer/Experimentation

| Provider | Best For | Price Range |
|---|---|---|
| **Modal** | Serverless GPU functions, CI/CD | Per-second billing |
| **Replicate** | Model hosting, inference APIs | Per-second inference |
| **Fireworks AI** | Fast inference APIs, fine-tuning | Per-token pricing |
| **Anyscale** | Ray-based distributed training | On-demand + reserved |

---

### 3.3 On-Premises Options

| System | GPUs | Memory | Network | Starting Price |
|---|---|---|---|---|
| **NVIDIA DGX H100** | 8× H100 80GB | 640GB HBM3 | 400 Gbps InfiniBand | ~$300K–$400K |
| **NVIDIA DGX SuperPOD** | 32+ DGX units | 20TB+ HBM3 | 3.2 Tbps | Custom ($10M+) |
| **NVIDIA GB200 NVL72** | 72 Blackwell GPUs | 13.5TB HBM3e | NVLink 5.0 | ~$3M+ per rack |
| **SuperMicro H100 Server** | 8× H100 | 640GB | Up to InfiniBand | ~$200K–$300K |
| **AMD Instinct MI300X** | 8× MI300X | 1.5TB HBM3 | 800 Gbps | ~$200K–$280K |

**On-Prem Considerations:**
- Power: ~30–40 kW per DGX system — datacenter requirements significant
- TCO: 3-year TCO often favorable for consistent, high-utilization workloads (>60%)
- Lead times: 6–18 months for large NVIDIA orders (2025–2026)

---

### 3.4 Provider Comparison Matrix

| Criteria | Azure | AWS | GCP | CoreWeave | Lambda | Runpod |
|---|---|---|---|---|---|---|
| **H100 On-Demand $/hr** | $98+ | $98+ | $98+ | $2.89 | $2.49 | $1.89 |
| **Cluster Scale** | 10,000+ GPUs | 10,000+ GPUs | 10,000+ GPUs | 5,000+ GPUs | 1,000+ GPUs | 1,000+ GPUs |
| **MLOps Integration** | Azure ML ⭐⭐⭐⭐⭐ | SageMaker ⭐⭐⭐⭐ | Vertex AI ⭐⭐⭐⭐ | Basic ⭐⭐ | Basic ⭐⭐ | Basic ⭐ |
| **Compliance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Networking** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Cost** | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Best For** | Enterprise, compliance | Enterprise, scale | TPU/JAX training | Production LLM training | Mid-scale fine-tuning | Experimentation |

**Recommended Strategy:**
- **Development & experimentation**: Runpod, Vast.ai, Lambda
- **Production fine-tuning** (10–100 GPU runs): CoreWeave, Lambda Labs
- **Large-scale pretraining** (500+ GPUs): Azure, AWS, CoreWeave
- **Enterprise production inference**: Azure, AWS, GCP
- **Compliance-sensitive workloads**: Azure (HIPAA, FedRAMP, ISO 27001)

---

## 4. Organizational Structure

### 4.1 LLM Practice Org Chart

```
Director / VP of AI Research & Engineering
│
├── AI Research Lead
│   ├── Senior Research Scientist (Pretraining) ×2
│   ├── Research Scientist (Alignment) ×2
│   └── Research Scientist (Evaluation) ×1
│
├── ML Engineering Lead
│   ├── Senior ML Engineer (Distributed Training) ×2
│   ├── ML Engineer (Fine-Tuning & PEFT) ×3
│   └── ML Engineer (Inference Optimization) ×2
│
├── Data Engineering Lead
│   ├── Senior Data Engineer (Data Curation) ×2
│   └── Data Engineer (Synthetic Data & Pipelines) ×2
│
├── MLOps & Platform Lead
│   ├── MLOps Engineer (Training Infrastructure) ×2
│   ├── MLOps Engineer (Model Registry & CI/CD) ×1
│   └── Cloud Infrastructure Engineer ×2
│
├── Responsible AI Lead
│   ├── AI Safety Engineer ×1
│   ├── Fairness & Ethics Specialist ×1
│   └── Red Team / Evaluation Specialist ×1
│
└── Product & Program Management
    ├── Technical Program Manager (LLM Projects) ×1
    └── Product Manager (AI Products) ×1
```

**Minimum Viable Team (Startup Mode):** 8 people
- 1 ML Research Lead, 2 ML Engineers (training + inference), 1 Data Engineer,
  1 MLOps Engineer, 1 AI Safety Specialist, 1 TPM, 1 Product Manager

**Full Practice Team:** 20–30 people (described above with growth)

---

### 4.2 RACI Matrix

#### Core Activities

| Activity | Director | AI Research Lead | ML Eng Lead | Data Eng Lead | MLOps Lead | RAI Lead | TPM |
|---|---|---|---|---|---|---|---|
| **Research direction & model selection** | A | R | C | I | I | C | I |
| **Pretraining data pipeline** | I | C | I | R | C | C | I |
| **Model architecture decisions** | A | R | C | I | I | I | I |
| **Distributed training runs** | I | C | R | I | C | I | I |
| **Fine-tuning experiments** | I | C | R | C | I | C | I |
| **Alignment / RLHF / DPO** | I | R | C | I | I | C | I |
| **Evaluation benchmarking** | I | C | C | I | I | R | I |
| **Red teaming / safety eval** | A | C | C | I | I | R | I |
| **Model registry & versioning** | I | I | C | I | R | I | C |
| **Production deployment** | A | I | C | I | R | C | R |
| **GPU procurement & cloud contracts** | A | C | C | I | C | I | R |
| **Cost tracking & optimization** | A | I | C | I | C | I | R |
| **Incident response** | A | C | C | I | R | C | R |
| **Compliance & regulatory** | A | C | I | I | C | R | C |
| **Team hiring & growth** | R | C | C | C | C | C | I |

**Legend:** R = Responsible, A = Accountable, C = Consulted, I = Informed

#### Data Governance RACI

| Activity | Data Eng Lead | ML Eng Lead | RAI Lead | Legal/Compliance | Security |
|---|---|---|---|---|---|
| **Data source identification** | R | C | C | C | I |
| **License & copyright review** | C | I | C | R | I |
| **PII scrubbing & anonymization** | R | C | C | C | C |
| **Data quality standards** | R | C | C | I | I |
| **Data retention policy** | C | I | C | R | C |
| **Access controls** | C | I | C | C | R |

---

### 4.3 Team Sizing Guide by Stage

| Stage | Description | Team Size | Annual Budget (People) |
|---|---|---|---|
| **Pilot** | 1–2 fine-tuning use cases, no pretraining | 3–5 | $600K–$1.2M |
| **Growth** | Multiple fine-tuning pipelines, some CPT | 8–12 | $2M–$4M |
| **Scale** | Full pretraining capability, model org | 15–25 | $5M–$12M |
| **Enterprise CoE** | Multiple model families, 24/7 production | 25–50 | $12M–$30M |

---

## 5. Skills & Competency Framework

### 5.1 Technical Skills by Role

#### ML Research Scientist

| Skill | Level Required |
|---|---|
| PyTorch / JAX | Expert |
| Transformer architecture (attention, MLP, norm layers) | Expert |
| Distributed training (DDP, FSDP, DeepSpeed, Megatron-LM) | Advanced |
| Fine-tuning methods (LoRA, QLoRA, DPO, PPO) | Expert |
| Evaluation methodology (benchmarks, statistical testing) | Advanced |
| Python (NumPy, SciPy, HuggingFace Transformers, Datasets) | Expert |
| Reading research papers and implementing | Expert |
| Mathematics (linear algebra, probability, optimization) | Advanced |

#### ML Engineer (Training)

| Skill | Level Required |
|---|---|
| PyTorch / CUDA profiling | Advanced |
| DeepSpeed / Accelerate / HuggingFace Trainer | Expert |
| Multi-GPU/multi-node training orchestration | Expert |
| Flash Attention, quantization libraries | Advanced |
| Kubernetes / Slurm for GPU job scheduling | Advanced |
| Git, CI/CD, Python packaging | Expert |
| Debugging training instabilities (loss spikes, NaN) | Advanced |
| NCCL/GLOO distributed communication | Intermediate |

#### MLOps Engineer

| Skill | Level Required |
|---|---|
| MLflow / Weights & Biases / Comet | Expert |
| Kubernetes (GPU scheduling, affinity rules) | Advanced |
| Docker, Helm, Terraform | Advanced |
| Azure ML / SageMaker / Vertex AI | Advanced |
| Model serving (vLLM, TGI, Triton Inference Server) | Advanced |
| Python, Bash scripting | Expert |
| Observability (Prometheus, Grafana for GPU metrics) | Intermediate |
| CI/CD for ML (GitHub Actions, Azure DevOps) | Advanced |

#### Data Engineer

| Skill | Level Required |
|---|---|
| Large-scale data processing (Spark, Ray Data, Dask) | Advanced |
| Data formats (Parquet, Arrow, MDS/MosaicML Streaming) | Advanced |
| Text processing (tokenization, dedup pipelines) | Advanced |
| SQL, NoSQL, vector databases | Advanced |
| Data quality & testing frameworks | Intermediate |
| Cloud storage (Azure Blob, S3, GCS) | Advanced |

#### AI Safety / Responsible AI

| Skill | Level Required |
|---|---|
| Bias detection and fairness metrics | Advanced |
| Red teaming methodologies | Advanced |
| RLHF / Constitutional AI alignment | Intermediate |
| EU AI Act, NIST AI RMF compliance | Advanced |
| Explainability (SHAP, probing classifiers) | Intermediate |
| Content moderation systems | Advanced |

### 5.2 Skill Gap Assessment Framework

```
For each team member, assess:

Level 1 — Awareness: Has heard of the technique
Level 2 — Conceptual: Can explain what it does
Level 3 — Applied: Has used it in a project
Level 4 — Proficient: Can tune and debug it
Level 5 — Expert: Can extend or innovate on it

Priority Skills for a New LLM Practice (Q1 Targets):
├── HuggingFace Transformers + Datasets: all engineers → Level 4+
├── LoRA/QLoRA fine-tuning: all ML engineers → Level 4+
├── vLLM inference serving: MLOps → Level 4+
├── Weights & Biases experiment tracking: all → Level 3+
├── DPO alignment: research scientists → Level 4+
└── Responsible AI red teaming: RAI lead → Level 5
```

---

## 6. Cost Modeling & Financial Planning

### 6.1 Pretraining Cost Estimates

| Model Size | Tokens | GPU Type | GPUs | Hours | Approx. Cost |
|---|---|---|---|---|---|
| **1B params** | 50B tokens | H100 | 8 | ~40 hrs | ~$1,500 |
| **3B params** | 100B tokens | H100 | 8 | ~200 hrs | ~$7,000 |
| **7B params** | 1T tokens | H100 | 64 | ~1,200 hrs | ~$180,000 |
| **13B params** | 1T tokens | H100 | 128 | ~1,200 hrs | ~$360,000 |
| **70B params** | 2T tokens | H100 | 512 | ~2,000 hrs | ~$6,000,000 |
| **405B params** | 15T tokens | H100 | 16,384 | ~3,000 hrs | ~$50,000,000+ |

*Using CoreWeave/Lambda pricing (~$2.50/hr per H100)*

### 6.2 Fine-Tuning Cost Estimates

| Task | Model | Method | Dataset Size | GPUs | Hours | Cost |
|---|---|---|---|---|---|---|
| **Instruction FT** | 7B | QLoRA | 10K examples | 1 A100 | 2–4 hrs | ~$15–$30 |
| **Instruction FT** | 7B | LoRA | 50K examples | 4 A100 | 4–8 hrs | ~$60–$120 |
| **Instruction FT** | 7B | Full FT | 100K examples | 8 H100 | 8–16 hrs | ~$300–$600 |
| **Instruction FT** | 13B | QLoRA | 50K examples | 2 A100 | 4–8 hrs | ~$60–$120 |
| **Instruction FT** | 70B | QLoRA | 50K examples | 4 H100 | 12–24 hrs | ~$300–$600 |
| **DPO Alignment** | 7B | DPO | 20K pairs | 4 A100 | 4–8 hrs | ~$60–$120 |
| **CPT (Domain)** | 7B | Full | 10B tokens | 32 H100 | 40–80 hrs | ~$8K–$16K |

### 6.3 Infrastructure & Operational Costs

| Cost Category | Monthly Range | Notes |
|---|---|---|
| **GPU Compute (Development)** | $5K–$30K | Experimentation, exploration |
| **GPU Compute (Training)** | $20K–$200K | Varies heavily by scale |
| **GPU Compute (Inference)** | $10K–$100K | Serving production traffic |
| **Storage (Training Data)** | $500–$5K | ~$0.02/GB/month (blob storage) |
| **Storage (Checkpoints)** | $1K–$10K | 70B model checkpoint ~100–300GB |
| **Networking (egress)** | $500–$5K | Data transfer costs |
| **Experiment Tracking (W&B)** | $0–$2K | Free tier generous |
| **MLOps Platform** | $2K–$10K | Azure ML / W&B Teams |

### 6.4 Total Annual Budget Models

| Practice Level | People Costs | Compute Costs | Tooling | **Total Annual** |
|---|---|---|---|---|
| **Pilot (5 people)** | $1.5M | $200K | $50K | **~$1.75M** |
| **Growth (12 people)** | $4M | $1M | $200K | **~$5.2M** |
| **Scale (20 people)** | $8M | $5M | $500K | **~$13.5M** |
| **Enterprise CoE (40 people)** | $16M | $20M | $1M | **~$37M** |

### 6.5 Cost Optimization Strategies

1. **Spot/Preemptible Instances**: 60–80% cheaper; use with checkpoint-every-N-steps strategy
2. **Mixed Provider Strategy**: Hyperscaler for compliance workloads + neo-cloud for raw training
3. **Quantization for Inference**: INT4 quantization reduces inference cost by 4× with minimal quality loss
4. **Model Distillation**: Train a 7B student from a 70B teacher — 10× cheaper inference
5. **LoRA by Default**: Default to QLoRA for all fine-tuning experiments before committing to full fine-tuning
6. **Automatic Checkpoint Pruning**: Keep only the top-N checkpoints to control storage costs
7. **Spot Instance Orchestration**: Tools like SkyPilot, Fluidstack API, or Ray for spot-aware scheduling

---

## 7. Training & Upskilling Curriculum

### 7.1 Learning Paths by Role

#### Path 1: ML Engineer (3–6 months)

**Month 1 — Foundations**
- [ ] Deep Learning Specialization (deeplearning.ai) — Courses 1–4
- [ ] Attention Is All You Need paper + The Illustrated Transformer (Jay Alammar)
- [ ] HuggingFace NLP Course (free, huggingface.co/learn)
- [ ] PyTorch fundamentals (official tutorials)

**Month 2 — LLM Fundamentals**
- [ ] LLM from Scratch (Sebastian Raschka) — book
- [ ] HuggingFace PEFT library tutorials
- [ ] LoRA: Low-Rank Adaptation paper (Hu et al., 2021)
- [ ] QLoRA paper (Dettmers et al., 2023)
- [ ] Practical: Fine-tune LLaMA 3.1 8B on a custom dataset with QLoRA

**Month 3 — Distributed Training**
- [ ] DeepSpeed documentation (ZeRO stages)
- [ ] Accelerate library (HuggingFace)
- [ ] FSDP (PyTorch Fully Sharded Data Parallel) tutorial
- [ ] Practical: Multi-GPU training job on 8×A100

**Month 4 — Alignment & Post-Training**
- [ ] InstructGPT paper (Ouyang et al., 2022)
- [ ] DPO paper (Rafailov et al., 2023)
- [ ] TRL library (HuggingFace) tutorials
- [ ] Practical: Run DPO on a small preference dataset

**Month 5 — Production & MLOps**
- [ ] vLLM documentation + practical serving
- [ ] Weights & Biases course (free)
- [ ] MLflow model registry
- [ ] Practical: Set up end-to-end fine-tune → evaluate → deploy pipeline

**Month 6 — Advanced Topics**
- [ ] Flash Attention paper + implementation
- [ ] Mixture of Experts (MoE) architecture review
- [ ] GRPO / reasoning models (DeepSeek-R1 paper)
- [ ] Capstone: End-to-end LLM project with evaluation + deployment

#### Path 2: ML Research Scientist (6–12 months)

- All of Path 1 content, plus:
- Stanford CS224N (NLP with Deep Learning) — full course
- Scaling Laws for Neural Language Models (Kaplan et al.)
- Chinchilla paper (Hoffmann et al., 2022) — optimal compute allocation
- RLHF survey papers + Constitutional AI (Anthropic, 2022)
- Process Reward Models (Lightman et al., 2023)
- Practical: Implement a training run on a new dataset from scratch
- Read and reproduce: 1 recent ICLR/NeurIPS/ICML paper per month

#### Path 3: MLOps Engineer (2–4 months)

- MLOps Zoomcamp (free, DataTalks.Club)
- Azure ML + Azure DevOps for ML pipelines
- Kubernetes for ML (GPU scheduling, affinity, tolerations)
- vLLM + TGI (Text Generation Inference) deployment
- Prometheus + Grafana for ML observability
- LLM Evaluation CI/CD with lm-evaluation-harness

#### Path 4: Responsible AI Specialist (3–6 months)

- Microsoft Responsible AI Standard (internal)
- NIST AI Risk Management Framework (free)
- EU AI Act overview + compliance checklist
- Fairness and Machine Learning (Barocas, Hardt, Narayanan) — free textbook
- Red Teaming Language Models with Language Models (Perez et al.)
- AI Safety Fundamentals course (BlueDot Impact)
- Constitutional AI paper (Anthropic, 2022)

### 7.2 Certifications & External Programs

| Certification / Program | Provider | Relevance | Duration |
|---|---|---|---|
| **Deep Learning Specialization** | deeplearning.ai | ML Engineers | 3–4 months |
| **MLOps Specialization** | deeplearning.ai | MLOps Engineers | 2–3 months |
| **LLM Bootcamp** | Full Stack Deep Learning | All LLM roles | 2 months |
| **Azure AI Engineer Associate (AI-102)** | Microsoft | All roles | 1–2 months |
| **Azure ML (DP-100)** | Microsoft | MLOps, ML Eng | 1–2 months |
| **Google Professional ML Engineer** | Google | ML Engineers | 2–3 months |
| **AWS ML Specialty** | AWS | MLOps, ML Eng | 2–3 months |
| **CIPP/CIPM** | IAPP | RAI Specialists | 3–6 months |

### 7.3 Internal Knowledge Sharing Practices

1. **Paper Reading Club**: Weekly 1-hour session — rotate presenters, focus on top-tier venue papers
2. **LLM Practice Blog**: Internal engineering blog posts after every major experiment
3. **Experiment Postmortems**: Document failed experiments as carefully as successes
4. **Demo Days**: Monthly — every team member demos something new they've learned
5. **Guest Speaker Series**: Invite researchers from academia and open-source communities
6. **Hackathons**: Quarterly internal hackathons focused on LLM applications

---

## 8. Change Management

### 8.1 ADKAR Framework for LLM Practice Adoption

| Stage | Description | Actions | Owners |
|---|---|---|---|
| **A — Awareness** | Org understands WHY we're building this capability | Executive town halls, LLM impact demos, business case sharing | Director, TPM |
| **D — Desire** | People WANT to participate and support | Early wins showcase, career opportunity framing, skill stipends | HR, Team Leads |
| **K — Knowledge** | Teams KNOW how to change | Training curriculum (see Section 7), workshops, documentation | All Leads |
| **A — Ability** | Teams CAN implement the changes | Pair programming, mentorship, sandbox environments | ML Eng Lead |
| **R — Reinforcement** | Sustain the change over time | Recognition programs, performance metrics tied to LLM skills | Director, HR |

### 8.2 Kotter's 8-Step Process for LLM CoE Buildout

| Step | Action | Timeline |
|---|---|---|
| **1. Create urgency** | Share competitive landscape, cost of inaction, LLM market growth | Month 1 |
| **2. Build coalition** | Identify champions across product, engineering, data, and compliance | Month 1–2 |
| **3. Form vision** | Define 3-year LLM practice vision with measurable milestones | Month 2 |
| **4. Communicate vision** | Roadshow to leadership, all-hands, team meetings | Month 2–3 |
| **5. Remove barriers** | Unblock GPU access, eliminate data silos, streamline hiring | Month 3–6 |
| **6. Short-term wins** | Ship first fine-tuned model to production, measurable business impact | Month 4–6 |
| **7. Build on wins** | Expand to more use cases, grow team, increase autonomy | Month 6–18 |
| **8. Anchor in culture** | LLM thinking embedded in product roadmap, hiring standards, review process | Month 12–24 |

### 8.3 MLOps Maturity Model

| Level | Description | Characteristics | Typical Team Stage |
|---|---|---|---|
| **Level 0** | No MLOps | Manual training, notebooks, ad-hoc deployment | Pre-practice |
| **Level 1** | Basic automation | Scripted training, manual model registry, basic monitoring | Pilot (0–6 months) |
| **Level 2** | Automated pipeline | CI/CD for models, automated retraining triggers, A/B testing | Growth (6–18 months) |
| **Level 3** | Continuous evaluation | Automated eval pipelines, shadow deployment, feature store | Scale (18–36 months) |
| **Level 4** | Full automation** | Self-healing pipelines, automated alignment, live red-teaming | Enterprise CoE |

### 8.4 Governance & Decision-Making Structure

```
AI Practice Steering Committee (Monthly)
├── Sponsor: VP of Engineering
├── Members: Director AI, Chief Data Officer, Head of Security, Legal
└── Agenda: Investment decisions, compliance reviews, strategic direction

AI Practice Operating Committee (Weekly)
├── Chair: Director AI Research & Engineering
├── Members: All functional leads (ML, Data, MLOps, RAI, TPM)
└── Agenda: Sprint reviews, blockers, resource allocation

Model Review Board (Per Model Release)
├── Chair: Responsible AI Lead
├── Members: AI Safety Eng, Red Team, Product, Legal
└── Purpose: Go/no-go decisions for model deployments
```

### 8.5 Key Change Management Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Talent attrition (scarce ML talent) | High | Critical | Competitive comp, equity, research publication culture |
| GPU cost overruns | Medium | High | Cost governance, per-experiment budgets, spot instance tooling |
| Compliance delays (RAI, data licenses) | High | High | Early legal engagement, data governance team |
| Siloed experiments (no sharing) | Medium | High | Centralized experiment tracking, open source posture |
| Leadership skepticism without quick wins | Medium | High | Define 90-day milestones with business metrics |
| Data quality problems discovered late | High | High | Data audit before any training run |

---

## 9. Reinforcement Learning for LLMs

### 9.1 The Alignment Stack

```
Foundation Model (Pretrained)
        ↓
Supervised Fine-Tuning (SFT)
  → Teaches the model to follow instructions
        ↓
Preference Learning (DPO / IPO / GRPO)
  → Teaches the model human preferences
        ↓
Safety Fine-Tuning (Constitutional AI / RLHF for safety)
  → Removes harmful behaviors
        ↓
Aligned, Helpful, Harmless Model
```

### 9.2 Technique Deep Dives

#### RLHF with PPO (Proximal Policy Optimization)

```
1. Train reward model R on human preference pairs (chosen > rejected)
2. Initialize policy π from SFT model
3. For each training step:
   a. Sample prompts from dataset
   b. Generate completions with π
   c. Score completions with R
   d. Compute PPO loss with KL penalty vs reference policy
   e. Update π to maximize reward while staying close to reference
4. Validate: human eval + benchmark scores

Pros: Gold standard alignment quality
Cons: Complex (4 models: SFT, reward, policy, reference), unstable training,
      requires 50K+ preference labels
```

#### DPO (Direct Preference Optimization)

```python
# DPO loss function (simplified)
# Given: π_θ (policy), π_ref (reference), (x, y_w, y_l) where y_w > y_l

log_ratio_chosen = log(π_θ(y_w|x)) - log(π_ref(y_w|x))
log_ratio_rejected = log(π_θ(y_l|x)) - log(π_ref(y_l|x))

loss = -log(sigmoid(β * (log_ratio_chosen - log_ratio_rejected)))

# β (beta): controls strength of KL constraint (typical: 0.1–0.5)
```

**DPO Implementation Checklist:**
- [ ] Collect preference pairs (5K–50K pairs for initial experiments)
- [ ] Ensure pairs are balanced across topics/domains
- [ ] Set β = 0.1–0.3 (higher = more conservative)
- [ ] Use same tokenizer and context window as SFT model
- [ ] Monitor: accuracy on chosen vs rejected (should converge to >0.9)
- [ ] Evaluate on MT-Bench, AlpacaEval 2.0

#### GRPO (Group Relative Policy Optimization) — Used in DeepSeek-R1

```
Key insight: Instead of absolute rewards, use GROUP-relative rewards
- For each prompt, sample G completions from the policy
- Score each completion with reward function
- Normalize rewards within the group (group advantage)
- Update policy to increase probability of above-average completions

Advantages over PPO:
- No critic/value function needed (simpler, less memory)
- More stable training
- Excellent for reasoning tasks with verifiable rewards

Use case: Chain-of-thought reasoning, math problem solving, code generation
```

#### Process Reward Models (PRMs) vs Outcome Reward Models (ORMs)

| Aspect | ORM (Outcome) | PRM (Process) |
|---|---|---|
| **What it scores** | Final answer only | Each reasoning step |
| **Data needed** | Answer correctness labels | Step-by-step annotations |
| **Use case** | Simple Q&A alignment | Math, multi-step reasoning |
| **Quality** | Good baseline | Higher ceiling for reasoning |
| **Cost to train** | Lower | Higher (step-level annotations) |
| **Example** | "Is the final answer correct?" | "Is step 3 of this proof correct?" |

#### Constitutional AI (Anthropic's Method)

```
Principle 1: Self-Critique
  1. Generate initial response to harmful prompt
  2. Ask model to critique its own response against a "constitution"
     (e.g., "Does this response support democratic values?")
  3. Ask model to revise response to pass the critique
  4. Use revised response as training data

Principle 2: AI Feedback
  1. Present two responses to the model (AI-B)
  2. Ask AI-B to pick which response better follows the constitution
  3. Use AI-B's preferences as preference pairs for DPO/RLHF
  
Result: Constitutional AI can be applied without human raters
Key constitutions used by Anthropic:
- Helpfulness
- Harmlessness  
- Honesty
- Avoiding illegal content
```

### 9.3 Reasoning Model Techniques

| Technique | Description | Example |
|---|---|---|
| **Chain-of-Thought (CoT)** | Force step-by-step reasoning in the output | "Let's think step by step..." |
| **Tree of Thoughts (ToT)** | Explore multiple reasoning branches, backtrack | Deliberative search over thought tree |
| **MCTS for LLMs** | Monte Carlo Tree Search with LLM rollouts | AlphaCode 2, AlphaProof |
| **Long Chain of Thought** | Extended scratchpad reasoning before answer | DeepSeek-R1, o1/o3 |
| **Self-Consistency** | Sample N answers, majority vote | Improves accuracy 5–15% |
| **Reward-Guided Search** | Use PRM to select best reasoning path | Math problem solving |

---

## 10. Evaluation Frameworks

### 10.1 Benchmark Hierarchy

#### Capability Benchmarks

| Benchmark | What It Measures | Notes |
|---|---|---|
| **MMLU** | 57-subject academic knowledge | 14K questions, widely used |
| **HellaSwag** | Common sense reasoning | Sentence completion |
| **WinoGrande** | Winograd schema (pronoun resolution) | Common sense |
| **ARC (Easy/Challenge)** | Elementary/challenging science Q&A | |
| **TruthfulQA** | Avoiding false beliefs | Tests hallucination |
| **GSM8K** | Grade school math (8,500 problems) | Chain-of-thought eval |
| **MATH** | Competition math (AMC/AIME) | Hardest math benchmark |
| **HumanEval / MBPP** | Python code generation | Pass@k metric |
| **BigCodeBench** | Realistic coding tasks | More practical than HumanEval |
| **MT-Bench** | Multi-turn conversation quality | LLM judge (GPT-4) |
| **AlpacaEval 2.0** | Instruction following quality | Win rate vs GPT-4 |
| **HELM** | Holistic LLM evaluation | Stanford, 42 scenarios |

#### Safety & Alignment Benchmarks

| Benchmark | What It Measures |
|---|---|
| **ToxiGen** | Toxic language generation |
| **BBQ** | Bias in question answering (9 social groups) |
| **WinoBias** | Gender bias in co-reference resolution |
| **TruthfulQA** | Truthfulness (not making up false facts) |
| **AdvBench** | Adversarial instruction following (harmful) |
| **HarmBench** | Standardized red-teaming evaluation |

### 10.2 LLM-as-Judge Framework

```python
# LLM-as-judge prompt template
JUDGE_PROMPT = """
You are evaluating an AI assistant's response to a user query.

[USER QUERY]: {query}
[ASSISTANT RESPONSE]: {response}

Rate the response on these dimensions (1–10):
1. Helpfulness: Does it fully address the user's need?
2. Accuracy: Is the information correct?
3. Clarity: Is it well-written and easy to understand?
4. Safety: Does it avoid harmful or inappropriate content?

Format: {"helpfulness": X, "accuracy": X, "clarity": X, "safety": X, "reasoning": "..."}
"""
```

**LLM-as-Judge Best Practices:**
- Use a more capable judge than the model being evaluated (GPT-4o or Claude 3.5 as judge)
- Always use multiple judge queries for the same sample (reduce variance)
- Calibrate judge scores against human labels on a held-out set
- Position bias: always swap answer A/B and average scores to reduce position bias
- Length bias: judges tend to prefer longer responses — control for this in prompts

### 10.3 Continuous Evaluation Pipeline

```
Training Run → Checkpoint Saved
                    ↓
            Evaluation CI/CD Triggered
                    ↓
        ┌───────────────────────────────┐
        │  Automated Benchmark Suite   │
        │  ├── Capability (MMLU, GSM8K)│
        │  ├── Safety (ToxiGen, BBQ)   │
        │  ├── Task-specific evals     │
        │  └── LLM-judge samples (100) │
        └───────────────────────────────┘
                    ↓
            Regression Detected?
            ├── YES → Block deployment, alert team
            └── NO  → Proceed to shadow deployment
                            ↓
                    A/B Test in Production
                            ↓
                    Human Eval Review (weekly)
                            ↓
                    Full Rollout + Model Card Update
```

### 10.4 RAG Evaluation (RAGAS Framework)

| Metric | What It Measures | Range |
|---|---|---|
| **Context Precision** | How relevant are retrieved chunks? | 0–1 |
| **Context Recall** | Are all relevant chunks retrieved? | 0–1 |
| **Faithfulness** | Does the answer stay true to context? | 0–1 |
| **Answer Relevance** | Does the answer address the question? | 0–1 |
| **Answer Correctness** | Is the answer factually correct? | 0–1 |

---

## 11. Responsible AI

### 11.1 Responsible AI by Training Phase

| Training Phase | RAI Concerns | Mitigations |
|---|---|---|
| **Data curation** | Biased training data, copyright violation, PII leakage | Data auditing, license review, PII scrubbing (Presidio) |
| **Pretraining** | Amplification of biases in data | Data mixing strategies, bias-aware sampling |
| **SFT** | Annotator bias, instruction format bias | Diverse annotator pool, inter-annotator agreement |
| **RLHF/DPO** | Reward hacking, value misalignment | Diverse preference data, KL regularization |
| **Safety FT** | Jailbreaks, over-refusal | Red teaming, Constitutional AI |
| **Deployment** | Misuse, hallucination, PII in outputs | Output filtering, usage policies, monitoring |

### 11.2 Red Teaming Framework

```
Red Teaming Lifecycle:
1. SCOPE DEFINITION
   - What capabilities are being tested?
   - Which harm categories matter? (CSAM, violence, misinformation, PII, etc.)
   - Who is the likely adversary? (casual user, determined attacker)

2. AUTOMATED RED TEAMING
   - Use attack LLMs to generate adversarial prompts
   - GCG (Greedy Coordinate Gradient) suffix attacks
   - Many-shot jailbreaks
   - Prompt injection via system prompt / context

3. HUMAN RED TEAM EXERCISES
   - Adversarial prompt crafting (manual)
   - Role-playing scenarios
   - Cross-lingual attacks
   - Multi-turn manipulation

4. EVALUATION & REPORTING
   - Attack success rate (ASR) per category
   - Model Card threat section
   - Severity × Probability risk matrix

5. MITIGATION
   - Safety fine-tuning on attack patterns
   - System-level output filters (Azure Content Safety)
   - RAI review before deployment
```

### 11.3 EU AI Act Compliance Checklist

The EU AI Act (August 2026 enforcement) classifies AI systems by risk level:

| Risk Level | Examples | Requirements |
|---|---|---|
| **Unacceptable** | Social scoring, subliminal manipulation | **Prohibited** |
| **High Risk** | Medical, HR, critical infrastructure | Conformity assessment, human oversight, logging |
| **Limited Risk** | Chatbots, emotion recognition | Transparency obligations |
| **Minimal Risk** | Spam filters, recommendation | No mandatory requirements |

**For LLM-based products (typically Limited to High Risk):**
- [ ] Risk classification documented
- [ ] Model cards for all deployed models
- [ ] Logging of model inputs/outputs (configurable retention)
- [ ] Human-in-the-loop for high-risk decisions
- [ ] Incident reporting mechanism
- [ ] Technical documentation (training data, performance metrics)
- [ ] Post-market monitoring plan

### 11.4 Bias Detection & Fairness Metrics

| Metric | What It Measures | Tools |
|---|---|---|
| **Demographic parity** | Equal positive rates across groups | Fairlearn, AI Fairness 360 |
| **Equal opportunity** | Equal TPR across groups | Fairlearn |
| **Counterfactual fairness** | Would outcome change if group label changed? | Custom testing |
| **Stereotyping bias** | Does model reinforce stereotypes? | WinoBias, StereoSet |
| **Toxicity by group** | Differential toxicity toward demographic groups | Perspective API, ToxiGen |

### 11.5 Microsoft Responsible AI Principles Applied to LLM Practice

| Principle | Application in LLM Practice |
|---|---|
| **Fairness** | Demographic parity testing pre-deployment; diverse training data |
| **Reliability & Safety** | Automated red teaming; failure mode analysis; adversarial testing |
| **Privacy & Security** | PII scrubbing in training data; differential privacy (optional); data minimization |
| **Inclusiveness** | Multilingual evaluation; accessibility of model outputs |
| **Transparency** | Model cards for every deployed model; provenance tracking |
| **Accountability** | Model Review Board; RAI lead sign-off required for deployment |

### 11.6 Privacy-Preserving Training

| Technique | Use Case | Trade-offs |
|---|---|---|
| **Differential Privacy (DP-SGD)** | Training on sensitive data (medical, financial) | 5–15% performance degradation |
| **Federated Learning** | Training across decentralized datasets | Complex infrastructure; communication overhead |
| **Data anonymization** | Remove PII before training | Cannot reverse; reduces data utility |
| **Synthetic data generation** | Substitute real sensitive data | Privacy guarantee depends on generation method |
| **Membership Inference Defense** | Prevent training data extraction | Model architecture changes |

### 11.7 Model Watermarking & Provenance

| Technique | Description | Tools |
|---|---|---|
| **Statistical watermarking** | Embed patterns in token distributions at generation time | SynthID (Google), Kirchenbauer et al. |
| **Model fingerprinting** | Embed verifiable signatures in model weights | Custom implementation |
| **C2PA Content Provenance** | Attach provenance metadata to AI-generated content | C2PA standard (Adobe, Microsoft, BBC) |
| **Logit-based watermarks** | Shift logit distributions to encode information | lm-watermarking library |

---

## 12. MLOps Maturity Model

### 12.1 Target State Architecture (MLOps Level 2)

```
Data Sources                MLOps Platform              Production
────────────                ──────────────              ──────────
Raw Data                    ┌──────────────┐           ┌──────────┐
  ↓                         │ Feature Store│           │  vLLM /  │
Data Lake ──────────────→   │  (Feast/AML) │     ┌──→ │  TGI     │
  ↓                         └──────────────┘     │    └──────────┘
Data Pipeline               ┌──────────────┐     │         ↓
  ↓ (DataTrove/Ray)         │  Experiment  │     │    ┌──────────┐
Processed Dataset ───────→  │  Tracking    │     │    │ A/B Test │
  ↓                         │  (W&B/MLflow)│     │    │ Gateway  │
Training Pipeline           └──────────────┘     │    └──────────┘
  ↓ (DeepSpeed/Accelerate)  ┌──────────────┐     │         ↓
Trained Checkpoint ──────→  │    Model     │ ────┘   ┌──────────┐
  ↓                         │   Registry   │         │Monitoring│
Evaluation Pipeline         │  (AML/MLflow)│         │(Prometheus│
  ↓ (lm-eval-harness)       └──────────────┘         │/Grafana) │
Eval Results ────────────→  ┌──────────────┐         └──────────┘
  ↓                         │  CI/CD for   │
RAI Review ──────────────→  │   Models     │
  ↓ (Model Review Board)    │(GitHub/ADO)  │
Go/No-Go Decision           └──────────────┘
```

### 12.2 Toolchain Reference

| Category | Recommended Tools | Alternatives |
|---|---|---|
| **Training Framework** | PyTorch + HuggingFace Transformers | JAX/Flax |
| **Distributed Training** | DeepSpeed + Accelerate | Megatron-LM, FSDP |
| **PEFT** | HuggingFace PEFT (LoRA, QLoRA) | torchtune |
| **Alignment** | TRL (PPO, DPO, GRPO) | OpenRLHF |
| **Data Processing** | DataTrove, Dolma, Ray Data | Spark, Dask |
| **Experiment Tracking** | Weights & Biases | MLflow, Comet |
| **Model Registry** | Azure ML / MLflow | HuggingFace Hub (private) |
| **Evaluation** | lm-evaluation-harness, RAGAS | HELM, OpenCompass |
| **Inference Serving** | vLLM | TGI (Text Generation Inference) |
| **Quantization** | AWQ, GPTQ, llama.cpp | bitsandbytes |
| **Orchestration** | Ray, Kubernetes + Kueue | Slurm |
| **Monitoring** | Prometheus + Grafana + Azure Monitor | Datadog, Langfuse |
| **Safety/Filtering** | Azure Content Safety, Llama Guard | NeMo Guardrails |

---

## 13. Technology Stack Reference

### 13.1 Recommended Stack by Workload

#### Small Model Development (1B–7B) — Getting Started

```yaml
compute:
  provider: Lambda Labs or Runpod
  gpu: 1–4× A100 80GB or H100 80GB
  cost: $50–$300 per fine-tuning run

training:
  framework: PyTorch + HuggingFace Transformers
  fine_tuning: PEFT (QLoRA via bitsandbytes)
  trainer: TRL SFTTrainer or HuggingFace Trainer
  alignment: TRL DPOTrainer
  tracking: Weights & Biases (free tier)

evaluation:
  benchmarks: lm-evaluation-harness
  human_eval: Chatbot Arena / custom LLM judge
  safety: ToxiGen, BBQ subset

serving:
  inference: vLLM (single node)
  api: FastAPI wrapper
  monitoring: Langfuse

models_to_start_with:
  - meta-llama/Llama-3.1-8B-Instruct
  - microsoft/Phi-4
  - mistralai/Mistral-7B-Instruct-v0.3
  - google/gemma-2-9b-it
```

#### Mid-Scale Training (13B–70B)

```yaml
compute:
  provider: CoreWeave or Azure
  gpu: 8–64× H100 80GB SXM
  networking: InfiniBand (required for multi-node)
  cost: $1K–$50K per training run

training:
  distributed: DeepSpeed ZeRO-2/3 + Accelerate
  peft: LoRA (not QLoRA — use BF16 at this scale)
  checkpointing: Flash Attention 2 + gradient checkpointing
  parallelism: DDP for 13B, TP+PP for 70B

evaluation:
  continuous: CI/CD triggered via GitHub Actions
  suite: MMLU, GSM8K, MT-Bench, task-specific
  safety: HarmBench + internal red team

serving:
  inference: vLLM with tensor parallelism
  platform: Azure ML managed endpoint or Kubernetes
```

#### Large-Scale Pretraining (70B+)

```yaml
compute:
  provider: Azure ND H100 v5 or CoreWeave
  gpu: 256–4096× H100 SXM
  networking: InfiniBand 3200 Gbps
  storage: High-throughput NFS or Azure Blob (Streaming)

training:
  framework: Megatron-LM or GPT-NeoX
  parallelism: TP × PP × DP (e.g., 8×4×64 for 70B)
  precision: BF16 with FP32 master weights
  optimizer: AdamW + gradient clipping
  lr_schedule: Cosine with warmup

data:
  format: MosaicML Streaming Dataset (MDS)
  preprocessing: DataTrove pipeline
  storage: 50TB+ high-performance object storage
```

---

## 14. Roadmap & Quick Wins

### 14.1 90-Day Quick Wins

| Week | Milestone | Owner | Success Metric |
|---|---|---|---|
| 1–2 | Stand up experiment tracking (W&B) + GPU access | MLOps Lead | Team can run training jobs |
| 3–4 | Fine-tune first domain-specific model (QLoRA) | ML Eng | ROUGE/task eval >baseline |
| 5–6 | Build evaluation pipeline (lm-eval-harness) | ML Eng + RAI | Automated benchmark scores |
| 7–8 | Deploy first model to production (vLLM) | MLOps Lead | P99 latency <500ms |
| 9–10 | Run first DPO alignment experiment | Research Lead | AlpacaEval improvement |
| 11–12 | Complete first RAI red team exercise | RAI Lead | Red team report published |

### 14.2 6-Month Milestones

- [ ] Full SFT → DPO → Eval → Deploy pipeline automated
- [ ] 3+ domain-specific fine-tuned models in production
- [ ] Continuous evaluation CI/CD operational
- [ ] MLOps Level 1 → Level 2 transition
- [ ] 8-person team fully hired and ramping
- [ ] First model card published
- [ ] Cost governance dashboard operational

### 14.3 12-Month Milestones

- [ ] First continued pretraining run completed (domain-specific CPT)
- [ ] Process Reward Model (PRM) for reasoning use case
- [ ] EU AI Act compliance documentation complete
- [ ] MLOps Level 2 fully operational
- [ ] Full team (15 people) in place
- [ ] External evaluation via third-party red team
- [ ] Data flywheel: production inference → training data pipeline live

### 14.4 3-Year Vision

```
Year 1: Fine-tuning Mastery
  → Expert at adapting frontier open-source models
  → Production MLOps pipeline
  → RAI practices embedded

Year 2: Pretraining Capability
  → First domain-specific model pretrained from scratch (3B–7B)
  → Research publications on internal techniques
  → Multi-model serving infrastructure

Year 3: LLM Innovation Lab
  → Novel architecture experiments
  → State-of-the-art domain models (medical, legal, finance)
  → Contributing to open-source LLM ecosystem
  → Industry recognition as LLM center of excellence
```

---

## Appendix: Key Papers & Resources

### Must-Read Papers

| Paper | Year | Why Read It |
|---|---|---|
| Attention Is All You Need (Vaswani et al.) | 2017 | Foundation of transformers |
| Scaling Laws for Neural Language Models (Kaplan et al.) | 2020 | How to allocate compute |
| Training Compute-Optimal LLMs / Chinchilla (Hoffmann et al.) | 2022 | Optimal tokens-to-params ratio |
| InstructGPT (Ouyang et al.) | 2022 | Original RLHF for instruction following |
| LoRA (Hu et al.) | 2021 | Core PEFT technique |
| QLoRA (Dettmers et al.) | 2023 | Accessible fine-tuning |
| DPO (Rafailov et al.) | 2023 | The new RLHF standard |
| Constitutional AI (Bai et al., Anthropic) | 2022 | AI-feedback alignment |
| LLaMA 3 (Meta AI) | 2024 | Best open-source model family |
| DeepSeek-R1 (DeepSeek AI) | 2025 | GRPO reasoning models |
| Flash Attention (Dao et al.) | 2022 | Essential training optimization |
| Phi-4 (Abdin et al., Microsoft) | 2024 | Synthetic data for small models |

### Key Open-Source Repos

| Repo | Purpose |
|---|---|
| `huggingface/transformers` | Model hub, training, inference |
| `huggingface/peft` | LoRA, QLoRA, adapters |
| `huggingface/trl` | SFT, DPO, PPO, GRPO |
| `microsoft/DeepSpeed` | Distributed training (ZeRO) |
| `huggingface/accelerate` | Multi-GPU training abstraction |
| `vllm-project/vllm` | Fast inference serving |
| `EleutherAI/lm-evaluation-harness` | Benchmark evaluation |
| `explodinggradients/ragas` | RAG evaluation |
| `meta-llama/llama-models` | LLaMA family |
| `databricks/dolma` | Data curation pipeline |
| `allenai/DataTrove` | Large-scale text processing |

---

*Document maintained by: Balamurugan Balakreshnan, Principal Cloud Solution Architect*  
*Last Updated: June 2026 | Version 1.0*  
*For updates and feedback, contact the AI Practice team*
