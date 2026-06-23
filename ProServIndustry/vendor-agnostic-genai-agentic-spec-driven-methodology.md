# Vendor-Agnostic GenAI & Agentic AI Platform

## A Spec-Driven Methodology and Open-Source Component Reference

**Author context:** Reference architecture for hyperscaler-agnostic and AI-neocloud-portable GenAI / agentic systems
**Version:** 1.0
**Last updated:** 2026-06-23
**Status:** Working reference — review the *Constitution* and *Portability Tenets* sections before adopting

> **Scope.** This document specifies (1) a layered, vendor-neutral reference architecture for generative-AI and agentic applications, (2) a curated catalog of *open-source* components for every layer, with agnostic-design notes and lock-in traps, (3) a coverage of **AI neoclouds** as first-class compute targets alongside AWS / Azure / GCP, and (4) a **spec-driven development (SDD) methodology** — adapted from the Specify → Plan → Tasks → Implement loop — extended for the probabilistic, eval-gated nature of GenAI systems.
>
> **A note on currency.** The GenAI tooling landscape moves monthly. Component *names and roles* are stable; *licenses, version numbers, and benchmark figures* drift. Treat every license note below as "verify at adoption time," and pin versions in your specs. Nothing here should be read as a current benchmark claim.

---

## Table of Contents

1. [Why Vendor-Agnostic, and Why Spec-Driven](#1-why-vendor-agnostic-and-why-spec-driven)
2. [Design Principles & Portability Tenets](#2-design-principles--portability-tenets)
3. [The Portability Problem: Where Lock-In Hides](#3-the-portability-problem-where-lock-in-hides)
4. [Layered Reference Architecture](#4-layered-reference-architecture)
5. [Component-by-Component Analysis (Open Source)](#5-component-by-component-analysis-open-source)
6. [AI Neoclouds as a Compute Target](#6-ai-neoclouds-as-a-compute-target)
7. [The Spec-Driven Methodology (Extended for GenAI)](#7-the-spec-driven-methodology-extended-for-genai)
8. [Specification Templates](#8-specification-templates)
9. [Portability Scorecard & Decision Framework](#9-portability-scorecard--decision-framework)
10. [Reference Stack Blueprints (Three Tiers)](#10-reference-stack-blueprints-three-tiers)
11. [Anti-Patterns & Lock-In Traps](#11-anti-patterns--lock-in-traps)
12. [Migration & Exit Playbook](#12-migration--exit-playbook)
13. [Appendix A: Master Component Catalog](#appendix-a-master-component-catalog)
14. [Appendix B: Glossary](#appendix-b-glossary)

---

## 1. Why Vendor-Agnostic, and Why Spec-Driven

### 1.1 The agnostic thesis

A GenAI platform that is wired to one hyperscaler's proprietary primitives — a managed agent runtime, a closed model endpoint, a proprietary vector index, a serverless event glue — inherits that vendor's pricing power, region availability, model roadmap, and rate limits. The cost of that coupling is invisible until you need to:

- **Arbitrage GPU price/availability** by moving inference to a neocloud (CoreWeave, Crusoe, Lambda, Nebius) where H100/H200/B200 capacity is cheaper or simply *available*.
- **Meet data-residency / sovereignty** constraints a given region can't satisfy.
- **Swap models** (closed → open-weight, or vendor A → vendor B) without rewriting the application.
- **Negotiate** — leverage only exists when you can credibly leave.

Vendor-agnostic design treats *every* cloud — hyperscaler or neocloud — as an interchangeable substrate beneath stable, open interfaces. The application binds to **portable contracts** (OpenAI-compatible chat API, S3-compatible object storage, the Kubernetes API, OpenTelemetry, MCP/A2A), never to a provider SKU.

### 1.2 Why spec-driven is the natural fit

Agentic systems are **non-deterministic**. The same prompt yields different token streams; the same plan yields different tool calls. "Vibe coding" an agent — prompting until it looks right — produces systems whose behavior lives in chat history that resets when the context window fills. Spec-Driven Development (SDD) makes a **written specification the source of truth**, and the implementation a *generated, regenerable artifact* measured against that spec.

For an agnostic platform this is doubly valuable: the spec captures **intent and contracts** independent of any vendor, so the *same spec can target AWS today and a neocloud tomorrow*. The methodology in Section 7 extends the canonical four-phase loop (Specify → Plan → Tasks → Implement) with two phases that GenAI specifically demands: **Evaluate** (the spec is only "done" when eval gates pass) and **Operate** (drift, cost, and safety are continuous specifications, not one-time checks).

---

## 2. Design Principles & Portability Tenets

These ten tenets are the **constitution** of the platform. Every component choice and every spec must comply, or document an explicit, time-boxed exception.

| # | Tenet | What it means in practice |
|---|-------|---------------------------|
| **T1** | **Open standards over proprietary APIs** | Bind to OpenAI-compatible inference, S3-compatible storage, Kubernetes, OpenTelemetry, SQL, MCP, A2A. A provider that only speaks its own dialect sits *behind* an adapter. |
| **T2** | **Open weights as the default model posture** | The architecture must run end-to-end on open-weight models (Llama, Mistral/Mixtral, Qwen, DeepSeek, Gemma, etc.). Closed models are a *pluggable upgrade*, never a hard dependency. |
| **T3** | **Containers + Kubernetes as the portable runtime** | Anything that runs is an OCI image scheduled by Kubernetes. No reliance on a single cloud's proprietary serverless or managed-agent runtime for core logic. |
| **T4** | **Infrastructure as Code, provider-modularized** | IaC with a clean seam between provider-specific modules and provider-neutral application modules. Swapping clouds changes modules, not the app. |
| **T5** | **Data gravity is managed, not assumed** | Object storage via S3 API; vectors in a self-hostable engine; relational state in portable Postgres. No data trapped in a proprietary store with no bulk export. |
| **T6** | **Stateless application tier; externalized state** | Agents, orchestrators, and gateways are stateless and horizontally scalable. Memory, sessions, and checkpoints live in portable backing stores. |
| **T7** | **One model-access seam (the gateway)** | All model calls route through a single OpenAI-compatible gateway. The app never imports a vendor SDK directly. |
| **T8** | **Observability is open by construction** | Traces, metrics, and logs emit OpenTelemetry / OTLP. Eval and tracing tools consume open formats; no telemetry locked in a proprietary console. |
| **T9** | **Protocol-based tool & agent interop** | Tools are exposed via MCP; cross-agent communication via A2A. Agents and tools are composable across frameworks and vendors. |
| **T10** | **Exit cost is a tracked, tested metric** | "How long to stand this up on a different cloud?" is measured (target ≤ 2 weeks for the core path), with a rehearsed migration runbook. |

---

## 3. The Portability Problem: Where Lock-In Hides

Lock-in is rarely a single decision; it accumulates at seams. Map every seam to an **open contract** and a **fallback**.

| Seam | Proprietary trap (examples) | Portable contract | Open fallback |
|------|-----------------------------|-------------------|---------------|
| **Model access** | Provider-native SDK, proprietary tool-calling schema | OpenAI-compatible `/v1/chat/completions` + tools | Self-hosted vLLM/SGLang behind the same API |
| **Embeddings** | Closed embedding endpoint, fixed dimensionality | Pluggable embedding interface; store model + dim in metadata | Self-hosted `bge`, `e5`, `gte`, Nomic, Jina via TEI |
| **Vector search** | Managed index with no bulk export | Standard upsert/query + re-embed pipeline | Milvus / Qdrant / Weaviate / pgvector |
| **Orchestration** | Managed agent runtime / proprietary workflow DSL | Code-first orchestration on K8s; portable workflow engine | LangGraph / CrewAI / Temporal |
| **State & memory** | Proprietary session store | Postgres + Redis + object storage | Self-hosted equivalents |
| **Eventing/queues** | Cloud-specific queue/bus semantics | AMQP/Kafka protocol; CloudEvents | Kafka/Redpanda, NATS, RabbitMQ |
| **Compute** | Instance SKUs, proprietary autoscaler | Kubernetes node pools + GPU operator | Any CNCF-conformant cluster |
| **Identity** | Cloud-native IAM only | OIDC / SPIFFE/SPIRE for workload identity | Keycloak, Dex |
| **Telemetry** | Proprietary APM agent | OpenTelemetry / OTLP | Grafana stack, SigNoz |
| **IaC state** | Provider-coupled tooling | Modular IaC, remote state in S3-compatible store | OpenTofu / Terraform + Crossplane |

**Rule of thumb:** if a component cannot be replaced *without touching application code*, the seam is wrong — wrap it.

---

## 4. Layered Reference Architecture

The platform is a stack of independently replaceable layers. Each arrow is an **open protocol**, not a vendor binding.

```
┌─────────────────────────────────────────────────────────────────────┐
│  L9  EXPERIENCE        Web / API / chat / IDE / voice surfaces        │
│                        (REST/gRPC/WebSocket; A2A for agent-to-agent)  │
├─────────────────────────────────────────────────────────────────────┤
│  L8  AGENT ORCHESTRATION   Planner · multi-agent graph · HITL ·       │
│                            durable workflows  (LangGraph/CrewAI/      │
│                            AutoGen + Temporal)                        │
├─────────────────────────────────────────────────────────────────────┤
│  L7  TOOL & PROTOCOL    MCP servers (tools/resources) · A2A ·         │
│                         code sandbox · function registry             │
├─────────────────────────────────────────────────────────────────────┤
│  L6  GUARDRAILS & POLICY   I/O validation · PII · jailbreak ·         │
│                            safety classifiers · OPA policy            │
├─────────────────────────────────────────────────────────────────────┤
│  L5  RETRIEVAL / RAG    Chunking · embeddings · hybrid search ·       │
│                         rerank · context assembly                     │
├─────────────────────────────────────────────────────────────────────┤
│  L4  MODEL GATEWAY      OpenAI-compatible routing · fallback ·        │
│                         caching · rate-limit · cost/budget · keys     │
├─────────────────────────────────────────────────────────────────────┤
│  L3  INFERENCE / SERVING   vLLM · SGLang · TGI · TensorRT-LLM ·       │
│                            Ollama · KServe/Ray Serve                  │
├─────────────────────────────────────────────────────────────────────┤
│  L2  DATA & STATE       Object (S3 API) · Postgres · Redis ·          │
│                         Vector DB · Kafka/NATS · feature/cache        │
├─────────────────────────────────────────────────────────────────────┤
│  L1  COMPUTE PLATFORM   Kubernetes · GPU operator · Ray ·            │
│                         autoscaling · scheduling (Volcano/Kueue)      │
├─────────────────────────────────────────────────────────────────────┤
│  L0  INFRASTRUCTURE     Any hyperscaler OR AI neocloud, via           │
│                         OpenTofu/Terraform + Crossplane modules       │
└─────────────────────────────────────────────────────────────────────┘
        ⟂ CROSS-CUTTING: Observability/Eval · CI-CD · Secrets/Identity ·
          FinOps · Prompt & Model Registry · Data/Model Governance
```

**Reading the stack agnostically:** L0–L3 are where neoclouds compete hardest (raw GPU + serving). L4–L9 are pure software you own and run on *any* L1. The gateway (L4) is the single chokepoint that lets L5–L9 stay ignorant of which cloud, which model, and which serving engine is underneath.

---

## 5. Component-by-Component Analysis (Open Source)

For each layer: the **purpose**, the **open-source options**, a **selection lens**, and the **agnostic-design note** (how to keep the seam portable). Licenses are indicative — **verify before adoption**, several have changed over time.

### 5.1 L3 — Inference & Serving Engines

**Purpose:** load model weights, manage GPU/KV memory, batch requests, stream tokens. This is the throughput/latency/cost-defining layer.

| Engine | Best at | License (verify) | Notes |
|--------|---------|------------------|-------|
| **vLLM** | General-purpose high-throughput serving; broad model + hardware support; de-facto OSS default | Apache-2.0 | PagedAttention, continuous batching, prefix caching, OpenAI-compatible server, tensor/pipeline parallel. The safe agnostic baseline. |
| **SGLang** | High-throughput structured generation, RadixAttention prefix reuse, agent/tool-heavy workloads | Apache-2.0 | Strong for complex multi-call agent traffic and constrained decoding. |
| **TensorRT-LLM** | Peak latency/throughput on NVIDIA when you can invest in compilation | Apache-2.0 | Highest performance ceiling on NVIDIA; more build complexity; NVIDIA-specific — keep behind the gateway. |
| **Hugging Face TGI** | Mature, integration-rich serving | Verify (has shifted over time) | Solid production server; confirm current license posture. |
| **Ollama** | Local/dev, edge, single-node, quick model pulls | MIT | Excellent DX and laptop/edge; exposes OpenAI-compatible API — same contract as the big engines. |
| **llama.cpp** | CPU / quantized / edge / Apple Silicon | MIT | GGUF quantization; runs where GPUs don't. |
| **KServe / Ray Serve** | *Serving control plane* on top of engines (autoscale, canary, multi-model) | Apache-2.0 | Not engines themselves — they orchestrate engines on K8s. Use for production rollout patterns. |

**Selection lens:** start on **vLLM** (portable default). Add **SGLang** if agent traffic is prefix-heavy or needs structured outputs. Reach for **TensorRT-LLM** only when latency SLOs justify NVIDIA-specific compilation. Use **Ollama/llama.cpp** for dev parity and edge.

**Agnostic note:** every engine above can expose an **OpenAI-compatible endpoint**. Standardize on that. The application must not know whether it is talking to vLLM on a neocloud, TensorRT-LLM on a hyperscaler GPU node, or Ollama on a laptop.

### 5.2 L4 — Model Gateway / Router (the keystone of agnosticism)

**Purpose:** one seam for all model access — multi-provider routing, automatic fallback, semantic/exact caching, budgets, rate limits, key management, and a **uniform OpenAI-compatible interface** over both open and closed models.

| Option | Role | License (verify) | Notes |
|--------|------|------------------|-------|
| **LiteLLM** | Proxy/SDK normalizing 100+ providers to OpenAI format; routing, fallback, budgets, virtual keys | MIT | The reference OSS gateway. Put *everything* behind it. |
| **Envoy AI Gateway / Higress / Kong AI Gateway** | API-gateway-grade traffic management for LLM traffic | Apache-2.0 | Use when you need enterprise gateway features (mTLS, WAF, quotas) at the edge. |
| **OpenLLM / vLLM router** | Lightweight self-host routing | Apache-2.0 | Simpler setups; fewer provider adapters than LiteLLM. |

**Why this is the keystone:** Tenet **T7** says the app makes *one* kind of model call. The gateway turns "switch from a closed model to self-hosted Llama on a neocloud" into a **config change**, not a code change. It is also where you enforce **budgets, fallback chains, and caching** uniformly.

**Agnostic note:** define routing in the spec (primary → fallback → budget caps), and keep provider credentials in a portable secret store (see cross-cutting). Never let the gateway's *config format* leak into application code.

### 5.3 L8 — Agentic Orchestration Frameworks

**Purpose:** turn a model into an agent — planning, tool use, multi-agent collaboration, human-in-the-loop, and stateful, resumable execution.

| Framework | Paradigm | Best for | License (verify) |
|-----------|----------|----------|------------------|
| **LangGraph** | Graph / state-machine; durable execution; HITL checkpoints | Stateful, controllable, production agent workflows where you need explicit control flow and resumption | MIT |
| **CrewAI** | Role-based "crews" of cooperating agents | Fast to build role/task team patterns; growing A2A support | MIT |
| **AutoGen** (and the **AG2** community fork) | Conversational multi-agent; event-driven | Conversational/chat-pattern agents, research-style multi-agent; .NET option | MIT |
| **Microsoft Agent Framework / Semantic Kernel** | Enterprise SDK, plugins, planners; .NET + Python | Enterprise apps already in the MS ecosystem but wanting OSS portability | MIT |
| **LlamaIndex (Workflows / AgentWorkflow)** | Data/RAG-centric agents | Retrieval-heavy agents and document workflows | MIT |
| **Haystack** | Pipeline/DAG composition | Composable RAG + agent pipelines, strong production lineage | Apache-2.0 |
| **DSPy** | Programmatic prompt/program optimization | Treating prompts as optimizable programs; auto-tuning | MIT/Apache (verify) |
| **OpenAI Agents SDK / Strands / Pydantic AI / Smolagents** | Lightweight code-first agents | Minimal, typed, or code-execution-centric agents | Apache-2.0/MIT (verify) |
| **OpenAgents** | Native MCP + A2A, persistent agent networks | Interop-first multi-agent deployments | Verify |

**Selection lens:**
- **Need explicit, resumable control flow + HITL?** → LangGraph (pair with Temporal for durability).
- **Need role-based teams fast?** → CrewAI.
- **Conversational/research multi-agent?** → AutoGen/AG2.
- **Already .NET / enterprise MS?** → Agent Framework / Semantic Kernel (still OSS, still portable).
- **Retrieval is the core?** → LlamaIndex or Haystack.
- **Want to *optimize* prompts as code?** → DSPy as a layer beneath any of the above.

**Agnostic note:** frameworks are the *least* locked-in layer (all OSS, all model-pluggable via the gateway) but the *most tempting* to over-couple. Keep business logic in your own modules; treat the framework as a library, not the architecture. Expose tools via **MCP** (Section 5.4) so they survive a framework swap. Avoid a framework's proprietary managed-cloud runtime — run the OSS core on your K8s.

### 5.4 L7 — Tool & Agent Interop Protocols

**Purpose:** standardize how agents call tools/data (MCP) and how agents talk to each other (A2A), so capabilities compose across frameworks and vendors.

| Standard | Role | Governance | Notes |
|----------|------|-----------|-------|
| **MCP (Model Context Protocol)** | Open protocol for exposing tools, resources, and prompts to any model/agent | Linux Foundation-hosted open standard | The "USB-C for tools." Write a tool once as an MCP server; any MCP-aware agent uses it. 2026 roadmap emphasizes transport scalability, async task semantics, and enterprise readiness (auth, audit, gateways). |
| **A2A (Agent-to-Agent)** | Open protocol for inter-agent discovery and messaging | Open standard (Linux Foundation lineage) | Lets agents built in different frameworks/vendors collaborate. Pair with MCP: A2A between agents, MCP between agent and tools. |
| **OpenAPI / JSON-Schema tools** | Classic function/tool definitions | Open | Fallback when MCP isn't available; wrap in MCP where possible. |

**Agnostic note:** **MCP + A2A are the single biggest agnostic lever in the agent layer.** A tool behind MCP is reusable across LangGraph, CrewAI, AutoGen, and vendor agents alike. Implement an **MCP gateway** (auth, allow-listing, audit) in front of tool servers for enterprise control without coupling to any one platform.

### 5.5 L5 — Retrieval / RAG

**Purpose:** ground generation in your data — ingest, chunk, embed, index, retrieve (hybrid: dense + sparse/BM25), rerank, and assemble context.

| Component | Open-source options | Notes |
|-----------|---------------------|-------|
| **Framework** | LlamaIndex, Haystack, LangChain | Pipeline + connectors + retrievers. Keep ingestion idempotent and re-embeddable. |
| **Embeddings (self-host)** | `bge` (BAAI), `e5`/`multilingual-e5`, `gte`, **Nomic Embed**, **Jina embeddings**, served via **HF Text Embeddings Inference (TEI)** | Pluggable behind an embedding interface; store `model@dim` in metadata so you can re-embed on switch. |
| **Rerankers** | `bge-reranker`, cross-encoder rerankers, ColBERT-style late interaction | Big quality lever; keep as a swappable stage. |
| **Hybrid search / sparse** | BM25 (OpenSearch / Elasticsearch OSS / Tantivy), SPLADE | Combine with dense for recall + precision. |
| **Parsing / chunking** | **Unstructured**, **Docling**, Apache Tika, **LlamaParse** alternatives | Normalize PDFs/HTML/Office to clean, chunkable text. |
| **Advanced patterns** | GraphRAG (graph-augmented), RAPTOR (hierarchical), contextual retrieval | Adopt only when eval shows base RAG plateauing. |

**Agnostic note:** the lock-in risk in RAG is the **embedding model + index format**. Mitigate with (1) an abstract embedding interface, (2) embeddings stored with their model/version, and (3) a **re-embed pipeline** treated as a first-class, runnable job. Then any vector engine and any embedding model become swappable.

### 5.6 L2 — Vector Database / Index

**Purpose:** store and ANN-search embeddings with metadata filtering, hybrid search, and scale.

| Engine | Profile | License (verify) | Notes |
|--------|---------|------------------|-------|
| **Milvus** | Distributed, billion-scale, GPU-accelerated indexes | Apache-2.0 | Heavy-duty scale; richest index options. |
| **Qdrant** | Rust, fast, great filtering & quantization, simple ops | Apache-2.0 | Excellent default for most workloads; strong DX. |
| **Weaviate** | Modules, hybrid search, multi-tenancy | BSD-3 | Good hybrid + ecosystem integrations. |
| **pgvector** (+ `pgvectorscale`) | Vectors *inside* Postgres | PostgreSQL license | Best when you already run Postgres and want one system of record; transactional consistency. |
| **Chroma** | Lightweight, dev-friendly, embeddable | Apache-2.0 | Great for prototypes; check scale ceiling for prod. |
| **OpenSearch / Elasticsearch (k-NN)** | Vectors alongside lexical search | Apache-2.0 / verify | Strong when you also need full-text and already operate the cluster. |

**Selection lens:** **pgvector** if Postgres is already core and scale is moderate; **Qdrant** as a clean dedicated default; **Milvus** at very large scale; **Weaviate/OpenSearch** when hybrid + ecosystem matter.

**Agnostic note:** avoid managed vector indexes with **no bulk export**. Whatever you pick, ensure (1) a standard upsert/query surface in your code, (2) periodic export of vectors + payloads to object storage, and (3) the re-embed pipeline from 5.5 — together these make the vector store a commodity.

### 5.7 L6 — Guardrails, Safety & Policy

**Purpose:** validate inputs/outputs, block jailbreaks and PII leakage, enforce schemas and business policy, and classify unsafe content.

| Component | Open-source options | Notes |
|-----------|---------------------|-------|
| **I/O validation / structure** | **Guardrails AI**, **NeMo Guardrails**, **Pydantic** + constrained decoding (Outlines, `lm-format-enforcer`, XGrammar) | Enforce JSON/schema and conversational rails deterministically. |
| **Safety classifiers (open weights)** | **Llama Guard**, **Prompt Guard**, **ShieldGemma**, **Granite Guardian** | Open-weight moderators you can self-host — no closed safety endpoint dependency. (Open weights ≠ OSI license; verify model terms.) |
| **PII / secrets** | **Microsoft Presidio**, spaCy NER pipelines | Detect/redact PII pre-prompt and pre-log. |
| **Policy engine** | **Open Policy Agent (OPA) / Rego**, Cedar | Externalize authorization and action policy from app code — same engine across clouds. |
| **Prompt-injection / red-team** | **Garak**, **PyRIT**, **promptfoo** (also eval) | Continuous adversarial testing in CI. |

**Agnostic note:** make guardrails a **sidecar/middleware** around the gateway (L4) and tool calls (L7), not inline in business logic. Policy in **OPA/Rego** travels unchanged across any cloud and any framework.

### 5.8 Cross-Cutting — Observability, Tracing & Evaluation

**Purpose:** trace agent runs, track tokens/cost/latency, version prompts, and — critically — **evaluate quality** as a gate, not an afterthought.

| Capability | Open-source options | License (verify) | Notes |
|-----------|---------------------|------------------|-------|
| **LLM tracing / observability** | **Langfuse**, **Arize Phoenix**, **OpenLLMetry** (Traceloop), **Helicone**, **SigNoz** | Langfuse: MIT core. Phoenix: Elastic-2.0 (verify). | Capture prompts, tool calls, token usage, latency, cost per run. |
| **Open telemetry substrate** | **OpenTelemetry** + **OTel GenAI semantic conventions**, Grafana/Tempo/Loki/Prometheus, SigNoz | Apache-2.0 | The portable backbone — emit OTLP; swap front-ends freely. |
| **Offline / batch eval** | **RAGAS** (RAG metrics), **DeepEval**, **promptfoo**, **OpenAI Evals**-style harnesses, **Phoenix evals** | Apache-2.0 / verify | Faithfulness, answer-relevance, context-precision/recall, task success, regression suites. |
| **Agent-trajectory eval** | LangSmith-style trajectory checks (OSS alternatives), tool-call accuracy, step-success | Verify | For multi-step agents, score the *path*, not just the final answer. |
| **Human feedback / annotation** | Argilla, Label Studio | Apache-2.0 | Capture human labels to build golden sets. |

**Agnostic note:** standardize on **OpenTelemetry + GenAI semantic conventions**. Tracing/eval tools that *consume OTLP* are interchangeable. Keep your **golden eval sets and metric definitions in the repo** (versioned), so evaluation is reproducible on any backend.

### 5.9 L8b — Durable Execution & Workflow

**Purpose:** make long-running, multi-step, failure-prone agent runs **reliable and resumable** — retries, timeouts, idempotency, human approvals that may take days.

| Option | Role | License (verify) | Notes |
|--------|------|------------------|-------|
| **Temporal** | Durable workflow engine; deterministic replay | MIT | The portable backbone for production agents; survives restarts and resumes mid-flight. |
| **LangGraph durability / checkpointers** | Built-in checkpointing to Postgres/Redis | MIT | Native to LangGraph; good for graph-shaped agents. |
| **Apache Airflow / Dagster / Prefect** | Batch data + ingestion/eval pipelines | Apache-2.0 | For RAG ingestion, re-embedding, eval batches. |
| **Argo Workflows / Events** | K8s-native pipelines & eventing | Apache-2.0 | Cloud-neutral CI/CD and batch on Kubernetes. |

**Agnostic note:** durable execution is what separates a demo from a system. Run **Temporal** (self-hosted) or LangGraph checkpointers on portable Postgres so the *control plane of your agents* is cloud-neutral.

### 5.10 L2b — Data, State, Messaging

| Need | Portable open-source choice | Contract |
|------|----------------------------|----------|
| **Object storage** | **MinIO** (or any S3-compatible) | S3 API |
| **Relational / system-of-record** | **PostgreSQL** (+ pgvector) | SQL / wire protocol |
| **Cache / session / short-term memory** | **Redis** / **Valkey** | RESP protocol |
| **Streaming / events** | **Apache Kafka** / **Redpanda** / **NATS** / **RabbitMQ** | Kafka or AMQP; **CloudEvents** payloads |
| **Agent long-term memory** | Postgres + vector store; OSS memory libs (**Mem0**, Letta/MemGPT-style) | Library over portable stores |
| **Feature/data lake** | Iceberg/Delta/Parquet on object storage | Open table formats |

**Agnostic note:** every store above speaks an **open protocol** with a self-hostable reference implementation. This is the layer where "no bulk export" kills you — never accept it.

### 5.11 L1 — Compute Platform

| Concern | Portable open-source choice | Notes |
|---------|----------------------------|-------|
| **Container orchestration** | **Kubernetes** (CNCF-conformant) | The portability substrate. Same manifests on EKS, AKS, GKE, or a neocloud's managed/raw K8s. |
| **GPU enablement** | **NVIDIA GPU Operator**, Device Plugin, MIG, DRA | Standard GPU scheduling across clusters. |
| **Distributed compute** | **Ray** (Serve/Train/Data) | Portable distributed runtime for serving, batch inference, fine-tuning. |
| **Batch/gang scheduling** | **Volcano**, **Kueue** | GPU job queuing and gang scheduling for training/batch. |
| **Serving control plane** | **KServe**, **Ray Serve**, **Seldon Core** | Autoscale-to-zero, canary, multi-model — on top of L3 engines. |
| **Packaging** | **Helm**, **Kustomize**, **KEDA** (event autoscaling) | Declarative, cloud-neutral deploys. |

**Agnostic note:** Kubernetes is the contract that makes hyperscalers and neoclouds interchangeable at L1. Keep workloads CNCF-conformant; avoid a provider's proprietary scheduler extensions in the hot path.

### 5.12 L0 — Infrastructure as Code & Multi-Cloud Control

| Concern | Portable open-source choice | Notes |
|---------|----------------------------|-------|
| **IaC** | **OpenTofu** (MPL fork of Terraform) or Terraform; **Pulumi** | Use OpenTofu if license-neutrality matters. Modularize: provider modules vs. app modules. |
| **Control plane / abstraction** | **Crossplane** | Define cloud-neutral resource abstractions; back them with provider-specific implementations. |
| **Cross-cloud GPU scheduling** | **SkyPilot** | Run/abstract training & inference across clouds *and neoclouds*, chasing price/availability. **A direct agnostic lever for neocloud arbitrage.** |
| **GitOps** | **Argo CD**, **Flux** | Declarative, auditable deploys to any cluster. |
| **Policy-as-code** | **OPA/Gatekeeper**, **Kyverno**, Checkov | Guardrails on infra across providers. |

**Agnostic note:** **SkyPilot + Crossplane + OpenTofu** is the trifecta that turns "which cloud?" into a scheduling decision. SkyPilot in particular is purpose-built to place GPU jobs on whichever provider (hyperscaler or neocloud) has capacity at the right price.

### 5.13 Cross-Cutting — Identity, Secrets, FinOps, Registry

| Concern | Portable open-source choice | Notes |
|---------|----------------------------|-------|
| **Workload identity** | **SPIFFE/SPIRE**, OIDC, **Keycloak**, Dex | Cloud-neutral identity for services and agents. |
| **Secrets** | **HashiCorp Vault** / **OpenBao**, External Secrets Operator | OpenBao is the OSS-license fork; keep provider keys here, not in code. |
| **FinOps / cost** | **OpenCost**, Kubecost (OSS tier), gateway-level token/budget metering (LiteLLM) | Per-model, per-tenant, per-agent cost attribution. |
| **Prompt & model registry** | Git-versioned prompts; **MLflow**; OCI artifacts for models | Treat prompts and model versions as versioned artifacts in CI. |
| **Model/data governance** | Model cards, data lineage (OpenLineage), eval reports in repo | Governance that travels with the spec. |

---

## 6. AI Neoclouds as a Compute Target

**AI neoclouds** are GPU-specialist clouds offering raw and managed accelerator capacity — often cheaper, sooner-available, or higher-end than hyperscaler GPU SKUs. In a vendor-agnostic design they are **peers of AWS/Azure/GCP at L0–L3**, reachable through the same Kubernetes/serving/gateway contracts.

### 6.1 The lanes (how the market splits)

| Lane | Representative providers | Use it for | Agnostic fit |
|------|--------------------------|-----------|--------------|
| **Hyperscaler-managed** | AWS, Azure, GCP | Enterprise procurement, breadth of managed services, compliance baseline | Baseline; keep core logic on portable K8s, not proprietary runtimes |
| **Tier-one neocloud (scale + price)** | **CoreWeave, Crusoe, Lambda, Nebius** | Reserved large-scale GPU fleets, training + heavy inference, better $/GPU-hr | Run your own K8s + vLLM/SGLang; pure compute substitution |
| **Full-stack compute + inference** | **Together AI**, Fireworks, Baseten | One-vendor model+serving pipelines, fast open-model endpoints | Consume via OpenAI-compatible API behind the gateway — trivially swappable |
| **Spot / marketplace** | **RunPod, Vast.ai** | Research, bursty/batch, cheapest spot GPUs | Great for eval/fine-tune batches via SkyPilot; treat as ephemeral |
| **Specialty / sovereign / power-advantaged** | **Voltage Park, Fluidstack**, regional sovereign GPU clouds | Data residency, dedicated capacity, specific regions | Same K8s contract; validate networking & storage parity |

> The **binding constraint** in this market is often **power and reserved capacity**, not GPUs per se — procurement looks more like contracting for a power plant than buying SaaS. Reserve where you have steady baseline load; use spot/marketplace for bursty batch.

### 6.2 Patterns for staying agnostic across neoclouds + hyperscalers

1. **Two consumption modes, one contract.**
   - *Bring-your-own-serving:* rent raw GPUs (CoreWeave/Lambda/Crusoe/RunPod), run **your** vLLM/SGLang on **your** K8s. You own the OpenAI-compatible endpoint.
   - *Managed inference:* call a neocloud's open-model endpoint (Together/Fireworks/Baseten) **through LiteLLM**. Identical call site to option 1.
   In both cases the application sees one OpenAI-compatible API. **This is the whole game.**

2. **SkyPilot for placement & arbitrage.** Declare the job (model, GPU type, region constraints); let SkyPilot place it on whichever provider has capacity at the best price, hyperscaler or neocloud. Re-run elsewhere by changing a config, not code.

3. **Storage parity check.** Confirm each provider offers **S3-compatible** object storage (or run **MinIO** on their compute). Keep datasets/checkpoints in object storage, not local disk, so jobs are relocatable.

4. **Networking & egress modeling.** Neoclouds vary widely on egress pricing, private networking, and inter-region links. Model **data-transfer cost** explicitly in the FinOps spec — egress is the silent tax of multi-cloud.

5. **Capacity & contract terms in the spec.** Record reserved-vs-on-demand, commitment length, GPU generation (H100/H200/B200/MI300), and interruption policy in the **Infrastructure Spec** so the trade-offs are explicit and reviewable.

6. **Compliance & residency.** Validate certifications (SOC 2, ISO 27001, region/sovereign requirements) per provider; some neoclouds are newer to enterprise compliance — make this a **gate**, not an assumption.

### 6.3 Neocloud evaluation checklist (put in the Infrastructure Spec)

- [ ] GPU generations & quantities available; reservation vs. spot; interruption SLA
- [ ] Kubernetes: managed offering or raw VMs (and who runs the control plane)
- [ ] S3-compatible object storage present (or MinIO-able)
- [ ] Egress / inter-region / cross-cloud transfer pricing modeled
- [ ] Private networking / VPC peering / PrivateLink-equivalent
- [ ] Compliance certifications vs. your data classification
- [ ] Region & data-residency coverage
- [ ] Open-model endpoint OpenAI-compatibility (if using managed inference)
- [ ] Support model & capacity guarantees in contract
- [ ] Exit terms: data export, no proprietary format lock-in

---

## 7. The Spec-Driven Methodology (Extended for GenAI)

The canonical SDD loop is **Specify → Plan → Tasks → Implement**, each phase producing a markdown artifact the next phase consumes (per GitHub Spec Kit and equivalents like Kiro/BMAD). For GenAI/agentic systems we extend it to a **six-phase loop**, because (a) correctness is *probabilistic* and must be **measured**, and (b) behavior, cost, and safety **drift** and must be **operated**.

```
        ┌──────────────────────────────────────────────────────────┐
        │                    PHASE 0: CONSTITUTION                  │
        │   Non-negotiable principles (the 10 Portability Tenets,   │
        │   safety, compliance, eval bar). Authored once, amended   │
        │   deliberately. Every later phase must comply.            │
        └──────────────────────────────────────────────────────────┘
                                   │
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │ 1 SPECIFY │──▶│  2 PLAN   │──▶│  3 TASKS  │──▶│4 IMPLEMENT│──▶│5 EVALUATE │
   │  WHAT/WHY │   │ HOW (arch │   │ small,    │   │ generate  │   │ gate on   │
   │  no stack │   │ +contracts│   │ reviewable│   │ to spec   │   │ eval sets │
   └───────────┘   │ vendor-   │   │ units)    │   └───────────┘   └─────┬─────┘
        ▲          │ neutral)  │   └───────────┘                         │
        │          └───────────┘                                          │ pass
        │                                                                 ▼
        │                                                          ┌───────────┐
        └──────────────────── spec is updated, not patched ◀───────│ 6 OPERATE │
                          (drift / cost / safety feed back)         │ observe + │
                                                                    │ govern    │
                                                                    └───────────┘
```

**Core rule:** *the spec — not the chat history, not the code — is the source of truth.* Code is a **generated artifact** regenerable from the spec. A change in behavior starts as a change in the spec.

### 7.1 Phase 0 — Constitution (author once, amend rarely)

A `constitution.md` of non-negotiables that **govern every other artifact**. For this platform it includes the **10 Portability Tenets** (Section 2), plus:

- **Safety bar:** no unguarded model output to users; PII redaction before logging; safety classifier on ingress/egress.
- **Eval bar:** quantitative gates that *must* pass before merge/deploy (defined in the Eval Spec).
- **Compliance bar:** data classification, residency, retention, audit.
- **Cost bar:** per-request / per-tenant budget ceilings enforced at the gateway.

Cross-artifact analysis checks later specs against the constitution. A violation is either fixed or recorded as a **time-boxed, signed-off exception**.

### 7.2 Phase 1 — Specify (WHAT and WHY — no tech stack)

Describe the capability in **user and outcome terms**: who the user is, the problem, the interaction, success metrics, acceptance criteria, and explicitly **out-of-scope** items. For agents, specify **observable behavior and boundaries**, not implementation.

GenAI-specific additions to a normal feature spec:
- **Capabilities & tasks** the agent must perform; **autonomy level** (suggest / act-with-approval / act-autonomously).
- **Tools/data** the agent may access (and must *not*).
- **Quality bar in user terms** ("answers cite a source," "no fabricated figures," "asks before irreversible actions").
- **Failure behavior** ("on low confidence, escalate to human").
- Mark unknowns explicitly as `[NEEDS CLARIFICATION: …]` rather than guessing.

**Artifact:** `spec.md` (Capability Spec — template in 8.1).

### 7.3 Phase 2 — Plan (HOW — architecture & contracts, vendor-neutral)

Add the technical *how*, expressed against **open contracts**, not vendor SKUs. This is where agnosticism is enforced:

- **Architecture** mapped to the L0–L9 layers (Section 4); chosen open-source component per layer with **rationale** and **fallback**.
- **Contracts:** the OpenAI-compatible model API, MCP tool schemas, A2A messages, data schemas, eventing payloads (CloudEvents).
- **Model strategy:** primary/fallback chain (open-weight default per Tenet T2), routing & budget rules at the gateway.
- **Infrastructure Spec:** target cloud(s) — hyperscaler and/or neocloud — with the Section 6.3 checklist; Kubernetes as the runtime; SkyPilot/Crossplane/OpenTofu modules.
- **Portability check:** for each component, "can this be replaced without touching app code?" If no → wrap it.
- Optionally generate **alternative plans** (e.g., self-hosted vLLM on neocloud vs. managed endpoint) and compare on cost/latency/lock-in.

**Artifacts:** `plan.md` (Architecture Spec — 8.2), `infrastructure.md` (8.3), `agent.md` per agent (8.4).

### 7.4 Phase 3 — Tasks (decompose into small, reviewable units)

The agent/team breaks the plan into **small, independently testable tasks**, ordered by dependency. Each task names the spec section it satisfies and its **acceptance check** (including which eval it must not regress). Example granularity: *"Implement the MCP server for the `lookup_order` tool with input schema X and audit logging"* — not *"build the tools."*

**Artifact:** `tasks.md` (8.5).

### 7.5 Phase 4 — Implement (generate to spec)

Execute tasks one at a time against the spec. Conventions that keep agnosticism intact:

- All model calls go through the **gateway** (no vendor SDKs in app code).
- All tools are **MCP servers**; all inter-agent calls are **A2A**.
- Everything is an **OCI image** deployed by **Helm/GitOps** to Kubernetes.
- IaC is **modular** (provider modules vs. app modules).
- Telemetry emits **OpenTelemetry/OTLP** from day one.

If implementation reveals the spec was wrong, **update the spec, then regenerate** — do not let code silently diverge.

### 7.6 Phase 5 — Evaluate (the GenAI-specific gate)

Probabilistic systems are not "done" because the code compiles — they're done when they **pass the eval bar**. This phase is a **hard gate** in CI/CD.

- **Golden datasets** versioned in the repo (inputs + expected behavior/answers + references).
- **RAG metrics:** faithfulness, answer relevance, context precision/recall (e.g., RAGAS/DeepEval).
- **Agent-trajectory metrics:** task success, tool-call accuracy, step efficiency, unsafe-action rate.
- **Regression gates:** new build must not drop key metrics below thresholds set in the Eval Spec.
- **Adversarial/red-team:** prompt-injection and jailbreak suites (Garak/PyRIT/promptfoo) run in CI.
- **Cost/latency budgets:** p95 latency and per-request token/cost ceilings checked.

**Artifact:** `eval.md` (8.6) + results stored as versioned reports. **Fail the gate → fix spec/impl, do not ship.**

### 7.7 Phase 6 — Operate (continuous specs: drift, cost, safety)

Production GenAI behavior changes even when code doesn't (model updates, data drift, prompt-injection in the wild). Operate treats these as **living specifications**:

- **Online eval & tracing:** sample production traces into Langfuse/Phoenix; run periodic eval on live traffic.
- **Drift detection:** alert when quality/cost/latency metrics move beyond Eval-Spec thresholds.
- **Safety monitoring:** log guardrail triggers; review jailbreak attempts.
- **FinOps:** per-model/tenant/agent cost attribution (OpenCost + gateway metering); enforce budgets.
- **Feedback loop:** human labels (Argilla/Label Studio) flow back into golden sets → Phase 1/5 of the *next* iteration.

The loop closes: operational findings **update the spec**, and the next build is regenerated and re-evaluated.

### 7.8 Tooling for running SDD

The methodology is tool-neutral. To operationalize it with AI coding agents, **GitHub Spec Kit** (the open-source `specify` CLI, MIT-licensed, agent-neutral) scaffolds `spec.md / plan.md / tasks.md` plus a `constitution.md`, and wires slash commands (`/specify`, `/plan`, `/tasks`, `/implement`). Alternatives/variants include **Kiro**, **BMAD**, and **Tessl**. Spec Kit is **agent-neutral by design** (Copilot, Claude, Gemini, Cursor, etc.) — itself an instance of the agnostic principle. Add the **Evaluate** and **Operate** artifacts (8.6, and operate runbooks) as repo conventions on top.

---

## 8. Specification Templates

Copy these into your repo under `specs/`. Each is a living markdown artifact.

### 8.1 Capability Spec (`spec.md`)

```markdown
# Capability: <name>
## Why
- Problem / user / business outcome:
- Success metrics (measurable):
## Users & Scenarios
- Primary user(s):
- Key scenarios (happy path + edge):
## Behavior (WHAT, not HOW)
- Inputs / outputs (in user terms):
- Autonomy level: [ suggest | act-with-approval | act-autonomous ]
- Quality bar: [ e.g. cites sources; no fabricated numbers; asks before irreversible acts ]
- Failure behavior: [ e.g. low confidence -> escalate to human ]
## Boundaries
- In scope:
- OUT of scope:
- Data the agent may access / must NOT access:
## Acceptance Criteria
- [ ] criterion (maps to an eval in eval.md)
## Open Questions
- [NEEDS CLARIFICATION: ...]
```

### 8.2 Architecture Spec (`plan.md`)

```markdown
# Architecture for: <capability>
## Layer Mapping (L0–L9) — chosen OSS component + rationale + fallback
| Layer | Component | Why | Fallback | Replaceable w/o app change? |
|-------|-----------|-----|----------|-----------------------------|
| L3 Serving   | vLLM       | ... | SGLang   | yes (OpenAI API) |
| L4 Gateway   | LiteLLM    | ... | Envoy AI | yes |
| L8 Orchestr. | LangGraph  | ... | CrewAI   | logic in own modules |
| L2 Vector    | Qdrant     | ... | pgvector | yes (re-embed pipeline) |
| ...          | ...        | ... | ...      | ... |
## Contracts
- Model API: OpenAI-compatible /v1/chat/completions (+ tools)
- Tools: MCP server schemas (link)
- Inter-agent: A2A messages (link)
- Data schemas / events (CloudEvents):
## Model Strategy
- Primary (open-weight): ...   Fallback chain: ...
- Routing & budget rules (gateway):
## Portability Check
- Any component NOT replaceable w/o code change? -> wrap it. List exceptions:
## Constitution Compliance
- Tenets satisfied / exceptions (time-boxed, signed-off):
```

### 8.3 Infrastructure Spec (`infrastructure.md`)

```markdown
# Infrastructure & Compute
## Targets (hyperscaler and/or neocloud)
| Provider | Lane | GPU gen | Reserved/Spot | Region | Compliance | Egress model |
## Runtime
- Kubernetes: managed / raw; GPU operator; serving control plane (KServe/Ray Serve)
- Placement/arbitrage: SkyPilot config; Crossplane abstractions; OpenTofu modules
## Storage & State
- Object (S3-compatible / MinIO); Postgres(+pgvector); Redis/Valkey; Kafka/NATS
## Neocloud Checklist (Section 6.3)
- [ ] ... (paste checklist, mark each)
## Exit
- Data export path; estimated time-to-restand on alt provider (target <= 2 weeks)
```

### 8.4 Agent Spec (`agent.md`, one per agent)

```markdown
# Agent: <name>
## Role & Goal
## Tools (MCP servers) — allow-list
| Tool | Input schema | Side effects | Approval required? | Audit |
## Memory
- Short-term (session): Redis/Postgres
- Long-term: vector store / memory lib; retention policy
## Planning & Control
- Pattern: [ react | plan-execute | graph/state-machine | multi-agent crew ]
- HITL checkpoints / interrupts:
- Durability: Temporal / LangGraph checkpointer
## Guardrails
- Input: injection/PII; Output: schema + safety classifier
- Policy (OPA/Rego): which actions gated
## A2A
- Which agents it talks to; message contracts
## Eval hooks
- Trajectory metrics; unsafe-action rate; task success (-> eval.md)
```

### 8.5 Tasks (`tasks.md`)

```markdown
# Tasks for: <capability>
- [ ] T01 <small, testable unit>  (satisfies spec §X)  | accept: <check / eval not regressed>
- [ ] T02 ...                      (depends on T01)
```

### 8.6 Eval Spec (`eval.md`)

```markdown
# Evaluation & Gates
## Golden Datasets (versioned in repo)
- path, size, how curated, refresh policy
## Metrics & Thresholds (CI gate)
| Metric | Tool | Threshold | Blocking? |
| faithfulness | RAGAS | >= 0.85 | yes |
| context recall | RAGAS | >= 0.80 | yes |
| task success | trajectory eval | >= 0.90 | yes |
| unsafe-action rate | red-team suite | = 0 | yes |
| p95 latency | load test | <= Xms | yes |
| cost/request | gateway meter | <= $Y | yes |
## Adversarial
- Suites: Garak / PyRIT / promptfoo (injection, jailbreak) — must pass
## Online (Operate)
- Sampling rate of prod traces; drift thresholds & alerts
```

---

## 9. Portability Scorecard & Decision Framework

Score every candidate component **0–2** per criterion (0 = lock-in risk, 1 = partial, 2 = fully portable). Adopt only components that clear your bar (e.g., total ≥ 12/16 and no zeros on C1/C2/C8).

| # | Criterion | 0 (risk) | 1 (partial) | 2 (portable) |
|---|-----------|----------|-------------|--------------|
| **C1** | Open license | Proprietary | Source-available/restrictive | OSI-approved |
| **C2** | Open/standard interface | Vendor-only API | Adapter exists | Speaks an open standard (OpenAI/S3/K8s/OTLP/MCP) |
| **C3** | Self-hostable | SaaS-only | Self-host w/ caveats | Fully self-hostable |
| **C4** | Data export | None/proprietary | Manual export | Bulk export to open format |
| **C5** | Multi-cloud runtime | One cloud | Some | Runs on any K8s/cloud incl. neoclouds |
| **C6** | Replaceable w/o app change | Code rewrite | Some refactor | Config-only swap |
| **C7** | Community & maintenance | Single-vendor, thin | Moderate | Healthy multi-stakeholder OSS |
| **C8** | Standards alignment | Bespoke | Partial | MCP/A2A/OTel/OpenAI-compatible |

**Decision heuristics:**
- Prefer the component that maximizes **C2 + C6** even at some feature cost — interface portability compounds.
- A "managed" option is acceptable **only** if it's consumed through an open contract behind your gateway/adapter (so it's swappable). Managed inference via OpenAI-compatible API ✅; a proprietary agent runtime holding your orchestration logic ❌.
- When two options tie, choose the one with the **healthier OSS governance** (C7) — it ages better.

---

## 10. Reference Stack Blueprints (Three Tiers)

All three are **fully vendor-agnostic** and run identically on a hyperscaler or a neocloud. Pick by scale/maturity.

### 10.1 Tier 1 — Lean / Startup / PoC

- **Serving:** Ollama or single-node vLLM (open-weight model)
- **Gateway:** LiteLLM
- **Orchestration:** LangGraph *or* CrewAI
- **RAG:** LlamaIndex + pgvector (one Postgres = system of record + vectors)
- **Embeddings:** `bge`/`e5` via TEI
- **Guardrails:** Pydantic + Llama Guard
- **Observability/Eval:** Langfuse + promptfoo
- **Runtime:** single K8s cluster (or Docker Compose for dev); MinIO + Postgres + Redis
- **IaC:** OpenTofu, one provider module
- **Why:** minimum moving parts, every piece swappable later; pgvector avoids a second datastore.

### 10.2 Tier 2 — Production / Scale-Up

- **Serving:** vLLM (+ SGLang for agent traffic) on KServe/Ray Serve, autoscaled
- **Gateway:** LiteLLM (+ Envoy AI Gateway at the edge)
- **Orchestration:** LangGraph + **Temporal** (durable, resumable, HITL)
- **Tools/Interop:** MCP servers behind an MCP gateway; A2A between agents
- **RAG:** LlamaIndex/Haystack + Qdrant (or Milvus); reranker; hybrid search
- **Guardrails:** NeMo/Guardrails AI + Presidio + OPA policy
- **Observability/Eval:** Langfuse + OpenTelemetry/Grafana + RAGAS/DeepEval in CI; Garak/PyRIT red-team
- **Runtime:** multi-node K8s, GPU operator, Volcano/Kueue; MinIO/S3, Postgres, Redis, Kafka/NATS
- **Multi-cloud:** SkyPilot for GPU placement (hyperscaler + neocloud); Crossplane abstractions; Argo CD GitOps
- **Why:** durable agents, eval-gated CI, and true neocloud arbitrage.

### 10.3 Tier 3 — Enterprise / Multi-Cloud / Sovereign

- Everything in Tier 2, plus:
- **Federated serving** across hyperscaler + multiple neoclouds (CoreWeave/Crusoe/Lambda/Nebius) via one OpenAI-compatible facade
- **Identity:** SPIFFE/SPIRE workload identity; Keycloak/OIDC; Vault/OpenBao secrets
- **Governance:** model cards, OpenLineage data lineage, signed eval reports, policy-as-code (Kyverno/Gatekeeper)
- **FinOps:** OpenCost + per-tenant gateway metering + budget enforcement
- **DR/residency:** region/sovereign-aware placement; rehearsed exit runbook (≤ 2 weeks restand)
- **Why:** regulated, multi-region, multi-vendor with leverage and sovereignty.

---

## 11. Anti-Patterns & Lock-In Traps

| Anti-pattern | Why it hurts | Agnostic fix |
|--------------|--------------|--------------|
| Importing a vendor model SDK directly in app code | Every call site couples to one provider | Route all calls through the gateway (LiteLLM) |
| Orchestration logic living in a proprietary managed-agent runtime | Your *core behavior* is now the vendor's | Run OSS framework core on your K8s; keep logic in your modules |
| Managed vector index with no bulk export | Data hostage; can't migrate or re-embed | Self-hostable engine + export-to-object-storage + re-embed pipeline |
| Embeddings without stored model/version | Can't switch embedding models without silent corruption | Store `model@dim`; treat re-embed as a runnable job |
| Telemetry only in a proprietary console | Can't move observability; no portable history | Emit OpenTelemetry/OTLP; front-ends become swappable |
| Tools as framework-specific functions | Tools die when you change frameworks | Expose tools as MCP servers |
| Cloud-specific queue/eventing semantics in business logic | Eventing rewrite on migration | Kafka/AMQP + CloudEvents payloads |
| "We'll add evals later" | Probabilistic regressions ship silently | Eval gate in CI from day one (Phase 5) |
| Proprietary IaC state coupling | Migration friction at the infra layer | Modular OpenTofu + remote state in S3-compatible store |
| Single-cloud GPU assumption | No price/availability leverage | SkyPilot placement across hyperscaler + neoclouds |
| Prompts only in code/chat history | Behavior unversioned, drifts | Prompts as versioned artifacts; spec is source of truth |

---

## 12. Migration & Exit Playbook

Exit cost is a **tested metric** (Tenet T10). Rehearse this so "leave a provider" is routine.

1. **Pre-condition (always-on):** datasets, checkpoints, vector exports in S3-compatible object storage; IaC modular; telemetry in OTLP; everything containerized.
2. **Stand up L1 elsewhere:** provision K8s + GPU operator on the target (hyperscaler or neocloud) via OpenTofu/Crossplane module swap.
3. **Restore L2 state:** MinIO/Postgres/Redis/Kafka from backups/exports; load vectors (or re-embed from source via the pipeline).
4. **Redeploy L3–L9:** same Helm charts/GitOps; point the gateway at the new serving endpoints (or managed neocloud endpoints).
5. **Re-point the gateway:** change model routing config — *no application code changes*.
6. **Re-run the eval gate (Phase 5):** confirm quality/cost/latency parity on the new substrate before cutover.
7. **Cutover & observe (Phase 6):** shift traffic; watch drift/cost dashboards; keep old provider warm until parity confirmed.

**Target:** core path re-stood on an alternate provider in **≤ 2 weeks**, validated by passing evals — not by "it deploys."

---

## Appendix A: Master Component Catalog

> Licenses are indicative — **verify at adoption**. "Open weights" denotes downloadable model weights, which may carry model-specific (non-OSI) terms.

| Layer | Component | Role | License (verify) |
|-------|-----------|------|------------------|
| L3 Serving | vLLM | High-throughput inference server | Apache-2.0 |
| L3 Serving | SGLang | Structured/agent inference | Apache-2.0 |
| L3 Serving | TensorRT-LLM | Peak NVIDIA performance | Apache-2.0 |
| L3 Serving | Hugging Face TGI | Mature serving | Verify |
| L3 Serving | Ollama | Local/edge serving | MIT |
| L3 Serving | llama.cpp | CPU/quantized/edge | MIT |
| L3 Control | KServe / Ray Serve / Seldon | Serving control plane | Apache-2.0 |
| L4 Gateway | LiteLLM | Multi-provider OpenAI-compatible gateway | MIT |
| L4 Gateway | Envoy AI Gateway / Higress / Kong AI Gateway | Edge LLM traffic mgmt | Apache-2.0 |
| L8 Agents | LangGraph | Graph/state-machine agents, durable | MIT |
| L8 Agents | CrewAI | Role-based multi-agent | MIT |
| L8 Agents | AutoGen / AG2 | Conversational multi-agent | MIT |
| L8 Agents | Microsoft Agent Framework / Semantic Kernel | Enterprise SDK | MIT |
| L8 Agents | LlamaIndex Workflows | RAG-centric agents | MIT |
| L8 Agents | Haystack | Pipeline/DAG agents + RAG | Apache-2.0 |
| L8 Agents | DSPy | Prompt-as-program optimization | Verify |
| L8 Agents | Pydantic AI / Smolagents / Strands | Lightweight code-first agents | MIT/Apache (verify) |
| L7 Protocol | MCP | Tool/resource protocol | Open standard (Linux Foundation) |
| L7 Protocol | A2A | Agent-to-agent protocol | Open standard |
| L8b Durable | Temporal | Durable workflow engine | MIT |
| L8b Batch | Airflow / Dagster / Prefect | Data & eval pipelines | Apache-2.0 |
| L8b K8s | Argo Workflows/Events | K8s-native pipelines | Apache-2.0 |
| L5 RAG | LlamaIndex / Haystack / LangChain | RAG frameworks | MIT/Apache |
| L5 Embeddings | bge / e5 / gte / Nomic / Jina (via TEI) | Self-host embeddings | Open weights / Apache (verify) |
| L5 Parsing | Unstructured / Docling / Tika | Document parsing | Apache-2.0 (verify) |
| L2 Vector | Milvus | Distributed vector DB | Apache-2.0 |
| L2 Vector | Qdrant | Vector DB (Rust) | Apache-2.0 |
| L2 Vector | Weaviate | Vector DB + modules | BSD-3 |
| L2 Vector | pgvector / pgvectorscale | Vectors in Postgres | PostgreSQL license |
| L2 Vector | Chroma | Lightweight vector DB | Apache-2.0 |
| L2 Search | OpenSearch / Elasticsearch (k-NN) | Hybrid vector+lexical | Apache-2.0 / verify |
| L6 Guardrails | Guardrails AI / NeMo Guardrails | I/O validation & rails | Apache-2.0 |
| L6 Safety | Llama Guard / Prompt Guard / ShieldGemma / Granite Guardian | Safety classifiers | Open weights (verify terms) |
| L6 PII | Microsoft Presidio | PII detection/redaction | MIT |
| L6 Policy | Open Policy Agent (OPA) / Cedar | Policy-as-code | Apache-2.0 |
| L6 Red-team | Garak / PyRIT / promptfoo | Adversarial testing | Apache-2.0/MIT (verify) |
| Obs | Langfuse | LLM tracing/observability | MIT (core) |
| Obs | Arize Phoenix | Tracing + evals | Elastic-2.0 (verify) |
| Obs | OpenLLMetry (Traceloop) / SigNoz / Helicone | OTel-based observability | Apache-2.0 (verify) |
| Obs | OpenTelemetry (+ GenAI conventions) | Telemetry substrate | Apache-2.0 |
| Eval | RAGAS / DeepEval / promptfoo | Offline eval | Apache-2.0 (verify) |
| Eval | Argilla / Label Studio | Human feedback/annotation | Apache-2.0 |
| L2 Object | MinIO | S3-compatible storage | Verify (AGPL lineage) |
| L2 SQL | PostgreSQL | System of record | PostgreSQL license |
| L2 Cache | Redis / Valkey | Cache/session/memory | Verify / BSD (Valkey) |
| L2 Stream | Kafka / Redpanda / NATS / RabbitMQ | Eventing | Apache-2.0 / verify |
| L2 Memory | Mem0 / Letta (MemGPT) | Agent long-term memory | Apache-2.0 (verify) |
| L1 Compute | Kubernetes | Container orchestration | Apache-2.0 |
| L1 Compute | Ray | Distributed compute | Apache-2.0 |
| L1 Sched | Volcano / Kueue | GPU/batch scheduling | Apache-2.0 |
| L1 GPU | NVIDIA GPU Operator | GPU enablement | Apache-2.0 |
| L0 IaC | OpenTofu / Terraform / Pulumi | Infrastructure as code | MPL / BUSL / Apache (verify) |
| L0 Control | Crossplane | Cloud-neutral control plane | Apache-2.0 |
| L0 Multi-cloud | SkyPilot | Cross-cloud GPU placement | Apache-2.0 |
| L0 GitOps | Argo CD / Flux | Declarative deploys | Apache-2.0 |
| L0 Policy | Kyverno / Gatekeeper | Infra policy-as-code | Apache-2.0 |
| X Identity | Keycloak / SPIFFE-SPIRE / Dex | Identity & workload identity | Apache-2.0 |
| X Secrets | Vault / OpenBao | Secrets management | BUSL / MPL (OpenBao) |
| X FinOps | OpenCost | Cost attribution | Apache-2.0 |
| X SDD | GitHub Spec Kit (Specify CLI) | Spec-driven workflow | MIT |

---

## Appendix B: Glossary

- **Agnostic seam:** an interface defined by an open standard so the implementation behind it is replaceable without app changes.
- **A2A (Agent-to-Agent):** open protocol for agents to discover and message each other across frameworks/vendors.
- **MCP (Model Context Protocol):** open protocol for exposing tools/resources/prompts to models and agents; Linux Foundation-hosted.
- **Neocloud:** GPU-specialist cloud provider (e.g., CoreWeave, Crusoe, Lambda, Nebius, Together, RunPod) offering raw/managed accelerator capacity.
- **OpenAI-compatible API:** the de-facto `/v1/chat/completions` (+ tools/embeddings) interface most engines and providers implement; the portability contract for L3/L4.
- **SDD (Spec-Driven Development):** methodology where a written spec is the source of truth and code is a generated artifact; canonical loop Specify → Plan → Tasks → Implement (here extended with Evaluate and Operate).
- **SkyPilot:** OSS framework that places GPU jobs on whichever cloud/neocloud has best price/availability.
- **Eval gate:** a CI checkpoint where quantitative quality/safety/cost thresholds must pass before merge/deploy.

---

*End of document. This is a living specification — keep it under version control and amend it deliberately as the ecosystem and your constitution evolve.*
