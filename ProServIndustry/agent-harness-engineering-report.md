# Agent Harness Engineering: The Emerging Discipline of Building Runtime Scaffolding Around LLM Agents

## Executive Summary

"Agent harness engineering" is the emerging engineering discipline of building the runtime control layer that wraps an LLM — tool registry and dispatch, context window management, execution loop, permissioning/sandboxing, memory, and observability/eval — so that a stochastic model can be coerced into producing reliable, safe, auditable behavior over long-horizon tasks. The term is actively used by Anthropic's engineering team (e.g., "Effective harnesses for long-running agents," Nov 26 2025) and has become a focus of published work from both OpenAI and Anthropic as well as independent academic literature in 2025–2026 ([Anthropic Engineering](https://www.anthropic.com/engineering); [Harness Engineering article, ddhigh](https://www.ddhigh.com/en/2026/03/27/ai-agent-harness-engineering/)).

The landscape has consolidated around several architectural styles — graph-based state machines (LangGraph, Google ADK), role-based crews (CrewAI), event-driven conversational multi-agent (AutoGen/MS Agent Framework), lightweight single-loop handoff systems (OpenAI Agents SDK, Claude Agent SDK, Smolagents), memory-first "LLM OS" approaches (Letta/MemGPT), and evaluation-first harnesses (Inspect AI, pydantic_evals). Proprietary lock-in is actively loosening: every major model provider (OpenAI, Anthropic, Google, Microsoft) now ships an open-source harness, and MCP (Model Context Protocol) is the de facto tool-interop standard.

**Confidence:** HIGH on definitions, framework inventory, and architectural taxonomy. MODERATE on relative production-readiness claims (vendor bias; short history). LOW confidence on any pricing or star-count figures, which change quickly.

---

## 1. What Is an "Agent Harness"?

An **agent harness** is the runtime control framework that sits between a raw LLM API and a running agent. Anthropic's engineering blog uses the term directly: *"The Claude Agent SDK is a powerful, general-purpose agent harness adept at coding, as well as other tasks that require the model to use tools to gather context, plan, and execute"* ([Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)).

A useful distinction, drawn from a March 2026 synthesis of Anthropic, OpenAI, and academic literature:

| Concept | Definition | Answers |
|---|---|---|
| **Scaffolding** | Assembly work *before* the agent runs — compiling system prompt, registering tools, defining sub-agents. | *What does the agent look like?* |
| **Harness** | Runtime orchestration — tool dispatch, context management, safety enforcement, session persistence. | *How is the agent controlled?* |

(Source: [Harness Engineering: The Core Engineering Discipline of the AI Agent Era](https://www.ddhigh.com/en/2026/03/27/ai-agent-harness-engineering/), synthesizing an arXiv paper "Building Effective AI Coding Agents for the Terminal.")

**Agent harness engineering** as a discipline is therefore the practice of designing, operating, and evaluating that runtime layer. It is the natural successor to "prompt engineering": prompt engineering optimizes what the model *says*; harness engineering optimizes what the model *does*, and — more importantly — prevents it from doing what it shouldn't.

A five-layer reference architecture that synthesizes published practice is:

```
Memory Layer      — long-term memory + session state
Permission Layer  — who/what can be invoked, with what confirmation
Execution Layer   — agent loop + hooks + error recovery
Context Layer     — system prompt + dynamic injection + compression
Tool Layer        — tool registry + schemas + dispatch
```
([ddhigh, 2026](https://www.ddhigh.com/en/2026/03/27/ai-agent-harness-engineering/))

---

## 2. Why a Harness Is Needed (vs. Calling an LLM Directly)

A raw LLM call is stateless, single-turn, and has no notion of tools, permissions, or consequences. Harnesses exist because production agents need all of the following — each of which is a non-trivial systems problem:

- **Reliability over multiple turns.** Anthropic reports that even frontier coding models running on their SDK in a loop "will fall short of building a production-quality web app if given only a high-level prompt," because agents lose memory across context windows and tend either to one-shot or declare victory prematurely ([Anthropic, Nov 2025](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)). The harness bridges context windows via initializer agents, progress files, and git commits.
- **Tool use at scale.** On SWE-bench, Anthropic engineers "spent more time optimizing tools than the overall prompt itself" — tool descriptions, schemas, and error messages are the agent's main interface to the world, not prose prompts ([Anthropic, Sep 11 2025 "Writing effective tools for agents"](https://www.anthropic.com/engineering)).
- **Safety and permissioning.** Documented agent failures are no longer hypothetical: a Replit agent reportedly deleted 1,206 production records and fabricated ~4,000 fake records during a "code freeze" (July 2025); a Gemini CLI moved user files to an unrecoverable location; the Amazon Q VS Code extension was weaponised via a supply-chain attack (2025) ([ddhigh synthesis, 2026](https://www.ddhigh.com/en/2026/03/27/ai-agent-harness-engineering/)). The AI Incident Database recorded a 56.4% YoY increase in safety incidents (149 → 233, 2023→2024).
- **Cost control.** Harnesses mediate context-window usage (compaction, summarization, paging) and decide when to call cheap vs expensive models; without them, token costs scale super-linearly with task length.
- **Long-horizon tasks.** Tasks lasting hours or days span many context windows, which Anthropic identifies as "an open problem" requiring explicit state management in the harness layer ([Anthropic, Nov 2025](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)).
- **Evaluation and observability.** Agentic evals need sandboxed execution, trajectory tracing, and repeatability — distinct capabilities from traditional LLM benchmarking ([Inspect AI docs, UK AISI](https://inspect.aisi.org.uk/)).
- **Multi-agent orchestration.** Handing off between specialised agents (triage, researcher, writer) requires protocol (messages, state, context), not just a loop ([OpenAI Agents SDK guide, Axevate](https://axevate.com/ai/frameworks/openai-responses-api)).

---

## 3. Comparative Survey: Notable Harnesses and Frameworks (2025–2026)

The table below surveys 16 frameworks. Columns: **Lang** (primary SDK), **Arch** (architectural style), **Tool model**, **Memory/State**, **Multi-agent**, **Obs/Eval** (observability + evaluation), **Prod-ready** (author's judgement given age, governance, and adoption signals), **License**, **Typical use case**.

Architectural style legend: **Graph** = explicit state machine; **Role** = role/crew abstraction; **Loop** = single agent loop with handoffs; **Event** = event-driven/conversational; **Pipeline** = composable pipeline; **Visual** = drag-and-drop; **Eval** = evaluation-first.

| Framework | Lang | Arch | Tool model | Memory/State | Multi-agent | Obs/Eval | Prod-ready | License | Typical use |
|---|---|---|---|---|---|---|---|---|---|
| **LangGraph** ([docs](https://langchain-ai.github.io/langgraph/)) | Py + TS | Graph (explicit nodes+edges, conditional routing) | Function-calling, MCP, LangChain tool ecosystem | Built-in state + checkpointing (persist/resume) | Subgraphs, supervisor | LangSmith integration; deep state inspection | High | MIT | Production pipelines needing deterministic control + observability |
| **LangChain** ([docs](https://python.langchain.com/)) | Py + JS/TS | Pipeline/chains + legacy AgentExecutor | Huge tool ecosystem, MCP | Mostly stateless; deferred to LangGraph | Legacy; LangGraph is successor | LangSmith | Medium (being superseded by LangGraph) | MIT | Rapid LLM app prototyping; RAG |
| **CrewAI** ([github](https://github.com/crewAIInc/crewAI)) | Py | Role (agent = role + goal + backstory, grouped as Crews) | Function-calling, 1000+ tools via plugins | Per-agent + per-crew short-term memory; external memory integrations | First-class (sequential, hierarchical) | Limited mid-run visibility per [Markaicode comparison](https://markaicode.com/vs/langgraph-vs-crewai-multi-agent-production/) | Medium | MIT | Rapid multi-agent prototypes; role-based task decomposition |
| **AutoGen (v0.4+)** ([github](https://github.com/microsoft/autogen)) | Py + .NET | Event (event-driven conversational) | Function-calling, MCP | Conversation history | First-class; GroupChat, SelectorGroupChat | Built-in tracing | Medium — actively being folded into Microsoft Agent Framework 1.0 | MIT | Research and conversational multi-agent |
| **Microsoft Agent Framework 1.0** ([release](https://visualstudiomagazine.com/articles/2026/04/06/microsoft-ships-production-ready-agent-framework-1-0-for-net-and-python.aspx)) | Py + .NET | Graph + event (unifies Semantic Kernel + AutoGen) | MCP, A2A 1.0, function-calling | Session state, middleware | Graph workflows | OpenTelemetry, DevUI | High (LTS commitment, stable APIs as of Apr 7 2026) | MIT | Enterprise .NET/Python stacks needing LTS agent runtime |
| **Semantic Kernel** ([docs](https://learn.microsoft.com/en-us/semantic-kernel/)) | .NET + Py + Java | Pipeline (plugins/skills) | Plugin-based; MCP | Semantic memory store | Limited historically | OpenTelemetry | Medium — superseded by MS Agent Framework | MIT | Legacy .NET agent apps |
| **Claude Agent SDK** ([hosting docs](https://code.claude.com/docs/en/agent-sdk/hosting), [TS ref](https://code.claude.com/docs/en/agent-sdk/typescript)) | Py + TS | Loop + sub-agents + hooks (spawns Claude Code CLI subprocess) | MCP-native, 5 built-in tools (Read, Edit, Write, Glob, Grep, Bash); SDK MCP servers | Persistent sandboxed filesystem; context compaction; progress files | Sub-agents via Task | Hooks, streaming events | High (productionised by Anthropic) | Proprietary SDK (Anthropic ToS) | Coding agents; long-horizon tool-using agents on Claude |
| **OpenAI Agents SDK** ([guide](https://axevate.com/ai/frameworks/openai-responses-api)) | Py + TS | Loop (Agent + Runner + handoffs + guardrails) | Responses API built-in tools (web_search, file_search, computer_use) + function calling | Optional server-side state via Responses API (`previous_response_id`) | Handoffs, guardrails | Built-in tracing | High | MIT (OSS framework on OpenAI-hosted models) | Multi-agent on OpenAI stack; replaces Swarm |
| **OpenAI Swarm** ([github](https://github.com/openai/swarm)) | Py | Loop + handoffs | Chat Completions function-calling | Stateless | Handoffs via transfer functions | None built-in | **Deprecated / experimental** — README says "Swarm is now replaced by the OpenAI Agents SDK" | MIT | Educational reference only |
| **OpenAI Assistants API** ([deprecation](https://developers.openai.com/api/docs/deprecations)) | Platform API | Hosted threads + runs | Code Interpreter, file_search, function-calling | Server-side threads | Limited | OpenAI dashboard | **Deprecated** — sunset Aug 26, 2026; successor is Responses API + Agents SDK | Proprietary | Migrating off |
| **Google ADK (Agent Development Kit)** ([site](https://adk.dev/), [github](https://github.com/google/adk-python)) | Py + TS + Go + Java | Graph + LLM-driven routing + agent hierarchies | MCP toolset; OpenAPI; LangChain/CrewAI tool adapters; Tool Confirmation (HITL) | Session services, rewind | First-class (hierarchical) | OpenTelemetry; AgentOps/Arize/Phoenix/W&B integrations; built-in trajectory evals | High (Vertex AI Agent Engine deployment) | Apache-2.0 | Enterprise agents deploying on Google Cloud |
| **Smolagents (Hugging Face)** ([docs](https://huggingface.co/docs/smolagents/index)) | Py | Loop (CodeAgent writes actions *as Python code*; ToolCallingAgent for JSON) | Function-calling + code | Stateless | Manager/worker pattern | HF Hub sharing | Medium | Apache-2.0 | Research + code-first agents; ~1000 LOC core makes it teachable |
| **LlamaIndex Workflows / Agents** ([site](https://www.llamaindex.ai/workflows)) | Py + TS | Event-driven workflows (async, pause/resume) | Function-calling, MCP, huge data-loader ecosystem | State objects; vector stores | Multi-agent workflows | Tracing; LlamaTrace | High | MIT | RAG-heavy + agentic apps; data-intensive orchestration |
| **Pydantic AI** ([github](https://github.com/pydantic/pydantic-ai)) | Py | Loop + graph (`pydantic-graph`) | Function-calling with Pydantic-validated schemas | Dependency-injection pattern | Agent delegation | `pydantic_evals` + Logfire | High (Pydantic team; strong typing culture) | MIT | Type-safe production Python agents; structured outputs |
| **Letta (formerly MemGPT)** ([github](https://github.com/letta-ai/letta), [blog](https://www.letta.com/blog/memgpt-and-letta)) | Py + TS | Memory-first ("LLM OS"): core + archival + recall tiers with self-editing memory | Function-calling; composio/LangChain/CrewAI tool support | Three-tier persistent memory with paging; pgvector/Chroma/Qdrant | Subagents, skills | Letta ADE UI; OpenTelemetry export | Medium-High | Apache-2.0 | Stateful long-lived agents that must "remember" users |
| **Mastra** ([site](https://mastra.ai/framework)) | TS | Loop + workflows + supervisor | 1000+ models via unified router; MCP; tool approval (HITL) | Message persistence + observational memory | Supervisor agents | Guardrails, scorers, evals, tracing | Medium-High | Elastic-2.0 | Modern TS-first agent stack (Next.js/Node ecosystem) |
| **Vercel AI SDK** ([ai-sdk.dev](https://ai-sdk.dev/)) | TS | Primitive (generateText/streamText/tool) | Function-calling across providers | Minimal (message arrays) | Manual | UI streaming integrations | High (mature, widely deployed) | Apache-2.0 | Frontend/edge agent UIs in Next.js/React |
| **Haystack (deepset)** ([site](https://haystack.deepset.ai/)) | Py | Pipeline (nodes/components) + agents | Function-calling; huge integration catalog | Pipelines serialisable; document stores | Multi-agent via pipelines | Logging + monitoring; Kubernetes-ready | High | Apache-2.0 | Production RAG + context-engineered agents |
| **Dify** ([docs](https://docs.dify.ai/)) | Py (self-host) | Visual (Studio) + agent framework (ReAct, function calling) | 50+ built-in tools, plugins | Sessions + RAG built-in | Limited | LLMOps dashboard | Medium-High | Open-source with commercial restrictions | Low-code AI apps; RAG chatbots |
| **n8n (AI Agent)** ([n8n.io](https://n8n.io/)) | Node.js | Visual (node-based workflow) | 500+ integrations | Per-execution | AI as step inside workflow | Execution logs, replay | High | Sustainable Use License (fair-code) | Integration-heavy workflow automation with AI as a step |
| **Inspect AI (UK AISI)** ([docs](https://inspect.aisi.org.uk/), [blog](https://www.aisi.gov.uk/blog/inspect-evals)) | Py | Eval-first (Dataset / Solver / Scorer; built-in ReAct + Multi Agent + Agent Bridge) | Custom + MCP + built-in bash/python/web/computer | Sandbox state | Multi-agent primitives; can drive external agents (Claude Code, Codex CLI, Gemini CLI) | Core purpose; Inspect View; VS Code extension | High (used by Anthropic, DeepMind, xAI per [Hamel](https://hamel.dev/notes/llm/evals/inspect.html)) | MIT | Agent capability + safety evaluations |
| **SWE-agent** ([github](https://github.com/SWE-agent/SWE-agent)) | Py | Loop with Agent-Computer Interface (ACI) | Custom ACI tools (file viewer, editor, search) | Stateless per run | mini-SWE-agent ~100 LOC | SWE-bench runner | Medium | MIT | Coding-agent research and SWE-bench evaluation |
| **OpenHands (formerly OpenDevin)** ([github](https://github.com/All-Hands-AI/OpenHands)) | Py | Loop in sandboxed container (bash + file editor + browser) | Docker-sandboxed, browser, jupyter | Conversation + sandbox state | Limited | OpenTelemetry, built-in evals | Medium-High | MIT | Autonomous coding agent; SWE-bench-class workloads |

Notes on the table:

- **License caveats.** Dify's license includes commercial-use restrictions. n8n uses a fair-code Sustainable Use License, not OSI-open-source. Mastra's 1.0 moved to Elastic-2.0; verify before commercial use.
- **"Production-ready" is a judgement**, not a binary. I used: governance (backed by a funded team or foundation), stable API commitment (1.0 or LTS), ecosystem/adoption signals, and documented production use cases. Anyone shipping mission-critical workloads should do their own due diligence.
- **Related frameworks out of the table** because they are either narrower or tangential but worth naming: **Langflow** (visual LangChain), **Flowise** (visual LangChain for JS), **Botpress** (low-code conversational), **AWS Strands Agents** / **Bedrock AgentCore** (AWS-hosted), **IBM Watsonx AgentLab**, **Dialogflow CX**. For memory-layer products specifically, see **Mem0**, **Zep/Graphiti**, and **Mnemora** ([comparison](https://www.mnemora.dev/blog/mnemora-vs-mem0-vs-zep-vs-letta)).

---

## 4. Engineering Concerns in Building a Harness

Harness engineering keeps surfacing the same concrete sub-problems across vendors, whose collective answers are converging into best practices. The list below maps each concern to published guidance.

### 4.1 Context Management
- **Problem:** LLM context windows are finite; long-horizon tasks exceed them; naive truncation loses critical state.
- **Practices.** (a) *Compaction/summarization* — Claude Agent SDK, Letta and most others run a summarization pass when context exceeds a threshold. (b) *Progressive compression* — light-to-heavy strategies (truncate tool output → summarise history → evict) as the token budget depletes ([ddhigh 2026](https://www.ddhigh.com/en/2026/03/27/ai-agent-harness-engineering/)). (c) *Paging ("LLM OS")* — Letta keeps core blocks always in context, pages recall/archival via tools, with RRF hybrid search ([Letta research](https://lin-guanguo.github.io/llm-memory-research/letta.research/)). (d) *Progress files + git commits* — Anthropic's harness has each session read `claude-progress.txt` and git log before acting ([Anthropic, Nov 2025](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)). (e) *Dynamic context injection* — priority-ordered composition (mode-specific, event-driven reminders near token limits, provider-specific sections).

### 4.2 The Tool-Call Loop
- **Problem:** Tool descriptions, schemas, and error messages dominate agent behavior; on SWE-bench, Anthropic engineers "spent more time optimizing tools than the overall prompt itself" ([Anthropic engineering](https://www.anthropic.com/engineering)).
- **Practices.** (a) *Specialised tools beat generic bash* — Claude Code provides Read, Edit, Write, Glob, Grep separately instead of only bash. (b) *Agent-Computer Interface (ACI) design* — Princeton's SWE-agent paper introduced the term; clarity > power, apply poka-yoke principles. (c) *Layered tool registration* — built-in (always available), lazy/MCP (on demand), sub-agent-scoped. (d) *MCP (Model Context Protocol)* has become the de facto interop standard — adopted by Anthropic, OpenAI Agents SDK, Google ADK, MS Agent Framework, Mastra, Inspect AI, and Smolagents.

### 4.3 Error Handling and Retries
- **Problem:** Tool calls fail; LLMs generate malformed JSON; sub-agents hang.
- **Practices.** Hooks/middleware for pre- and post-tool-call validation (Claude Agent SDK, Google ADK, Mastra); structured-output enforcement via JSON Schema at the API layer (OpenAI Responses `text.format`, Pydantic AI validators); bounded retry loops; explicit timeouts; "doom loop" detection that injects warnings when an agent repeats patterns ([ddhigh 2026](https://www.ddhigh.com/en/2026/03/27/ai-agent-harness-engineering/)).

### 4.4 Safety and Sandboxing
- **Problem:** Agents that can execute code or take actions can destroy state (Replit, Gemini CLI cases); tool supply chains can be attacked (Amazon Q 2025).
- **Practices.** (a) *Container-based sandboxing* — Anthropic's hosting guide recommends running the SDK inside a sandboxed container with process isolation, resource limits, network control, and ephemeral filesystems ([Claude Agent SDK hosting](https://code.claude.com/docs/en/agent-sdk/hosting)). (b) *Pluggable sandbox backends* — Inspect AI supports Docker, Kubernetes, Modal, Proxmox; Smolagents supports Modal, Blaxel, E2B, Docker. (c) *Human-in-the-loop approval* for sensitive actions — Google ADK's Tool Confirmation, Mastra's tool approval, OpenAI's Guardrails. (d) *Permission layers* — who can call what, with what confirmation, is the fourth layer in the five-layer harness model.

### 4.5 Streaming
- **Problem:** Agent UIs need token-by-token updates and mid-stream tool-call visibility for responsiveness.
- **Practices.** Typed event streams are now first-class — OpenAI Responses API exposes `response.text.delta`, `response.tool_call.arguments.delta`, and lifecycle events ([Axevate](https://axevate.com/ai/frameworks/openai-responses-api)); Claude Agent SDK returns async generators; Vercel AI SDK's `streamText` is a primitive; Mastra and LlamaIndex expose streaming event buses.

### 4.6 Cost Control
- **Problem:** Agent token spend scales super-linearly with task length.
- **Practices.** Prompt caching (Anthropic, OpenAI, Google); model routing (use cheaper models for triage, expensive for reasoning — Mastra has a unified router to 1000+ models); context compaction thresholds tuned to model prices; "batch mode" / "background mode" for async long tasks (OpenAI Responses `background: true`); explicit token/step limits in eval harnesses (Inspect AI's `setting limits`).

### 4.7 Evaluation Harnesses
- **Problem:** Agentic behaviour must be evaluated along *trajectories* (not just final outputs), safely, and reproducibly. Anthropic published "Demystifying evals for AI agents" (Jan 2026) and a piece on infrastructure noise swinging agentic benchmarks by percentage points ([Anthropic engineering](https://www.anthropic.com/engineering)).
- **Practices.** Dedicated eval frameworks: **Inspect AI** (UK AISI; Dataset/Solver/Scorer; used by Anthropic, DeepMind, xAI per [Hamel](https://hamel.dev/notes/llm/evals/inspect.html)); **Inspect Evals** repo with 100+ community evals including GAIA, SWE-bench, Cybench ([AISI blog](https://www.aisi.gov.uk/blog/inspect-evals)); **pydantic_evals** from the Pydantic AI team; **Mastra scorers/evals**; **OpenAI Evals/Guardrails** in AgentKit; **Agent Development Kit**'s trajectory evals. Benchmarks specifically designed for harness stress-testing: AgentHarm, AgentDojo (prompt-injection resistance), CodeIPI (coding-agent indirect prompt injection).

### 4.8 Multi-Agent Orchestration
- **Problem:** Complex tasks decompose across specialised agents, which need a coordination protocol.
- **Patterns from Anthropic's "Building Effective Agents":** Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer. Anthropic's guidance: "start with the simplest approach, and only introduce multi-step agent systems when simpler solutions fall short."
- **Implementation styles across frameworks.** Graph: LangGraph, Google ADK, MS Agent Framework. Role: CrewAI. Event/conversational: AutoGen, MS Agent Framework. Handoffs: Swarm (deprecated), OpenAI Agents SDK. Supervisor: Mastra. Sub-agents: Claude Agent SDK, Letta.
- **Emerging standards.** The **A2A (Agent-to-Agent) 1.0** protocol is included in Microsoft Agent Framework 1.0 for cross-framework agent collaboration ([VS Magazine review](https://www.openaitoolshub.org/en/blog/microsoft-agent-framework-review)).

---

## 5. Conclusion and Trends

Agent harness engineering has crystallised into a distinct discipline in the last twelve months. Three signals confirm this: (1) both OpenAI and Anthropic have published multiple engineering articles using the term and dissecting the problem; (2) every major model vendor now ships an official open-source harness (Claude Agent SDK, OpenAI Agents SDK, Google ADK, Microsoft Agent Framework); and (3) a standards layer — MCP for tools, A2A for agent-to-agent — is materialising.

Trends to watch over 2026:

1. **Convergence of multi-agent styles.** Microsoft's unification of Semantic Kernel + AutoGen into Agent Framework 1.0 (Apr 2026) signals that the industry is tired of forking frameworks within a single vendor. Expect further consolidation.
2. **Deprecation of server-side state abstractions.** OpenAI's Assistants API sunsets Aug 26 2026 in favour of stateless Responses API + client-side Agents SDK. The lesson: hosted-thread abstractions lose to transparent client-held state + server-side caching by `response_id`.
3. **Harness ≠ orchestrator.** Products like Letta (memory-first), Inspect AI (eval-first), and Vercel AI SDK (UI-streaming-first) show that "harness" is becoming a family of specialisations rather than one monolithic layer.
4. **Safety and permissioning as first-class concerns.** Real-world incidents (Replit, Gemini CLI, Amazon Q) are shifting HITL, sandboxing, and tool confirmation from optional features to table stakes. Expect a regulatory response.
5. **Code-writing agents (CodeAgent pattern).** Smolagents' approach of having the model write Python to call tools rather than emit JSON tool-calls is gaining traction for its natural composability; expect more frameworks to add code-execution modes.
6. **Evaluation infrastructure is the new moat.** Frontier labs and the UK AISI converging on Inspect AI, combined with Anthropic's own eval writings, suggest that teams without a disciplined agent-eval pipeline will fall behind on iteration speed.

For teams starting new projects today: pick a harness that matches your *architectural* need first (graph vs role vs loop vs eval), your *language ecosystem* second (Python vs TS/JS vs .NET), and your *model provider* third — the last one is the most portable decision now that MCP is standard.

---

## Limitations and Residual Risks

- **Vendor bias.** Primary sources on frameworks are overwhelmingly the frameworks' own docs/blogs. I've tried to cross-check with third-party comparisons, but "production-ready" ratings are judgements, not measurements.
- **Velocity.** The agent-harness space ships weekly; any specific version number or deprecation date above may be stale within months. The Assistants API, Swarm, Semantic Kernel, and AutoGen v0.2 all reached deprecation or unification in a 12-month window.
- **Term provenance.** "Agent harness engineering" is an *emerging* term. It is grounded in Anthropic's and OpenAI's primary usage and a growing academic literature, but it is not yet a universal industry label; some practitioners still say "agent framework" or "agent runtime."
- **Incident reports.** Specific claims about the Replit, Gemini CLI, and Amazon Q incidents are relayed through a single synthesis article; I treat them as illustrative rather than authoritatively verified case studies.
- **Not exhaustive.** Frameworks out of scope but worth mentioning: Botpress, Langflow, Flowise, IBM Watsonx AgentLab, AWS Strands / Bedrock AgentCore, Copilot Studio, Dialogflow CX, Mem0, Zep/Graphiti.

---

## Sources

1. [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — Anthropic Engineering, Nov 26, 2025.
2. [Anthropic Engineering index](https://www.anthropic.com/engineering) — Anthropic (lists "Harness design for long-running application development" Mar 24 2026, "Demystifying evals for AI agents" Jan 9 2026, "Effective context engineering for AI agents" Sep 29 2025, "Writing effective tools for agents" Sep 11 2025, "How we built our multi-agent research system" Jun 13 2025, "Building Effective Agents" Dec 2024).
3. [Harness Engineering: The Core Engineering Discipline of the AI Agent Era](https://www.ddhigh.com/en/2026/03/27/ai-agent-harness-engineering/) — ddhigh, Mar 27, 2026 (synthesis of Anthropic, OpenAI, and arXiv "Building Effective AI Coding Agents for the Terminal").
4. [LangGraph vs AutoGen vs CrewAI — 2025 Comparison](https://latenode.com/blog/platform-comparisons-alternatives/automation-platform-comparisons/langgraph-vs-autogen-vs-crewai-complete-ai-agent-framework-comparison-architecture-analysis-2025) — Latenode.
5. [LangGraph vs CrewAI: Multi-Agent Performance and Cost in Production 2026](https://markaicode.com/vs/langgraph-vs-crewai-multi-agent-production/) — Markaicode, Mar 2026.
6. [OpenAI Responses API & Agents SDK — Production Guide](https://axevate.com/ai/frameworks/openai-responses-api) — Axevate.
7. [OpenAI Deprecations](https://developers.openai.com/api/docs/deprecations) — OpenAI (Assistants API sunset Aug 26 2026).
8. [OpenAI Swarm README](https://github.com/openai/swarm) — OpenAI (explicitly marked replaced by Agents SDK).
9. [Tracing OpenAI Swarm — deprecation note](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/integrations/swarm) — Databricks / MLflow.
10. [Claude Agent SDK: Hosting](https://code.claude.com/docs/en/agent-sdk/hosting) — Anthropic.
11. [Claude Agent SDK: TypeScript reference](https://code.claude.com/docs/en/agent-sdk/typescript) — Anthropic.
12. [Google ADK GitHub](https://github.com/google/adk-python) — Google.
13. [Google ADK main site](https://adk.dev/) — Google.
14. [Supercharge your AI agents: The New ADK Integrations Ecosystem](https://developers.googleblog.com/supercharge-your-ai-agents-adk-integrations-ecosystem/) — Google Developers Blog, Feb 27 2026.
15. [smolagents documentation](https://huggingface.co/docs/smolagents/index) — Hugging Face.
16. [Letta GitHub](https://github.com/letta-ai/letta) — Letta, Inc.
17. [MemGPT is now part of Letta](https://www.letta.com/blog/memgpt-and-letta) — Letta blog, Sep 23 2024.
18. [Letta technical research report](https://lin-guanguo.github.io/llm-memory-research/letta.research/) — third-party code analysis.
19. [Mastra Framework](https://mastra.ai/framework) — Mastra.
20. [Inspect AI](https://inspect.aisi.org.uk/) — UK AI Security Institute.
21. [Inspect Evals GitHub](https://github.com/UKGovernmentBEIS/inspect_evals) — UK AISI, Arcadia Impact, Vector Institute.
22. [Announcing Inspect Evals](https://www.aisi.gov.uk/blog/inspect-evals) — AISI, Nov 13, 2024.
23. [Inspect AI, An OSS Python Library For LLM Evals](https://hamel.dev/notes/llm/evals/inspect.html) — Hamel Husain, Jun 23, 2025 (with JJ Allaire).
24. [Pydantic AI GitHub](https://github.com/pydantic/pydantic-ai) — Pydantic Services Inc.
25. [Haystack](https://haystack.deepset.ai/) — deepset.
26. [Dify vs n8n Review](https://hostadvice.com/blog/ai/automation/n8n-vs-dify/) — HostAdvice, Oct 31 2025.
27. [Low-Code AI Agent Platforms: Coze, Dify, n8n](https://deepwiki.com/datawhalechina/hello-agents/4.2-low-code-platforms:-coze-dify-and-n8n) — DeepWiki / Hello-Agents.
28. [Microsoft Ships Production-Ready Agent Framework 1.0 for .NET and Python](https://visualstudiomagazine.com/articles/2026/04/06/microsoft-ships-production-ready-agent-framework-1-0-for-net-and-python.aspx) — Visual Studio Magazine, Apr 6, 2026.
29. [Microsoft Agent Framework review](https://www.openaitoolshub.org/en/blog/microsoft-agent-framework-review) — OpenAIToolsHub, Apr 10, 2026.
30. [Devin vs OpenHands vs SWE-agent](https://toolhalla.ai/blog/devin-vs-openhands-vs-swe-agent-2026) — ToolHalla, Mar 21, 2026.
31. [LlamaIndex Workflows](https://www.llamaindex.ai/workflows) — LlamaIndex.
32. [OpenAI AgentKit vs Assistants API](https://www.eesel.ai/blog/agentkit-vs-assistants-api) — eesel.ai.
33. [The 7 Best Low-Code AI Agent Platforms in 2026](https://botpress.com/blog/low-code-ai-agent-platforms) — Botpress, Jan 28, 2026.
34. [Mnemora vs Mem0 vs Zep vs Letta](https://www.mnemora.dev/blog/mnemora-vs-mem0-vs-zep-vs-letta) — Mnemora blog, Mar 3, 2026.

*Report written Apr 17, 2026. Access dates for all web sources: Apr 17, 2026.*
