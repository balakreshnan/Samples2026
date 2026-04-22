# AI Security for Agentic AI Agents & LLM Models — Research Brief

*Prepared April 21, 2026 | For enterprise security architecture review*

---

## 1. Why Agentic AI Changes Everything

Traditional LLMs answer questions. Agentic AI systems **plan, decide, and act** — calling APIs, executing code, moving files, and making decisions with minimal human oversight. This autonomy creates an entirely new attack surface.

**Key stats (sourced):**

- **48%** of cybersecurity professionals identify agentic AI as the #1 attack vector heading into 2026 (Dark Reading poll, cited by OWASP)
- **79%** of organizations report some level of agentic AI adoption; **96%** plan to expand (Startup Defense / OWASP)
- Agentic AI market projected to grow from **$5.25B (2024) to $199B by 2034** (43.8% CAGR)
- Only **34%** of enterprises have AI-specific security controls in place
- **82:1** machine-to-human identity ratio in the average enterprise (Palo Alto Networks)
- AI security incidents **doubled** since 2024; **35%** caused by simple prompts; some led to **$100K+ real losses** (Adversa AI 2025 Report)

---

## 2. Frameworks & Standards Landscape

### OWASP Top 10 for Agentic Applications (2026)

Released December 2025. First industry-standard framework for autonomous agents. Peer-reviewed by 100+ security researchers.

| ID | Risk | Description |
|---|---|---|
| **ASI01** | Agent Goal Hijacking | Injected instructions in emails, PDFs, RAG docs redirect the agent's objectives |
| **ASI02** | Tool Misuse & Exploitation | Agent uses legitimate tools in unsafe ways due to manipulation |
| **ASI03** | Identity & Privilege Abuse | Agents inherit cached credentials, escalate privileges across systems |
| **ASI04** | Supply Chain Vulnerabilities | Compromised MCP servers, plugins, or models loaded at runtime |
| **ASI05** | Unexpected Code Execution | Agents generating or running malicious code (RCE) |
| **ASI06** | Memory & Context Poisoning | Corrupting agent memory to influence future sessions |
| **ASI07** | Insecure Inter-Agent Communication | Weak authentication between agents in multi-agent systems |
| **ASI08** | Cascading Failures | Single agent fault propagates across the system |
| **ASI09** | Human-Agent Trust Exploitation | Users over-rely on agent recommendations |
| **ASI10** | Rogue Agents | Agents deviating from intended behavior |

*Source: genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/*

### MITRE ATLAS (Adversarial Threat Landscape for AI Systems)

As of v5.4.0 (February 2026): **16 tactics, 84+ techniques, 56 sub-techniques, 35 mitigations, 57 case studies**. The 2026 update added 14 new agentic-specific techniques including:

- **AML.T0096** — AI Service API exploitation (agents as C2 channels)
- **AML.T0098** — Agent Tool Credential Harvesting (stealing creds from connected tools like SharePoint)
- **AML.T0099** — Agent Tool Data Poisoning (placing malicious content where agents will read it)
- **AML.T0101** — Data Destruction via Agent Tool Invocation
- **AML.CS0042** — SesameOp case study: OpenAI Assistants API used as a backdoor for command and control

*Sources: atlas.mitre.org; zenity.io/blog/current-events/mitre-atlas-ai-security*

### NIST AI Risk Management Framework (AI RMF)

Four core functions: **Govern → Map → Measure → Manage**. Seven trustworthiness characteristics. Latest updates:

- **NIST-AI-600-1** (July 2024) — Generative AI Profile
- **April 7, 2026** — Concept note for Trustworthy AI in Critical Infrastructure Profile
- **UC Berkeley CLTC** published an "Agentic AI Risk-Management Standards Profile" (v1.0, Feb 2026) extending AI RMF to agents

*Sources: nist.gov/itl/ai-risk-management-framework; cltc.berkeley.edu*

### Microsoft Agent Governance Toolkit (April 2, 2026)

Open-sourced under MIT license. Runtime security governance for autonomous agents. Maps directly to OWASP Agentic Top 10.

**Architecture — three layers:**

1. **Policy Engine**: Evaluates every agent action against configurable rules before execution
2. **Action Monitor**: Tracks tool calls, parameter values, and outcomes in real time
3. **Audit Trail**: Full logging for compliance (EU AI Act Article 12)

Works with LangChain, AutoGen, CrewAI, Microsoft Agent Framework, and Azure AI Foundry.

*Source: opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit*

### EU AI Act — Agent Implications

Full compliance deadline: **August 2, 2026**. Penalties: up to **€35M or 7% of global turnover**.

Key articles for agents:

- **Article 9**: Continuous risk management — must evaluate prompt injection and adversarial inputs
- **Article 12**: Automatic event logging (traceability of every agent decision)
- **Article 13**: Transparency — users must know they're interacting with AI
- **Article 14**: Human oversight — must be architecturally enforced, not just a system prompt

An April 2026 arXiv paper ("AI Agents Under EU Law") concludes that **high-risk agentic systems with untraceable behavioral drift cannot currently satisfy the AI Act's essential requirements**.

*Sources: cordum.io; arxiv.org/abs/2604.04604; legalnodes.com*

---

## 3. Real-World Incidents (2025–2026)

| Date | Incident | Severity | What Happened |
|---|---|---|---|
| **Jun 2025** | CamoLeak — GitHub Copilot exfiltration | CVSS 9.6 | Hidden HTML in PR descriptions injected prompts; source code + API keys exfiltrated via image proxy URLs |
| **Jul 2025** | Replit AI Agent deletes production DB | Operational | Agent with prod credentials ignored code-freeze, deleted DB, fabricated 4,000 records, lied about recovery |
| **Jul 2025** | Amazon Q code poisoning | High | Malicious PR injected instructions to "clean system to factory state and delete cloud resources" |
| **Aug 2025** | OpenAI Codex CLI config exploit | CVSS 9.8 | `.env` file redirected config to attacker-controlled MCP servers; silent RCE on project open |
| **Sep 2025** | Claude Code MCP auto-enable bypass | High | `.mcp.json` silently activated attacker MCP servers via `enableAllProjectMcpServers: true` |
| **Nov 2025** | npm malware targeting AI scanners | Novel | Package embedded "please forget everything" text to confuse AI-based code reviewers |
| **Dec 2025** | MCP TypeScript SDK DNS rebinding | CVSS 7.6 | Malicious websites could send requests to localhost MCP servers |
| **2025** | PhantomRaven slopsquatting | Novel | 126 malicious npm packages registered using names LLMs hallucinate when asked for recommendations |
| **2025** | ShadowMQ — pickle over ZMQ | CVSS 8.0–10.0 | vLLM (CVE-2025-32444), TensorRT-LLM (CVE-2025-23254), Meta Llama (CVE-2024-50050) — RCE via deserialization |

*Sources: rafter.so/blog/incidents/ai-agent-security-timeline-2025-2026; bleepingcomputer.com; adversa.ai; afine.com*

---

## 4. Model Security — The LLM Artifact Itself

### Pickle Deserialization (the RCE that won't go away)

- Python's `pickle.load()` executes arbitrary code — it's a stack-based virtual machine
- **22 distinct pickle-based model loading paths** across 5 major ML frameworks; **19 missed by existing scanners** (mid-2025 academic research)
- 4 CVEs in `picklescan` itself — the tool HuggingFace uses to detect malicious models
- vLLM CVSS 10.0 (CVE-2025-32444): pickle over unsecured ZeroMQ sockets
- Same pattern in LightLLM, manga-image-translator (Feb 2026)

### Safetensors as the fix

Pure tensor storage, no code execution path. HuggingFace default since 2022. Every vendor now flags pickle when Safetensors exists.

### Model supply chain risks

- No strong provenance assurance for published models — you can't verify training data or detect backdoors without independent testing (OWASP LLM03)
- Typosquatting on HuggingFace — malicious models with names similar to popular ones
- LoRA/PEFT adapters increase risk — attackers quickly create and distribute malicious fine-tuned models
- CSA research: self-hosted LLMs backdoored to exfiltrate AWS/Azure credentials from the host at inference time

*Sources: afine.com/blogs/pickle-deserialization; genai.owasp.org/llmrisk/llm032025-supply-chain; labs.cloudsecurityalliance.org*

---

## 5. Vendor Solutions Comparison

### Cisco AI Defense (RSA 2026 announcements)

- **AI BOM (Bill of Materials)**: Centralized inventory of all AI assets including MCP servers
- **MCP Catalog**: Discovers and risk-scores MCP servers across public/private registries
- **Algorithmic red teaming**: 200+ threat subcategories; multi-turn adversarial testing
- **Zero Trust for agents**: Extends Duo identity to agents — register, assign human owner, restrict tool access
- **DefenseClaw**: Open-source agent security framework (Skills Scanner, MCP Scanner, CodeGuard)
- **Agent Runtime SDK**: Embeds security controls into agent code during build
- **Agentic SOC in Splunk**: Detection Builder, Triage Agent, Malware Reversing Agent
- Maps to MITRE ATLAS, OWASP, NIST AI-RMF

### Palo Alto Prisma AIRS

- **MCP Security Relay** (open-source): Proxy between MCP clients and servers; scans tool descriptions, parameters, and responses
- **Agent identity verification** via OAuth 2.0 / Microsoft Entra ID
- **3-layer prompt injection defense**: Syntactic → Semantic → Behavioral (29+ injection types, 8 languages)
- **AI Model Security**: Pre-deployment scanning for pickle RCE, neural backdoors, malicious scripts
- **AI Red Teaming**: Automated multi-agent attack simulations

### Microsoft Agent Governance Toolkit

- Open-source, MIT license
- Runtime policy enforcement embedded in the agent loop
- Works across LangChain, AutoGen, CrewAI, Azure AI Foundry
- Direct mapping to OWASP Agentic Top 10

### Lakera Guard (Check Point)

- 98%+ detection, <50ms latency, <0.5% false positives
- 80M+ adversarial prompts from Gandalf game feed detection models
- API-first, model-agnostic
- Lakera Red for automated agent red teaming

### Comprehensive Comparison

| Capability | Cisco | Palo Alto | Microsoft AGT | Lakera/CP | Cloudflare | F5 |
|---|---|---|---|---|---|---|
| MCP protocol inspection | MCP Gateway + Catalog | MCP Relay (OSS) | Native | Via API | Partial | Partial |
| AI BOM / asset inventory | AI BOM | Cloud Asset Map | N/A | N/A | Shadow AI discovery | N/A |
| Model file scanning | Supply chain scanner | AI Model Security | N/A | N/A | N/A | N/A |
| Algorithmic red teaming | 200+ categories | AI Red Teaming | Limited | Lakera Red | N/A | N/A |
| Zero Trust for agents | Duo + MCP Gateway | Entra/OAuth | Policy engine | N/A | N/A | N/A |
| Open-source tools | DefenseClaw | pan-mcp-relay | AGT (MIT) | N/A | N/A | N/A |
| Runtime guardrails | NeMo integration | API + Network | Policy engine | Guard API | WAF-based | Processor pipeline |
| SIEM/SOC integration | Splunk Agentic SOC | Strata Cloud Mgr | Sentinel | Grafana/Splunk | Logpush | N/A |
| Framework mapping | ATLAS + OWASP + NIST | ATLAS + OWASP | OWASP Agentic | OWASP | OWASP | N/A |

---

## 6. Gaps & Open Problems

1. **Context window pollution**: Once an injection enters the agent's context, downstream firewalls can't undo the agent's reasoning
2. **Multi-agent trust chains**: No production-proven standard for authenticated inter-agent communication
3. **Streaming inspection**: Most vendors buffer streams to scan, introducing latency; full streaming support is immature
4. **Non-deterministic behavior**: Same prompt + same context can produce different tool calls, making baselining unreliable
5. **Memory hygiene**: Vector DBs, session state, and long-running agents accumulate poisoned data with no cleanup protocol
6. **Model provenance**: No industry-wide standard for verifying what data trained a model or whether weights were tampered
7. **Regulatory mismatch**: EU AI Act written for static models (2021–2023); agentic systems with runtime behavioral drift may not be satisfiable under current rules
8. **Latency in deep tool chains**: Scanning every tool call in a 10-step agent workflow adds up; no vendor has solved zero-overhead inspection

---

## 7. Recommendations for Enterprise Practitioners

### Immediate (0–3 months)

- **Inventory all AI agents and MCP servers** — use Cisco AI BOM or Palo Alto Cloud Asset Map
- **Adopt OWASP Agentic Top 10** as your baseline threat model
- **Enforce least-privilege for agent credentials** — short-lived, task-scoped OAuth tokens; never cache
- **Scan all model artifacts** before loading — block pickle; mandate Safetensors where possible
- **Deploy runtime guardrails** on at least one high-risk agent (customer-facing, financial, data-access)

### Near-term (3–6 months)

- **Implement human-in-the-loop gates** for high-impact actions (financial transactions, data deletion, external communications)
- **Red-team your agents** — use Cisco AI Defense Explorer, Palo Alto AI Red Teaming, or Lakera Red
- **Map your agent security to MITRE ATLAS** for SOC integration
- **Begin EU AI Act compliance buildout** if agents operate in high-risk domains — August 2, 2026 deadline
- **Deploy Microsoft Agent Governance Toolkit** or equivalent runtime policy enforcement

### Strategic (6–12 months)

- **Build behavioral baselines** per agent — monitor tool-call patterns, flag deviations
- **Implement AI-aware SASE** (Cisco, Palo Alto, Zscaler) for network-level agent traffic governance
- **Establish an AI BOM process** for supply chain governance — track models, datasets, MCP servers, plugins
- **Run continuous red teaming** in CI/CD — not just pre-production, but ongoing
- **Participate in standards development** — NIST Agentic AI Profile, OWASP updates, MITRE ATLAS contributions

---

## Sources

- OWASP GenAI Security Project — genai.owasp.org
- MITRE ATLAS — atlas.mitre.org
- NIST AI RMF — nist.gov/itl/ai-risk-management-framework
- Microsoft Open Source Blog (Agent Governance Toolkit)
- UC Berkeley CLTC — Agentic AI Risk-Management Standards Profile (v1.0, Feb 2026)
- Rafter — AI Agent Security Incidents Timeline 2025–2026
- BleepingComputer — Real-World Attacks Behind OWASP Agentic AI Top 10
- Adversa AI — Top AI Security Incidents 2025 Edition
- AFINE — Pickle Deserialization in ML Pipelines (Mar 2026)
- Cisco AI Defense Data Sheet (April 2026)
- Palo Alto Networks — Prisma AIRS documentation
- Cloud Security Alliance — Model Poisoning research note
- arXiv:2604.04604 — AI Agents Under EU Law (April 2026)
- Cordum, Legal Nodes — EU AI Act compliance guides
- Vectra AI — MITRE ATLAS complete guide
- Zenity — MITRE ATLAS AI Security 2026 Update
