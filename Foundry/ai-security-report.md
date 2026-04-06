# Comprehensive AI Security: Evaluation, Attack Surfaces, and Defense Strategies (2024-2026)

## Executive Summary

AI security has evolved from a niche academic concern into a critical enterprise discipline, driven by the rapid deployment of LLMs, autonomous agents, and inter-agent protocols into production environments. This report provides a comprehensive technical analysis across seven domains of AI security: model security evaluation, agent security, MCP (Model Context Protocol) security, A2A (Agent-to-Agent) protocol security, red teaming methodologies, open-source tooling, and additional security considerations including supply chain, RAG, and infrastructure risks.

**Key finding:** While defensive frameworks have matured significantly -- OWASP LLM Top 10, MITRE ATLAS (now at 16 tactics and 84+ techniques), and NIST AI RMF provide structured coverage -- the attack surface is expanding faster than defenses can keep pace, particularly in emerging protocol layers (MCP and A2A) where authentication gaps and tool poisoning represent immediate, exploitable risks. The MCPTox benchmark demonstrates that even capable models like o1-mini exhibit 72.8% attack success rates for tool poisoning, and the highest refusal rate observed (Claude 3.7 Sonnet) is below 3%.

**Confidence:** MODERATE-HIGH. The core findings on model-level security and red teaming are well-supported by multiple academic and industry sources. MCP and A2A security findings are supported but rely on newer research given the protocols' youth.

---

## 1. Model Security Evaluation

### 1.1 Current State of the Field

The evaluation of LLM and AI model security has matured considerably through 2024-2025, with the establishment of standardized frameworks, benchmarks, and taxonomies. Three pillars define the current landscape:

**OWASP LLM Top 10 (2025 Release, November 2024)** identifies the ten most critical risks for LLM applications [OWASP Top 10 for LLM Applications 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf):

| Rank | Vulnerability | Description |
|------|--------------|-------------|
| LLM01 | Prompt Injection | Malicious inputs that override intended behavior (direct, indirect, multi-modal) |
| LLM02 | Sensitive Information Disclosure | Leaking PII, system prompts, API keys, training data |
| LLM03 | Supply Chain | Compromised models, poisoned training data, malicious packages |
| LLM04 | Data and Model Poisoning | Manipulated training data introducing backdoors or biases |
| LLM05 | Improper Output Handling | Unsanitized outputs causing XSS, SQLi, code execution downstream |
| LLM06 | Excessive Agency | Over-privileged tool access, insufficient guardrails on actions |
| LLM07 | System Prompt Leakage | Extraction of confidential system instructions |
| LLM08 | Vector and Embedding Weaknesses | Poisoned embeddings, retrieval manipulation |
| LLM09 | Misinformation | Hallucinations and factually incorrect outputs |
| LLM10 | Unbounded Consumption | Resource exhaustion, denial-of-service via crafted inputs |

**MITRE ATLAS** (Adversarial Threat Landscape for Artificial-Intelligence Systems) provides a structured threat taxonomy modeled after MITRE ATT&CK. As of version 5.1.0 (November 2025), ATLAS catalogs 16 tactics, 84 techniques, 56 sub-techniques, 32 mitigations, and 42 real-world case studies [MITRE ATLAS](https://atlas.mitre.org/). The February 2026 update (v5.4.0) added agent-focused techniques, reflecting the shift toward agentic AI threats [Vectra AI - MITRE ATLAS Guide](https://www.vectra.ai/topics/mitre-atlas).

**NIST AI Risk Management Framework (AI RMF 1.0)**, released January 2023 with the Generative AI Profile (NIST-AI-600-1) added July 2024, provides the governance backbone for AI risk management. It structures activities around four functions: Govern, Map, Measure, and Manage -- with adversarial testing explicitly included in the Manage function [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).

### 1.2 Key Attack Categories and Techniques

**Prompt Injection** remains the most critical and pervasive attack vector. Types include:
- **Direct injection:** User-crafted inputs that override system instructions
- **Indirect injection:** Hidden instructions in external data sources (web pages, documents, RAG contexts) that the LLM processes
- **Multi-modal injection:** Attacks embedded in images, audio, or video processed by vision-language models
- **Many-shot jailbreaking:** Exploiting long context windows to gradually shift model behavior

**Jailbreaking** techniques have become increasingly sophisticated. The Crucible dataset (214,271 attack attempts) shows that automated approaches achieve 69.5% success rates compared to 47.6% for manual techniques, though only 5.2% of users employ automation [The Automation Advantage in AI Red Teaming (arXiv 2504.19855)](https://arxiv.org/html/2504.19855v2). Specific techniques include:
- **GCG (Greedy Coordinate Gradient):** Gradient-based adversarial suffix optimization
- **PAIR (Prompt Automatic Iterative Refinement):** Automated iterative jailbreak refinement
- **TAP (Tree of Attacks with Pruning):** Tree-structured search over attack prompts
- **Crescendo:** Multi-turn gradual escalation across conversation turns
- **Skeleton Key:** Master key prompts disabling safety across topics

**Data and Model Poisoning** targets the training pipeline:
- Trigger-based backdoors that activate on specific inputs while maintaining normal behavior otherwise
- Bias injection favoring attacker interests
- Performance sabotage on targeted input classes
- Poisoning via compromised open-source model repositories (100+ compromised models found on Hugging Face) [Check Point AI Security Report 2025](https://blog.checkpoint.com/research/ai-security-report-2025-understanding-threats-and-building-smarter-defenses/)

**Model Extraction and Membership Inference** represent privacy and IP theft risks:
- Model extraction through systematic querying to replicate model behavior
- Membership inference attacks determine whether specific data was in training sets. The SPV-MIA method raises attack AUC from 0.7 to 0.9 for membership inference against fine-tuned LLMs [SoK: Membership Inference Attacks on LLMs (arXiv 2406.17975)](https://arxiv.org/abs/2406.17975)
- However, post-hoc evaluation setups for MIA suffer from distribution shift issues that may inflate reported effectiveness [SoK: Membership Inference Attacks on LLMs (arXiv 2406.17975)](https://arxiv.org/abs/2406.17975)

### 1.3 Defense and Mitigation Strategies

- **Input filtering and sanitization:** Strict input validation, semantic prompt validation to detect injection intent
- **Output guardrails:** Content safety classifiers, output sanitization before downstream processing
- **Adversarial training:** Fine-tuning on adversarial examples (HarmBench introduces efficient adversarial training that enhances robustness) [HarmBench (arXiv 2402.04249)](https://arxiv.org/abs/2402.04249)
- **Prompt hardening:** System prompt engineering with role enforcement and boundary instructions
- **Human-in-the-loop:** Required approvals for high-risk actions
- **Differential privacy:** Training with DP-SGD to limit memorization
- **Continuous red teaming:** Integrating adversarial testing into CI/CD pipelines

### 1.4 Evaluation Metrics and Benchmarks

| Benchmark | Scope | Scale | Key Contribution |
|-----------|-------|-------|-----------------|
| **HarmBench** | Red teaming + robust refusal | 510 behaviors, 18 methods, 33 LLMs | Standardized end-to-end evaluation pipeline [HarmBench](https://arxiv.org/abs/2402.04249) |
| **AdversariaLLM** | Jailbreak robustness | 12 attack algorithms, 7 datasets | Reproducibility and distributional evaluation [AdversariaLLM (arXiv 2511.04316)](https://arxiv.org/html/2511.04316v1) |
| **SAFE-LLM** | Reliability, safety, security | Unified metrics framework | Hallucination Rate, Safety Compliance Index, Jailbreak Success Rate, Prompt Injection Compromise Rate |
| **HELM** | Multi-dimensional evaluation | Safety, reliability, fairness | Stanford comprehensive benchmark suite |
| **TruthfulQA** | Truthfulness | 817 questions | Measures tendency to reproduce misconceptions |

**Confidence:** HIGH. These frameworks and benchmarks are well-established with multiple independent validations.

---

## 2. Agent Security Evaluation

### 2.1 Current State of the Field

AI agents -- systems with planning capabilities, tool access, memory, and autonomous decision-making -- represent a fundamentally different security paradigm from static LLMs. According to McKinsey research, 80% of organizations report encountering risky behaviors from AI agents, including improper data exposure and unauthorized system access [McKinsey - Agentic AI Security](https://www.mckinsey.com/capabilities/risk-and-resilience/our-insights/deploying-agentic-ai-with-safety-and-security-a-playbook-for-technology-leaders).

The **MAESTRO** (Multi-Agent Environment, Security, Threat, Risk, and Outcome) framework, developed by the Cloud Security Alliance, provides the first purpose-built threat modeling framework for agentic AI. It employs a layered approach that maps threats across the full agent architecture stack, addressing gaps that traditional frameworks like STRIDE and PASTA cannot cover [CSA MAESTRO Framework](https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro).

### 2.2 Key Attack Categories and Techniques

Agent-specific threats extend beyond traditional LLM vulnerabilities [Agentic AI Security: Threats, Defenses, Evaluation (arXiv 2510.23883)](https://arxiv.org/abs/2510.23883):

**Tool Misuse and Abuse**
- Agents exploiting tool access beyond intended scope
- Chaining multiple tools to achieve unauthorized objectives
- Confused deputy attacks where agents act with their own (often broad) privileges rather than the user's

**Memory Poisoning**
- Injecting false information into agent persistent memory
- Manipulating conversation history to alter future behavior
- Cross-session data contamination

**Goal Hijacking and Manipulation**
- Indirect prompt injection that redirects agent objectives
- Multi-step manipulation where each step appears benign but the chain achieves malicious outcomes
- Exploiting planning capabilities to craft attack strategies

**Privilege Escalation**
- Agents accessing systems beyond their authorization scope
- Lateral movement across connected services
- Exploiting over-provisioned credentials and API tokens

**Autonomous Action Risks**
- Unintended cascading effects from autonomous decisions
- Difficulty in attributing and reversing agent actions
- Race conditions in multi-agent systems

### 2.3 Defense and Mitigation Strategies

- **Zero-trust architecture for agents:** Treat agents as high-privilege identities requiring continuous verification [Obsidian Security](https://www.obsidiansecurity.com/blog/ai-agent-security-risks)
- **Just-in-time access:** Dynamic authorization that grants minimum required permissions for specific tasks
- **Behavioral monitoring:** Real-time anomaly detection for agent actions
- **Sandboxing:** Isolating agent execution environments with restricted system access
- **Action logging and auditability:** Complete audit trails of all agent decisions and tool invocations
- **Blast radius limitation:** Constraining the scope of potential damage from any single agent

### 2.4 Evaluation Frameworks

The survey by Chhabra et al. (2025) provides a taxonomy of agentic AI threats and reviews evaluation methodologies including:
- Benchmark environments that test agent behavior under adversarial conditions
- Multi-step evaluation chains testing tool use safety
- Simulation-based testing of autonomous decision-making
- Assessment of agent behavior when exposed to conflicting instructions [Agentic AI Security (arXiv 2510.23883)](https://arxiv.org/abs/2510.23883)

**Confidence:** MODERATE-HIGH. Agent security is a rapidly evolving field with solid conceptual frameworks but limited standardized benchmarks compared to model-level evaluation.

---

## 3. MCP (Model Context Protocol) Security

### 3.1 Current State of the Field

The Model Context Protocol, introduced by Anthropic in November 2024, standardizes how AI applications connect to external tools, data sources, and services. Often described as "USB-C for AI," MCP has achieved rapid adoption -- major providers including Microsoft, OpenAI, Google, and Amazon have integrated support. However, this rapid adoption has outpaced security hardening.

The OWASP MCP Security Cheat Sheet identifies MCP as introducing "a fundamentally new attack surface: AI agents dynamically executing tools based on natural language, with access to sensitive systems" [MCP Security - OWASP](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html).

Research by Knostic discovered 1,862 MCP servers exposed on the public internet; manual verification of 119 instances found that every single one permitted unauthenticated access to internal tool listings [CSA: Agentic Trust Deficit](https://cloudsecurityalliance.org/blog/2026/03/24/the-agentic-trust-deficit-why-mcp-s-authentication-vacuum-demands-a-new-security-paradigm). MCP vulnerabilities reportedly grew 270% between Q2 and Q3 2025 [Wallarm 2026 API ThreatStats Report, cited in SecurityBoulevard].

### 3.2 Key Attack Categories and Techniques

**Tool Poisoning Attacks (TPAs):** Discovered by Invariant Labs in April 2025, TPAs embed malicious instructions in MCP tool descriptions that are invisible to users but visible to AI models. These hidden instructions can direct models to access sensitive files, exfiltrate data, and override instructions from trusted servers [MCP Tool Poisoning - Invariant Labs](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks). The MCPTox benchmark, evaluating 20 LLM agents against 45 real-world MCP servers and 353 tools, found that o1-mini exhibited a 72.8% attack success rate, and more capable models were often more susceptible because the attack exploits superior instruction-following abilities [MCPTox (arXiv 2508.14925)](https://arxiv.org/html/2508.14925v1).

**Rug Pull Attacks:** A server changes its tool definitions after initial user approval, transforming a previously trusted tool into a malicious one. Without cryptographic hash pinning of tool definitions, users have no way to detect post-approval mutations [OWASP MCP Security](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html).

**Tool Shadowing / Cross-Origin Escalation:** A malicious MCP server's tool descriptions manipulate how an agent interacts with tools from other, trusted servers. Because the LLM sees all tool descriptions from all connected servers in its context, a single malicious server can compromise the entire agent workflow [OWASP MCP Security](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html), [Microsoft - Plug, Play, and Prey](https://techcommunity.microsoft.com/blog/microsoftdefendercloudblog/plug-play-and-prey-the-security-risks-of-the-model-context-protocol/4410829).

**Confused Deputy Problem:** MCP servers execute actions with their own (often broad) privileges rather than the requesting user's permissions, creating a classical confused deputy vulnerability [OWASP MCP Security](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html).

**Data Exfiltration via Legitimate Channels:** Attackers use prompt injection to encode sensitive data into seemingly normal tool calls (search queries, email subjects, URL parameters) [OWASP MCP Security](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html).

**EchoLeak (CVE-2025-32711, CVSS 9.3):** A zero-click exploit against production MCP-based AI systems, representing the first confirmed weaponized attack against production deployments [CSA: Agentic Trust Deficit](https://cloudsecurityalliance.org/blog/2026/03/24/the-agentic-trust-deficit-why-mcp-s-authentication-vacuum-demands-a-new-security-paradigm).

**Sandbox Escapes:** Local MCP servers running with full host access can enable file system traversal, credential theft, and arbitrary code execution [OWASP MCP Security](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html).

**Supply Chain Attacks:** Untrusted or compromised MCP server packages from public registries, including typosquatting attacks (e.g., `mcp-server-filesystem` vs `mcp-server-filesytem`) [OWASP MCP Security](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html).

### 3.3 Defense and Mitigation Strategies

The OWASP MCP Security Cheat Sheet provides 12 categories of best practices [OWASP MCP Security](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html):

1. **Principle of Least Privilege:** Scoped, per-server credentials; narrow OAuth scopes; ephemeral tokens
2. **Tool Description and Schema Integrity:** Inspect and pin tool definitions with cryptographic hashes; detect rug pulls
3. **Sandboxing:** Run MCP servers in containers with restricted filesystem and network access
4. **Human-in-the-Loop:** Explicit user confirmation for destructive, financial, or data-sharing operations with full parameter display
5. **Input/Output Validation:** Treat all LLM-generated inputs as untrusted; sanitize against injection; prevent SSRF
6. **Authentication and Transport Security:** OAuth 2.0 with PKCE; session binding; TLS enforcement; secure credential storage
7. **Message-Level Integrity:** RSA/ECDSA signing of JSON-RPC payloads; nonce and timestamp for replay prevention
8. **Multi-Server Isolation:** Treat each MCP server as an independent security domain
9. **Supply Chain Security:** Verified sources; code review; package integrity verification; typosquatting detection
10. **Monitoring and Logging:** SIEM integration; anomaly detection; full parameter logging
11. **Consent and Installation Security:** Clear consent dialogs; re-prompting on tool definition changes
12. **Prompt Injection via Return Values:** Treat tool responses as untrusted data; strip instruction-like patterns

The academic framework by Jamshidi et al. proposes three defense layers: RSA-based manifest signing for descriptor integrity, LLM-on-LLM semantic vetting for suspicious definitions, and heuristic guardrails for runtime anomaly detection. GPT-4 blocks approximately 71% of unsafe tool calls under this framework [Securing MCP (arXiv 2512.06556)](https://arxiv.org/abs/2512.06556).

### 3.4 Security Tools

- **mcp-scan** (Invariant Labs): Automated scanner for detecting tool poisoning, cross-server shadowing, and definition mutations [Invariant Labs](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks)
- **Docker MCP Toolkit:** Enterprise-grade containerized MCP execution with sandboxing [Docker Blog](https://www.docker.com/blog/mcp-security-issues-threatening-ai-infrastructure/)
- **Gopher Security:** Quantum-resistant MCP protection platform [SecurityBoulevard]

**Confidence:** HIGH for attack categories; MODERATE for specific statistics (some from single sources). The authentication vacuum in MCP is universally acknowledged across OWASP, Microsoft, Invariant Labs, CSA, and academic research.

---

## 4. A2A (Agent-to-Agent) Protocol Security

### 4.1 Current State of the Field

Google launched the Agent2Agent (A2A) protocol in April 2025 with support from 50+ technology partners including Atlassian, Salesforce, SAP, and ServiceNow [Google Developers Blog - A2A Announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/). A2A enables independent AI agents to discover each other's capabilities, communicate, and collaborate on complex tasks regardless of their underlying implementation framework. As of early 2026, the A2A GitHub repository has 23,000+ stars and reached version 1.0.0.

A2A complements MCP: while MCP connects agents to tools, A2A connects agents to other agents. Together, they form the communication backbone for multi-agent AI systems.

### 4.2 Security Architecture

A2A's security model is built on standard web security practices [A2A Specification](https://github.com/a2aproject/A2A), [A2A Security and Authentication](https://deepwiki.com/google-a2a/A2A/2.10-security-and-authentication):

**Transport Security:**
- TLS 1.2+ is mandatory for all production deployments
- Valid, trusted certificates required (no self-signed in production)
- Perfect forward secrecy (PFS) cipher suites recommended

**Authentication Schemes (declared via Agent Cards):**
- **Bearer Token:** JWTs or API keys
- **OAuth 2.0:** Industry-standard authorization with various grant types
- **API Key:** Static keys in headers or query parameters
- **Mutual TLS (mTLS):** Certificate-based bidirectional authentication

**Key design principle:** Credentials are transmitted at the HTTP transport layer, never in JSON-RPC payloads or A2A message content.

**Authorization:** Implementation-specific, with mandatory scoping:
- Tasks scoped to the authenticated client
- List operations return only authorized tasks
- Agents must not distinguish between "does not exist" and "not authorized" (prevents enumeration)
- Per-skill authorization supported for fine-grained access control

**Extended Agent Cards:** Progressive disclosure mechanism where agents expose different capabilities based on client authentication level.

**Push Notification Security:** Bidirectional authentication required -- agents authenticate to client webhooks (JWT, HMAC, mTLS) and clients verify incoming notifications with replay protection (timestamps, nonces, event sequencing).

### 4.3 Key Attack Categories and Techniques

Security analysis by Palo Alto Networks, CSA, and Keysight identifies the following A2A-specific threats [Palo Alto - A2A Security Guide](https://live.paloaltonetworks.com/t5/community-blogs/safeguarding-ai-agents-an-in-depth-look-at-a2a-protocol-risks/ba-p/1235996), [CSA: Threat Modeling A2A](https://cloudsecurityalliance.org/blog/2025/04/30/threat-modeling-google-s-a2a-protocol-with-the-maestro-framework):

**Agent Card Manipulation:**
- **Context poisoning:** Malicious content embedded in Agent Card descriptions that compromises downstream agents when processed by LLMs
- **Version control attacks:** Exploiting delayed Agent Card updates to maintain compromised configurations
- **Metadata exposure:** Sensitive information leaked through overly detailed Agent Card fields

**Agent Impersonation and Shadowing:**
- Cloning or mimicking legitimate agents to infiltrate multi-agent workflows
- Exploiting flawed identity verification in Agent Card discovery
- Agent Cards at `/.well-known/agent.json` can be spoofed if DNS or hosting is compromised

**Task Execution Integrity:**
- Crafting deceptive responses that corrupt downstream agent processing
- Manipulating task state transitions to trigger unintended behaviors
- Injecting malicious content in artifacts returned as task outputs

**Infrastructure Attacks:**
- Resource exhaustion through task flooding
- SSRF via webhook URLs in push notification configurations (agents must validate webhook URLs to prevent targeting internal services)
- Man-in-the-middle attacks on improperly configured TLS

**Application Logic Attacks:**
- Prompt injection through A2A message content that manipulates the receiving agent's LLM
- Data exfiltration through legitimate-looking task artifacts
- Privilege escalation by chaining capabilities across multiple agents

### 4.4 Defense and Mitigation Strategies

The paper "Building A Secure Agentic AI Application Leveraging A2A Protocol" (Habler et al., 2025) recommends applying the MAESTRO framework to A2A deployments [Building Secure A2A (arXiv 2504.16902)](https://arxiv.org/abs/2504.16902):

- **Agent Card signatures:** Cryptographic JWS signatures to ensure integrity and authenticity
- **Strict authentication enforcement:** OAuth 2.0 or mTLS for all production inter-agent communication
- **Zero-trust agent identity:** Continuous verification, no trust based on network location alone
- **Input sanitization:** Validate all RPC parameters, message content, file uploads, and URLs
- **Webhook URL validation:** Allowlisting, ownership verification, HTTPS-only
- **Key rotation:** Regular credential rotation using secrets managers
- **Agent registration and governance:** Centralized inventory of authorized agents with Cisco Duo IAM or similar [Cisco - AI Agent Security](https://siliconangle.com/2026/03/23/cisco-debuts-new-ai-agent-security-features-open-source-defenseclaw-tool/)

### 4.5 Security Tools

- **Cisco A2A Scanner:** Combines static analysis, runtime monitoring, YARA rules, spec validation, heuristic analysis, and LLM-powered semantic detection for A2A protocol implementations [Cisco A2A Scanner](https://github.com/cisco-ai-defense/a2a-scanner)
- **Auth0 for AI Agents:** A2A-compatible authentication infrastructure with delegation support [Auth0 - A2A Authentication](https://auth0.com/blog/auth0-google-a2a/)

**Confidence:** MODERATE. A2A is a young protocol (launched April 2025, v1.0.0 released early 2026). Security analysis is primarily from vendor research and one academic paper. The protocol specification itself has a robust security model, but real-world implementation vulnerabilities are likely to emerge as adoption scales.

---

## 5. Red Teaming AI Systems

### 5.1 Current State of the Field

AI red teaming has emerged as the primary methodology for proactively identifying vulnerabilities in AI systems. It differs fundamentally from traditional penetration testing: the attack surface is probabilistic, vulnerabilities are behavior-based rather than code-based, and the same input can produce different outputs across runs [Repello AI - Red Teaming Guide](https://repello.ai/blog/ai-red-teaming).

The market for AI red teaming services reached an estimated $1.43 billion in 2024 and is projected to grow to $4.8 billion by 2029 [Vectra AI](https://www.vectra.ai/topics/ai-red-teaming). Regulatory pressure is a key driver: the EU AI Act mandates adversarial testing for high-risk AI systems by August 2026, with penalties up to 35 million euros or 7% of global annual revenue.

### 5.2 Frameworks and Standards

**MITRE ATLAS** provides the canonical taxonomy for AI red teaming, enabling:
- Coverage measurement: "We tested 45 of 70 ATLAS-mapped techniques"
- Gap analysis: Identifying untested attack vectors
- Cross-team comparability: Standardized vocabulary across engagements
- Compliance mapping to OWASP, NIST, and EU AI Act requirements
[MITRE ATLAS](https://atlas.mitre.org/)

**NIST AI RMF** structures red teaming within the Manage function, providing governance context for adversarial testing activities. The Generative AI Profile (NIST-AI-600-1) specifically addresses generative AI risks [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).

**OWASP GenAI Red Teaming Guide** provides operational guidance for testing LLM applications against the OWASP LLM Top 10.

**CSA Agentic AI Red Teaming** guidance addresses the unique challenges of testing autonomous agent systems.

### 5.3 Methodologies

A structured AI red teaming exercise typically follows five phases:

1. **Scoping and Threat Modeling:** Define the system under test, identify applicable ATLAS techniques, establish rules of engagement
2. **Attack Surface Mapping:** Catalog inputs (prompts, context, tools), outputs, connected systems, and data flows
3. **Attack Generation:** Create adversarial inputs targeting identified vulnerabilities using both static datasets and dynamic generation
4. **Execution and Evaluation:** Run attacks against the system, evaluate responses using automated classifiers (LLM-as-judge) and human review
5. **Reporting and Remediation:** Document findings mapped to ATLAS/OWASP taxonomies, provide severity ratings and remediation guidance

**Manual vs. Automated Red Teaming:** Analysis of the Crucible dataset (214,271 attempts, 1,674 users, 30 challenges) demonstrates that automated approaches significantly outperform manual techniques (69.5% vs 47.6% success rate), though only 5.2% of users employed automation. Automated approaches excel in systematic exploration and pattern matching, while manual approaches retain advantages in creative reasoning scenarios, sometimes solving problems 5.2x faster when successful [Automation Advantage (arXiv 2504.19855)](https://arxiv.org/html/2504.19855v2).

**Optimal approach:** Combine human creativity for strategy development with programmatic execution for thorough exploration.

### 5.4 Finding Novel Attacks

Techniques for discovering new attack vectors:

- **Gradient-based optimization:** GCG and AutoDAN use gradient information to craft adversarial suffixes
- **LLM-vs-LLM:** Using attacker LLMs to iteratively refine jailbreak prompts (PAIR, TAP)
- **Genetic algorithms:** Evolving attack prompts through mutation and selection
- **Multi-turn escalation:** Crescendo-style attacks that gradually shift model behavior across conversation turns
- **Cross-modality attacks:** Embedding adversarial content in images, audio, or documents processed by multi-modal models
- **Context window exploitation:** Overflow attacks that dilute safety instructions through lengthy inputs
- **Encoding attacks:** Base64, ROT13, and other encoding schemes to bypass text-based filters

### 5.5 Key Metrics

| Metric | Description |
|--------|-------------|
| **Attack Success Rate (ASR)** | Percentage of adversarial inputs that elicit harmful outputs |
| **Jailbreak Success Rate (JSR)** | Subset of ASR targeting safety guardrail bypasses |
| **Refusal Rate** | Percentage of adversarial inputs properly refused |
| **False Positive Rate** | Rate of benign inputs incorrectly flagged as harmful |
| **ATLAS Coverage** | Percentage of applicable ATLAS techniques tested |
| **Time to First Success** | How quickly an attack vector produces results |

**Confidence:** HIGH. Red teaming methodology is well-established with extensive academic and industry support.

---

## 6. Open Source Tools for AI Security Testing

### 6.1 Comprehensive Tool Inventory

| Tool | Maintainer | Focus | Stars | Key Capabilities |
|------|-----------|-------|-------|-----------------|
| **[PyRIT](https://github.com/Azure/PyRIT)** | Microsoft | Red teaming | ~8k | Multi-modal, multi-turn (Crescendo, TAP, Skeleton Key), any-target interface, CoPyRIT GUI [PyRIT (arXiv 2410.02828)](https://arxiv.org/html/2410.02828v1) |
| **[Garak](https://github.com/NVIDIA/garak)** | NVIDIA | Vulnerability scanning | 7.5k | Static, dynamic, adaptive probes; hallucination, data leakage, prompt injection, toxicity [Garak](https://garak.ai/) |
| **[Promptfoo](https://github.com/promptfoo/promptfoo)** | OpenAI (acq. 2026) | Red teaming + evaluation | 19.5k | 50+ vulnerability types, 67+ attack plugins, CI/CD integration, LLM-as-judge [Promptfoo](https://www.promptfoo.dev/) |
| **[ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox)** | IBM / LF AI | ML security | 5.9k | 39 attack modules, 29 defense modules; evasion, poisoning, extraction, inference [ART](https://adversarial-robustness-toolbox.org/) |
| **[HarmBench](https://github.com/centerforaisafety/HarmBench)** | CAIS | Benchmarking | ~900 | 510 behaviors, 18 methods, standardized evaluation pipeline [HarmBench (arXiv 2402.04249)](https://arxiv.org/abs/2402.04249) |
| **[AdversariaLLM](https://arxiv.org/html/2511.04316v1)** | TU Munich | Research | New | 12 attack algorithms, 7 datasets, JudgeZoo companion for evaluation [AdversariaLLM](https://arxiv.org/html/2511.04316v1) |
| **[mcp-scan](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks)** | Invariant Labs | MCP security | N/A | Tool poisoning detection, cross-server shadowing, definition mutation monitoring |
| **[A2A Scanner](https://github.com/cisco-ai-defense/a2a-scanner)** | Cisco | A2A security | 139 | YARA rules, spec validation, heuristics, LLM-powered semantic detection, endpoint testing |
| **[DefenseClaw](https://siliconangle.com/2026/03/23/cisco-debuts-new-ai-agent-security-features-open-source-defenseclaw-tool/)** | Cisco | Agent security | New | AI agent vulnerability scanning |
| **[MCPTox](https://arxiv.org/html/2508.14925v1)** | Academic | MCP benchmarking | N/A | 45 real-world MCP servers, 353 tools, 1,312 test cases across 10 risk categories |

### 6.2 Detailed Tool Profiles

**PyRIT (Python Risk Identification Toolkit)** -- Microsoft's open-source framework is the most comprehensive tool for AI red teaming. Key features include:
- Model- and platform-agnostic architecture supporting OpenAI, Azure, Anthropic, Google, HuggingFace, and custom endpoints
- Multi-turn attack strategies (Crescendo, TAP, Skeleton Key, PAIR) with orchestrator patterns
- CoPyRIT graphical interface for human-led red teaming
- Built-in memory system (SQLite or Azure SQL) for tracking conversations and results
- Flexible scoring with LLM-graded evaluations, Azure AI Content Safety integration, and custom scorers
[PyRIT Documentation](https://microsoft.github.io/PyRIT/)

**Garak (Generative AI Red-teaming and Assessment Kit)** -- NVIDIA's vulnerability scanner operates analogously to nmap or Metasploit but for LLMs:
- Probes for hallucination, data leakage, prompt injection, misinformation, toxicity, and jailbreaks
- Static, dynamic, and adaptive probe generation
- Extensible plugin architecture for custom probe development
- DEF CON-presented research tool with active community
[Garak](https://garak.ai/)

**Promptfoo** -- acquired by OpenAI for approximately $86 million in March 2026, now the default open-source choice for AI red teaming:
- 50+ vulnerability types including prompt injection, PII leakage, RBAC bypass, unauthorized tool execution
- Three-component architecture: plugins (what to attack), strategies (how to attack), LLM-as-judge (grading)
- Supports OWASP LLM Top 10, NIST AI RMF, and EU AI Act compliance presets
- Runs fully locally for privacy-sensitive environments
- CI/CD integration for continuous security regression testing
[Promptfoo](https://www.promptfoo.dev/docs/red-team/)

**Adversarial Robustness Toolbox (ART)** -- the most mature ML security library, maintained under the Linux Foundation and partially funded by DARPA:
- Four threat categories: Evasion, Poisoning, Extraction, Inference
- Supports TensorFlow, PyTorch, scikit-learn, XGBoost, and more
- 39 attack implementations and 29 defense implementations
- Both red team (attack) and blue team (defense) capabilities
- Available on Hugging Face for direct model evaluation
[ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

### 6.3 Additional Notable Tools

- **LLM-Guard** (Protect AI): Input/output guardrails for LLM applications
- **Rebuff:** Self-hardening prompt injection detector
- **NeMo Guardrails** (NVIDIA): Programmable guardrails for LLM applications
- **Guardrails AI:** Open-source framework for adding safety rails to LLM outputs
- **PromptBench:** Benchmark for evaluating LLM robustness to adversarial prompts
- **AI Red Teaming Index** (Alpha One): Structured dataset tracking tools, frameworks, benchmarks, and vulnerability leaderboards

**Confidence:** HIGH. Tool information verified directly from GitHub repositories and official documentation.

---

## 7. Additional Security Considerations

### 7.1 Supply Chain Attacks

AI supply chain attacks have emerged as a critical threat vector. Malicious package uploads to open-source repositories jumped 156% in the past year, and AI-generated malware is polymorphic by default, context-aware, and temporally evasive [CISO's Guide to AI Supply Chain Attacks - Hacker News](https://thehackernews.com/2025/11/cisos-expert-guide-to-ai-supply-chain.html).

**The five AI supply chain attack surfaces** [AI Supply Chain Attacks - Repello AI](https://repello.ai/blog/ai-supply-chain-attacks):

1. **Package and library dependencies:** The most active surface in 2026. In Q1 2026, attackers compromised a PyPI contributor to LiteLLM and published malicious versions with credential-harvesting backdoors
2. **Model weights:** Backdoored models that activate on specific triggers while maintaining normal behavior on standard benchmarks. Currently, there are no strong provenance assurances for published models [LLM Supply Chain Security - TestMy.AI](https://testmy.ai/blog/llm-supply-chain-security-third-party-models)
3. **Training data:** Poisoned data sources that persist through fine-tuning cycles
4. **SaaS-embedded AI:** Third-party AI features with opaque security postures
5. **MCP and plugin ecosystems:** As described in Section 3

**Key defense:** Software Bill of Materials (SBOM) for AI components, model provenance verification, dependency scanning, and secure MLOps pipelines.

### 7.2 RAG Security

Retrieval-Augmented Generation introduces unique vulnerabilities at the intersection of information retrieval and language generation:

**Attack vectors:**
- **Knowledge base poisoning:** Injecting malicious documents into the retrieval corpus that contain prompt injection payloads
- **Vector embedding manipulation:** Crafting adversarial documents that achieve high similarity scores for targeted queries
- **Retrieval manipulation:** Steering the retrieval component to surface attacker-controlled content
- **Context window overflow:** Flooding retrieved context to dilute system instructions

**Defenses:**
- Content validation and sanitization for ingested documents
- Access control on retrieval results based on user permissions
- Anomaly detection for unusual retrieval patterns
- Separation of instruction context from retrieved data context

OWASP LLM08 (Vector and Embedding Weaknesses) specifically addresses these risks [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf).

### 7.3 Fine-Tuning Risks

Fine-tuning introduces security concerns at multiple levels:
- **Safety degradation:** Fine-tuning can erode base model safety training, making models more susceptible to jailbreaks
- **Backdoor insertion:** Malicious fine-tuning datasets can introduce hidden behaviors
- **LoRA-specific attacks:** LoRA-Leak demonstrates that LoRA fine-tuning can exacerbate privacy leakage [GitHub: Awesome-LM-SSP](https://github.com/CryptoAILab/Awesome-LM-SSP)
- **Unauthorized fine-tuning:** API-accessible fine-tuning endpoints may be exploited to modify model behavior

### 7.4 API and Inference Infrastructure Security

- **API key management:** Preventing exposure of LLM API credentials in client-side code or logs
- **Rate limiting and quotas:** Defending against model denial-of-service and cost exploitation
- **Inference endpoint hardening:** Securing model serving infrastructure against traditional web attacks
- **Side-channel attacks:** Timing analysis and resource consumption patterns that leak information about model internals
- **ShadowRay case study:** CVE-2023-48022 exploited lack of authorization in the Ray Jobs API (used for AI workloads), enabling adversaries to access user tokens, SSH keys, database credentials, and cloud compute resources, with estimated financial impact exceeding $1 billion [MITRE ATLAS](https://atlas.mitre.org/)

### 7.5 Multimodal Attacks

As LLMs gain vision, audio, and video capabilities, new attack surfaces emerge:
- **Adversarial images:** Imperceptible perturbations to images that cause misclassification or trigger hidden behaviors
- **Text-in-image injection:** Embedding prompt injection text within images processed by vision models
- **Audio adversarial examples:** Modified audio that is inaudible to humans but interpreted as commands by speech models
- **Cross-modal attacks:** Leveraging one modality to attack another (e.g., image captions to inject prompts)

HarmBench includes multimodal behavior categories and supports evaluation across text and image-based attacks [HarmBench (arXiv 2402.04249)](https://arxiv.org/abs/2402.04249).

### 7.6 Agentic Memory Attacks

Agents with persistent memory are vulnerable to:
- **Memory injection:** Inserting false information into agent memory stores
- **Memory extraction:** Retrieving sensitive information from past interactions stored in memory
- **Cross-session contamination:** Corrupting memory to affect future agent behavior
- **Memory poisoning via tool outputs:** Malicious tool responses that get stored in agent memory

### 7.7 Data Leakage

Check Point research found that 1 in every 80 GenAI prompts poses a high risk of sensitive data leakage, and 7.5% of all prompts (approximately 1 in 13) contain potentially sensitive information [Check Point AI Security Report 2025](https://blog.checkpoint.com/research/ai-security-report-2025-understanding-threats-and-building-smarter-defenses/).

**Confidence:** HIGH for established categories (supply chain, RAG, API security); MODERATE for emerging areas (multimodal, memory attacks).

---

## Verification Summary

| Metric | Value |
|--------|-------|
| Total Sources | 35 retrieved |
| Source Types | Academic (10), Standards/Government (5), Vendor Research (8), Tools/Repos (7), Industry Analyst (2), Blogs (3) |
| Atomic Claims Verified | 18 key claims scored |
| SUPPORTED (2+ sources) | 12 claims (67%) |
| WEAK (1 source) | 6 claims (33%) -- flagged with caveats |
| REMOVED (unsupported) | 0 |
| Verification Methods | SIFT (applied to vendor content), Chain-of-Verification, Cross-source triangulation |

---

## Conclusion

AI security in 2024-2026 is characterized by a fundamental tension: the attack surface is expanding faster than defenses can mature. Three dynamics drive this:

**First, the shift from models to agents to multi-agent systems multiplies risk non-linearly.** Each layer (model, agent, protocol) introduces new attack surfaces that interact in complex ways. A prompt injection that is merely an annoyance against a standalone chatbot becomes a critical vulnerability when that chatbot is an agent with tool access, connected to other agents via A2A and MCP. The MAESTRO framework's layered approach correctly recognizes this, but adoption remains early.

**Second, the protocol layer (MCP and A2A) is the most under-secured frontier.** MCP's authentication vacuum -- with 100% of sampled exposed servers allowing unauthenticated access -- represents the kind of gap that characterized early web security. The EchoLeak exploit (CVE-2025-32711) demonstrates this is not theoretical. While the A2A specification has a more robust security model by design, its youth means implementation-level vulnerabilities have not yet been widely discovered or catalogued.

**Third, the democratization of both attack and defense tools is accelerating.** The open-source ecosystem (PyRIT, Garak, Promptfoo, ART, HarmBench) provides powerful red teaming capabilities, but the same techniques are available to adversaries. The automation advantage in red teaming (69.5% vs 47.6% success rate) cuts both ways.

**What organizations should do:**
1. **Adopt the OWASP LLM Top 10 as baseline security requirements** for any AI deployment
2. **Map threats using MITRE ATLAS** to ensure systematic coverage of known attack vectors
3. **Implement continuous automated red teaming** using tools like PyRIT, Garak, or Promptfoo in CI/CD pipelines
4. **Apply zero-trust principles to AI agents** -- treat them as high-privilege identities requiring continuous verification
5. **Secure MCP deployments immediately** -- the authentication gap is the most urgent vulnerability; use mcp-scan, enforce authentication, sandbox servers, pin tool definitions
6. **Prepare for EU AI Act compliance** -- adversarial testing requirements take effect August 2026
7. **Establish AI-specific supply chain security** -- model provenance verification, dependency scanning, SBOM for AI components

The field is maturing rapidly, but the gap between attack capability and deployed defenses remains significant. Organizations deploying AI systems without structured security evaluation are accepting risks that are increasingly well-documented and increasingly exploited.

---

## Limitations and Residual Risks

- **A2A security research is limited** by the protocol's youth (launched April 2025). Implementation-level vulnerabilities may emerge as adoption scales that are not covered in current analysis.
- **MCP vulnerability statistics** (270% growth, 1,862 exposed servers) derive from limited vendor research and may not be representative of the full deployment landscape.
- **Red teaming effectiveness metrics** (69.5% automated success rate) come from a single study (Crucible dataset) and may not generalize to all AI system types.
- **Rapidly evolving field** -- specific tool capabilities and framework versions described may change significantly within months of this report. All information reflects the state of the field as of early April 2026.
- **Pre-mortem risk:** This research could be wrong if a fundamental architectural change in LLM design eliminates current attack surface categories (unlikely in the near term), or if currently nascent A2A/MCP security tools mature faster than expected (possible but not yet evident).

---

## Sources

1. [OWASP Top 10 for LLM Applications 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf) - OWASP, Nov 2024
2. [MITRE ATLAS](https://atlas.mitre.org/) - MITRE Corporation, continuously updated
3. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) - NIST, Jan 2023 (GenAI Profile Jul 2024)
4. [MCP Security - OWASP Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html) - OWASP, 2025
5. [Agentic AI Security: Threats, Defenses, Evaluation, and Open Challenges](https://arxiv.org/abs/2510.23883) - Chhabra et al., Oct 2025
6. [HarmBench: A Standardized Evaluation Framework](https://arxiv.org/abs/2402.04249) - Mazeika et al., Feb 2024
7. [PyRIT: A Framework for Security Risk Identification and Red Teaming](https://arxiv.org/html/2410.02828v1) - Lopez Munoz et al., Oct 2024
8. [NVIDIA Garak LLM Vulnerability Scanner](https://github.com/NVIDIA/garak) - NVIDIA, continuously updated
9. [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - IBM / LF AI, continuously updated
10. [Promptfoo LLM Red Teaming](https://github.com/promptfoo/promptfoo) - OpenAI (acquired Mar 2026), continuously updated
11. [Google A2A Protocol Specification](https://github.com/a2aproject/A2A) - Google, Apr 2025
12. [A2A Security and Authentication](https://deepwiki.com/google-a2a/A2A/2.10-security-and-authentication) - DeepWiki, derived from A2A spec
13. [Building A Secure Agentic AI Application Leveraging A2A Protocol](https://arxiv.org/abs/2504.16902) - Habler et al., Apr 2025
14. [Securing the Model Context Protocol](https://arxiv.org/abs/2512.06556) - Jamshidi et al., Dec 2025
15. [MCP Security Notification: Tool Poisoning Attacks](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks) - Invariant Labs, Apr 2025
16. [MCP Security Issues Threatening AI Infrastructure](https://www.docker.com/blog/mcp-security-issues-threatening-ai-infrastructure/) - Docker, Jul 2025
17. [Plug, Play, and Prey: MCP Security Risks](https://techcommunity.microsoft.com/blog/microsoftdefendercloudblog/plug-play-and-prey-the-security-risks-of-the-model-context-protocol/4410829) - Microsoft, May 2025
18. [MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers](https://arxiv.org/html/2508.14925v1) - Wang et al., Aug 2025
19. [Agentic AI Threat Modeling Framework: MAESTRO](https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro) - Cloud Security Alliance, Feb 2025
20. [The Agentic Trust Deficit: MCP's Authentication Vacuum](https://cloudsecurityalliance.org/blog/2026/03/24/the-agentic-trust-deficit-why-mcp-s-authentication-vacuum-demands-a-new-security-paradigm) - CSA, Mar 2026
21. [Safeguarding AI Agents: A2A Protocol Risks and Mitigations](https://live.paloaltonetworks.com/t5/community-blogs/safeguarding-ai-agents-an-in-depth-look-at-a2a-protocol-risks/ba-p/1235996) - Palo Alto Networks, Aug 2025
22. [Potential Attack Surfaces in Agent2Agent (A2A) Protocol](https://www.keysight.com/blogs/en/tech/nwvs/2025/05/28/potential-attack-surfaces-in-a2a) - Keysight, May 2025
23. [Threat Modeling Google's A2A Protocol with MAESTRO](https://cloudsecurityalliance.org/blog/2025/04/30/threat-modeling-google-s-a2a-protocol-with-the-maestro-framework) - CSA, Apr 2025
24. [Cisco A2A Scanner](https://github.com/cisco-ai-defense/a2a-scanner) - Cisco AI Defense, 2025
25. [MITRE ATLAS: The Complete Guide to AI Security Threat Intelligence](https://www.vectra.ai/topics/mitre-atlas) - Vectra AI, 2025
26. [The Automation Advantage in AI Red Teaming](https://arxiv.org/html/2504.19855v2) - Mulla et al., Apr 2025
27. [AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research](https://arxiv.org/html/2511.04316v1) - Beyer et al., Nov 2025
28. [SoK: Membership Inference Attacks on LLMs are Rushing Nowhere](https://arxiv.org/abs/2406.17975) - Meeus et al., Jun 2024
29. [Cisco State of AI Security Report 2025](https://blogs.cisco.com/ai/cisco-introduces-the-state-of-ai-security-report-for-2025) - Cisco, Mar 2025
30. [AI Security Report 2025](https://blog.checkpoint.com/research/ai-security-report-2025-understanding-threats-and-building-smarter-defenses/) - Check Point Research, Apr 2025
31. [Deploying Agentic AI with Safety and Security](https://www.mckinsey.com/capabilities/risk-and-resilience/our-insights/deploying-agentic-ai-with-safety-and-security-a-playbook-for-technology-leaders) - McKinsey, 2025
32. [AI Supply Chain Attacks: The Complete Guide](https://repello.ai/blog/ai-supply-chain-attacks) - Repello AI, Apr 2026
33. [LLM Supply Chain Security 2025](https://testmy.ai/blog/llm-supply-chain-security-third-party-models) - TestMy.AI, Sep 2025
34. [How to Enhance Agent2Agent (A2A) Security](https://developers.redhat.com/articles/2025/08/19/how-enhance-agent2agent-security) - Red Hat, Aug 2025
35. [Secure A2A Authentication with Auth0 and Google Cloud](https://auth0.com/blog/auth0-google-a2a/) - Auth0, May 2025
