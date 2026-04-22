# AI Firewalls: Security for Models, Agents, and Tool-Calling Workflows

## Executive Summary

AI firewalls are an emerging class of security solutions purpose-built to protect AI systems -- large language models (LLMs), AI agents, and their tool-calling workflows -- from threats that traditional network firewalls and web application firewalls (WAFs) cannot detect. They operate at the **application and semantic layer**, inspecting the content of prompts, API inputs, and model-generated outputs rather than packet headers and IP addresses. AI firewalls produce real-time **pass/fail (allow/block/modify) verdicts** on every interaction with an AI model, enforcing policies against prompt injection, data leakage, jailbreaks, toxic content, and unauthorized tool use.

The category is consolidating rapidly. Robust Intelligence (acquired by Cisco in October 2024 for its "industry's first AI Firewall") now powers Cisco AI Defense [Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/robust-intelligence-is-part-of-cisco/index.html). F5 acquired CalypsoAI in September 2025 for $180M to add AI guardrails to its application delivery platform [F5](https://www.f5.com/company/news/press-releases/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises). Lakera, Protect AI (LLM Guard), NVIDIA NeMo Guardrails, CrowdStrike Falcon AIDR, and Cloudflare Firewall for AI round out the competitive landscape. Analyst frameworks from Gartner (AI TRiSM), Forrester (AEGIS), OWASP (LLM Top 10), MITRE (ATLAS), and NIST (AI RMF) all provide the threat models and governance structures that AI firewalls are designed to enforce.

**Confidence: HIGH** -- Definition, architecture, and vendor landscape are well-documented across 20+ independent and vendor sources. Standards alignment is confirmed by primary framework documentation.

---

## 1. What an AI Firewall Is

### 1.1 Definition

An AI firewall is a security solution that monitors and controls interactions between artificial intelligence systems and their environment. Unlike traditional firewalls, which filter network traffic based on IP addresses, ports, and protocols, AI firewalls inspect the data that flows into and out of AI models, including text prompts, API calls, and generated outputs [Radware](https://www.radware.com/cyberpedia/ai-firewall/). The main focus is guarding against attacks and misuse specific to AI workflows: prompt injection, data poisoning, model exploitation, and output manipulation [Radware](https://www.radware.com/cyberpedia/ai-firewall/).

Securiti describes an LLM firewall as being "placed at different instances of LLM interactions, such as prompts and responses," distinguishing it from traditional firewalls that "monitor network traffic" [Securiti](https://securiti.ai/what-is-llm-firewall/). Cloudflare frames its Firewall for AI as "an advanced Web Application Firewall (WAF) specifically tailored for applications using LLMs" that can "be deployed in front of applications to detect vulnerabilities and provide visibility to model owners" [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/).

### 1.2 How It Differs from Traditional Firewalls and WAFs

The fundamental differences span four dimensions:

| Dimension | Traditional Firewall / WAF | AI Firewall |
|-----------|---------------------------|-------------|
| **OSI Layer** | Network/transport layer (L3-L4) or application layer (L7) for HTTP | Application and **semantic layer** -- analyzes meaning, intent, and context of natural language |
| **What It Inspects** | Packet headers, IP addresses, ports, HTTP request patterns, SQL injection signatures | Prompt content, API inputs, model-generated outputs, retrieval-augmented generation (RAG) data, tool-call parameters |
| **Threat Model** | DDoS, SQLi, XSS, credential stuffing, known CVEs | Prompt injection, jailbreaks, PII leakage, toxic output, hallucination, data poisoning, model extraction, excessive agency |
| **Adaptiveness** | Primarily signature-based with some behavioral analysis | Behavioral baselining, semantic classifiers, ML-based anomaly detection, integration with AI-specific threat intelligence |

Sources: [Radware](https://www.radware.com/cyberpedia/ai-firewall/), [Securiti](https://securiti.ai/what-is-llm-firewall/), [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/)

A key insight from Cloudflare's analysis: traditional applications are **deterministic** -- a bank app accepts defined operations like `GET /balance` or `POST /transfer`, and security can focus on controlling those endpoints. LLM operations are **non-deterministic by design** -- interactions are based on natural language, making it much harder to identify problematic requests via simple signature matching [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/). Furthermore, in traditional applications the control plane (code) is well-separated from the data plane (database), while in LLMs the training data becomes part of the model itself, making data exposure control fundamentally more difficult [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/).

### 1.3 What Kinds of Rules and Policies It Enforces

AI firewalls enforce policies across several categories:

- **Prompt injection detection**: Identifying and blocking crafted inputs designed to override system instructions or manipulate model behavior [Radware](https://www.radware.com/cyberpedia/ai-firewall/), [Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/ai-runtime/index.html)
- **Jailbreak blocking**: Preventing attempts to bypass safety filters (e.g., the "DAN" technique) [Securiti](https://securiti.ai/what-is-llm-firewall/), [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/)
- **PII redaction / data loss prevention (DLP)**: Scanning prompts and responses for personally identifiable information, credit card numbers, API keys, and other sensitive data [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/), [Lakera](https://www.lakera.ai/)
- **Output filtering**: Blocking toxic, biased, offensive, or off-topic model outputs [Radware](https://www.radware.com/cyberpedia/ai-firewall/), [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/)
- **Topic control**: Preventing the model from engaging on prohibited subjects (politics, competitors, religion, etc.) [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails), [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/)
- **Tool-call governance**: Controlling which tools/APIs an AI agent can invoke and under what conditions [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/)
- **Model endpoint allow/deny**: Restricting which AI models users or applications can access [Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/ai-runtime/index.html)
- **Rate limiting and abuse prevention**: Throttling request volumes to prevent denial-of-service attacks against models [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/)

---

## 2. Scope of Protection Across Three Layers

### 2.1 Model-Level: Inference Endpoint Protection

At the model level, AI firewalls function as inline proxies that sit between users/applications and the model inference endpoint. They inspect every prompt before it reaches the model and every response before it is returned to the user.

**Cisco AI Defense (AI Runtime)** "inspects every input and automatically blocks malicious payloads before they can cause damage" including prompt injection, prompt extraction, DoS, and command execution. On the output side, it "scans model outputs to ensure there is no sensitive information, hallucinations, or other harmful content. Responses that fall outside an organization's standards will be blocked" [Cisco AI Runtime](https://www.cisco.com/site/us/en/products/security/ai-defense/ai-runtime/index.html).

**Cloudflare Firewall for AI** operates identically to a traditional WAF deployment: it is placed in front of the LLM application and "scans every request to identify attack signatures." It assigns a **maliciousness score from 1 to 99** (1 = most likely malicious) and tags prompts with content categories. Customers can create WAF rules to block or handle requests based on score thresholds, combinable with other signals like bot score or attack score [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/).

**Protect AI's LLM Guard** operates as a code-level firewall with two layers: **Input Scanners** (pre-process prompts to anonymize PII, ban topics/substrings, detect injections) and **Output Scanners** (post-process responses for compliance, content policy, factual consistency). "Before any prompt ever reaches the model, it passes through LLM Guard's Input Controls. If the input passes all checks, it's forwarded onward... The raw model response comes back into LLM Guard first. Once the output is deemed safe, it's returned to your application" [LLM Guard GitHub](https://github.com/protectai/llm-guard), [Protect AI](https://protectai.com/llm-guard).

### 2.2 Agent-Level: Runtime Guardrails and Action Approval

As AI agents become more autonomous -- planning, reasoning, and executing multi-step tasks -- the security surface expands beyond simple prompt/response pairs. AI firewalls at the agent level provide:

**Agent runtime guardrails**: NVIDIA NeMo Guardrails is "a scalable solution for orchestrating AI guardrails that keep agentic AI applications safe, reliable, and aligned." It supports multi-agent deployments and integrates with frameworks like LangChain, LangGraph, and LlamaIndex [NVIDIA](https://developer.nvidia.com/nemo-guardrails). Its flow management capability "blocks, filters, or tailors next action or responses based on your requirements" [NVIDIA](https://developer.nvidia.com/nemo-guardrails).

**Tool/action approval**: CrowdStrike Falcon AIDR, integrated with NVIDIA NeMo Guardrails, applies policies at critical points including "chat input sanitization, chat output filtering, RAG data ingestion, and **agent tool invocation**" [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/). This means the firewall evaluates not just what the agent says but what it attempts to do -- blocking unauthorized tool calls or data access attempts.

**Chain-of-thought monitoring**: Forrester's AEGIS framework identifies the need to "detect hallucinations and deploy circuit breakers" in agent workflows, treat agents as a new identity class, and "enforce least agency" -- a principle analogous to least privilege but for AI agent capabilities [Forrester AEGIS](https://www.forrester.com/blogs/introducing-aegis-the-guardrails-cisos-need-for-the-agentic-enterprise/).

The urgency of agent-level controls is underscored by recent incidents: VentureBeat reported that "adversaries injected malicious prompts into legitimate AI tools at more than 90 organizations in 2025, stealing credentials and cryptocurrency" -- and the next generation of autonomous SOC agents shipping in 2026 "can rewrite your firewall rules, modify IAM policies, and quarantine endpoints" [VentureBeat](https://venturebeat.com/security/adversaries-hijacked-ai-security-tools-at-90-organizations-the-next-wave-has-write-access-to-the-firewall).

### 2.3 Tool-Level: MCP Servers, Function Calls, API Invocation Controls

The tool level addresses the external actions that AI agents take -- function calls, API invocations, and interactions with Model Context Protocol (MCP) servers and other integration points.

**Lakera** explicitly positions itself as securing "GenAI, agents, and MCPs for enterprise teams" with "real-time threat detection, prompt attack prevention, and data leakage protection" [Lakera](https://www.lakera.ai/). The product provides runtime protection for the tools and data sources that agents connect to.

**CrowdStrike Falcon AIDR** policies are designed to be applied at multiple points in the AI workflow: "chat input sanitization, chat output filtering, RAG data ingestion, and agent tool invocation." Each policy consists of a set of enabled detectors configured to "detect, block, redact, encrypt, or transform content" [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/).

**Cisco AI Defense** provides "model and application agnostic security" that "protects your generative AI applications, including chatbots, retrieval augmented generation (RAG) apps, and AI agents" and offers "network-level visibility and enforcement" using "a multitude of enforcement points" [Cisco AI Runtime](https://www.cisco.com/site/us/en/products/security/ai-defense/ai-runtime/index.html). The integration with NVIDIA NeMo Guardrails enables Cisco to extend enforcement into the NeMo ecosystem where developers define tool-calling boundaries [Cisco Blog](https://blogs.cisco.com/ai/cisco-ai-defense-integrates-with-nvidia-nemo-guardrails).

**Note on MCP-specific security**: MCP (Model Context Protocol) server security is still an emerging area. While vendors like Lakera explicitly reference MCP protection, and CrowdStrike addresses tool invocation broadly, dedicated MCP firewall standards are not yet formalized. This is a rapidly evolving space. **Confidence: MODERATE** -- vendor claims exist but independent validation of MCP-specific capabilities is limited.

### 2.4 How Inline Policy Enforcement Works: Pass/Fail Verdicts

AI firewalls do produce explicit **pass/fail verdicts** on requests. The enforcement model operates as follows:

**Request flow:**
1. User or application sends a prompt/request to the AI model endpoint
2. The AI firewall intercepts the request **before** it reaches the model
3. Input scanners evaluate the prompt against configured policies (injection detection, PII scanning, topic filtering, etc.)
4. A verdict is rendered: **ALLOW** (pass through), **BLOCK** (reject and return error/safe response), or **MODIFY** (redact sensitive data, sanitize content, then forward)
5. If allowed, the prompt reaches the model and a response is generated
6. Output scanners evaluate the response against output policies
7. A second verdict is rendered on the response: ALLOW, BLOCK, or MODIFY
8. The final response reaches the user only after passing both gates

**Concrete enforcement mechanisms by vendor:**

| Vendor | Enforcement Actions | Verdict Style |
|--------|-------------------|---------------|
| **Cloudflare** | Score-based (1-99) + category tags; customer creates WAF rules to block/allow based on thresholds | Score + rule-based block/allow |
| **Cisco AI Defense** | Block malicious inputs, block non-compliant outputs, auto-configured guardrails per model | Binary block/allow + customizable |
| **CrowdStrike AIDR** | Detect, block, redact, encrypt, or transform | Multi-action policies |
| **LLM Guard** | Input scanners return pass/fail per scanner; output scanners validate compliance | Per-scanner boolean + sanitized output |
| **NVIDIA NeMo** | Block, filter, or tailor next action/response | Flow-based actions |

Sources: [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/), [Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/ai-runtime/index.html), [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/), [LLM Guard](https://github.com/protectai/llm-guard), [NVIDIA](https://developer.nvidia.com/nemo-guardrails)

Gartner's TRiSM framework validates this model, defining runtime enforcement as: "Applied to models, applications and agent interactions to support transactional alignment with organizational governance policies. Applicable connections, processes, communications, inputs and outputs are inspected for violations of policies and expected behavior. Anomalies are highlighted and either **blocked, autoremediated or forwarded to humans** or incident response systems for investigation, triage, response and applicable remediation" [F5 citing Gartner TRiSM](https://www.f5.com/company/blog/ai-security-through-the-analyst-lens-insights-from-gartner-forrester-and-kuppingercole).

---

## 3. How Rules Work: Configurable Policies

### 3.1 Types of Configurable Rules

AI firewalls support a rich set of policy types. Drawing from LLM Guard's open-source scanner library -- which provides the most transparent view into actual implementation -- and from vendor documentation, the following categories are configurable:

**Input (Prompt) Policies:**

| Policy Type | Description | Example Implementation |
|-------------|-------------|----------------------|
| **Regex / DLP patterns** | Pattern matching for PII, secrets, credit cards, SSNs, API keys | LLM Guard `Regex` scanner, `Secrets` scanner; Cloudflare SDD managed rules |
| **PII anonymization** | Detect and replace PII with placeholders before sending to model | LLM Guard `Anonymize` scanner (uses NER models for entity detection) |
| **Prompt injection detection** | ML-based classification of whether input attempts to override system instructions | LLM Guard `PromptInjection` scanner; Cloudflare scoring (1-99); Lakera Guard |
| **Topic banning** | Block prompts about prohibited topics (politics, competitors, etc.) | LLM Guard `BanTopics` scanner; NeMo Guardrails topical rails; Cloudflare category tags |
| **Substring banning** | Block specific words or phrases | LLM Guard `BanSubstrings` scanner |
| **Code detection** | Detect and optionally block code in prompts | LLM Guard `BanCode` scanner |
| **Language enforcement** | Restrict prompts to specific languages | LLM Guard `Language` scanner |
| **Token limits** | Enforce maximum prompt length to prevent resource abuse | LLM Guard `TokenLimit` scanner |
| **Toxicity detection** | ML-based classification of harmful, offensive, or hateful language | LLM Guard `Toxicity` scanner |
| **Invisible text detection** | Detect hidden unicode characters used in adversarial attacks | LLM Guard `InvisibleText` scanner |
| **Gibberish detection** | Block nonsensical inputs used in probing attacks | LLM Guard `Gibberish` scanner |
| **Rate limiting** | Throttle requests by IP, API key, session, or prompt complexity | Cloudflare Rate Limiting; Radware behavioral limits |

**Output (Response) Policies:**

| Policy Type | Description | Example Implementation |
|-------------|-------------|----------------------|
| **Sensitive data scanning** | Detect PII, credentials, or proprietary data in outputs | LLM Guard `Sensitive` scanner; Cloudflare SDD |
| **Toxicity/bias filtering** | Block offensive, biased, or harmful output | LLM Guard `Toxicity`, `Bias` scanners |
| **Factual consistency** | Validate output against provided context (RAG grounding) | LLM Guard `FactualConsistency` scanner; NeMo Guardrails fact-checking |
| **Relevance checking** | Ensure response is on-topic relative to prompt | LLM Guard `Relevance` scanner |
| **No-refusal detection** | Detect when model refuses to answer (policy compliance check) | LLM Guard `NoRefusal` scanner |
| **URL safety** | Check URLs in output for reachability and safety | LLM Guard `MaliciousURLs`, `URLReachability` scanners |
| **Language consistency** | Ensure output language matches input language | LLM Guard `LanguageSame` scanner |
| **JSON validation** | Validate structured output format | LLM Guard `JSON` scanner |
| **Competitor mention blocking** | Block references to specified competitors | LLM Guard `BanCompetitors` scanner |
| **De-anonymization** | Restore anonymized PII in outputs for authorized recipients | LLM Guard `Deanonymize` scanner |

Sources: [LLM Guard GitHub](https://github.com/protectai/llm-guard), [Protect AI](https://protectai.com/llm-guard), [Cloudflare](https://blog.cloudflare.com/firewall-for-ai/), [NVIDIA NeMo](https://developer.nvidia.com/nemo-guardrails)

### 3.2 Model Allowlists and Endpoint Controls

Cisco AI Defense provides **model and application discovery** that automatically inventories AI assets -- "models, agents, knowledge bases, and vector stores across your distributed cloud environments" [Cisco Blog](https://blogs.cisco.com/ai/cisco-ai-defense-integrates-with-nvidia-nemo-guardrails). This discovery capability enables organizations to enforce model allowlists, ensuring that only approved models are accessible and that shadow AI deployments are detected.

Radware notes that AI firewalls help discover "shadow AI" endpoints -- "unauthorized or untracked AI-enabled services used within an organization, often deployed outside of official IT oversight" -- and bring them under centralized policy enforcement [Radware](https://www.radware.com/cyberpedia/ai-firewall/).

### 3.3 Tool Approval Policies

CrowdStrike Falcon AIDR enables teams to "create named detection policies tailored to their specific security requirements. A policy is a set of enabled detectors configured to detect, block, redact, encrypt, or transform content. Policies serve as AI guardrails applied at critical points in AI agent and application workflows such as chat input sanitization, chat output filtering, RAG data ingestion, and **agent tool invocation**" [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/). With over 75 built-in classification rules and support for custom data classification, these policies can govern which tools an agent is permitted to call and what data it is permitted to pass to those tools.

### 3.4 Rule Implementation: Static + Dynamic

As a practical matter, production guardrail systems combine **static rules** (fast, sub-millisecond regex checks) with **dynamic ML classifiers** (more accurate but higher latency). A technical guide on building LLM guardrails notes: "Rule-based checks are fast (sub-millisecond) and should be the first line of defence. For more sophisticated injection detection or topic classification, follow up with a lightweight LLM-as-classifier call" using a small fast model, keeping "classifier latency under 100ms" [ML Journey](https://mljourney.com/how-to-build-llm-guardrails-for-production-applications/).

NVIDIA NeMo Guardrails benchmarks show that "orchestrating up to five GPU-accelerated guardrails in parallel increases detection rate by 1.4x while adding only ~0.5 seconds of latency" [NVIDIA](https://developer.nvidia.com/nemo-guardrails).

---

## 4. Standards and Frameworks Alignment

### 4.1 OWASP LLM Top 10 (2025 Edition)

The OWASP Top 10 for LLM Applications 2025 is the most widely adopted vulnerability taxonomy for LLM security. Its top risks include:

| Rank | Vulnerability | AI Firewall Mitigation |
|------|--------------|----------------------|
| **LLM01** | Prompt Injection | Input scanning with ML classifiers and pattern matching |
| **LLM02** | Sensitive Information Disclosure | Output scanning for PII/secrets; input PII redaction |
| **LLM03** | Supply Chain Vulnerabilities | Model scanning and allowlisting (not directly firewall, but ecosystem) |
| **LLM04** | Data and Model Poisoning | Retrieval monitoring in RAG pipelines |
| **LLM05** | Insecure Output Handling | Output validation and sanitization |
| **LLM06** | Excessive Agency | Tool-call governance and action approval |
| **LLM07** | System Prompt Leakage | Prompt extraction detection |
| **LLM08** | Vector and Embedding Weaknesses | RAG data scanning |
| **LLM09** | Misinformation | Factual consistency checking |
| **LLM10** | Unbounded Consumption | Rate limiting and token limits |

Sources: [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf), [Cloudsine](https://www.cloudsine.tech/making-sense-of-ai-security-frameworks-owasp-mitre-atlas-and-the-nist-rmf/)

AI firewalls directly address **LLM01, LLM02, LLM05, LLM06, LLM07, and LLM10** through inline enforcement. They partially address LLM04 and LLM09 through RAG retrieval monitoring and factual grounding checks. LLM03 (supply chain) requires broader model security tooling beyond the runtime firewall.

Cisco AI Defense explicitly states it "makes it easy to comply with AI security standards, including the OWASP Top 10 for LLM Applications" and publishes an AI security taxonomy mapping its capabilities to OWASP risks [Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/ai-runtime/index.html). Lakera is cited in the "OWASP LLM and GenAI Security Landscape Guide 2025 as addressing the Top 10 risks" [Lakera](https://www.lakera.ai/).

### 4.2 MITRE ATLAS

MITRE ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems) is a globally accessible knowledge base of adversary tactics and techniques against AI-enabled systems, modeled after the MITRE ATT&CK framework. As of version 5.1.0 (November 2025), ATLAS catalogs **16 tactics, 84 techniques, 56 sub-techniques, 32 mitigations, and 42 real-world case studies** [Vectra AI](https://www.vectra.ai/topics/mitre-atlas). The February 2026 update (v5.4.0) added further agent-focused techniques [Vectra AI](https://www.vectra.ai/topics/mitre-atlas).

Key ATLAS tactics relevant to AI firewalls:

- **ML Model Access (AI-specific tactic)**: Adversaries interact with an AI system through its intended interface to gather information or launch attacks. AI firewalls defend against this by inspecting all model interactions.
- **ML Attack Staging (AI-specific tactic)**: Adversaries prepare attacks targeting AI models. AI firewalls detect staging attempts through behavioral analysis.
- **LLM Prompt Injection (AML.T0051)**: A technique under Initial Access where adversaries use crafted prompts to manipulate LLM behavior. This is the primary threat that prompt injection detection addresses.
- **Data Poisoning**: Attacks on training data or RAG retrieval data. AI firewalls with retrieval monitoring can partially mitigate this at inference time.

ATLAS provides a critical link: approximately **70% of ATLAS mitigations map to existing security controls**, making integration with current SOC workflows practical [Vectra AI](https://www.vectra.ai/topics/mitre-atlas). This means AI firewalls can be positioned within existing security infrastructure rather than requiring entirely new operational models.

Over 150 organizations are engaged with ATLAS, and it was collaboratively developed with Microsoft, Intel, Verizon, Zenity, CrowdStrike, and 10+ other organizations [NIST/MITRE](https://csrc.nist.gov/csrc/media/Presentations/2025/mitre-atlas/TuePM2.1-MITRE%20ATLAS%20Overview%20Sept%202025.pdf).

### 4.3 NIST AI Risk Management Framework (AI RMF)

The NIST AI RMF provides voluntary, non-sector-specific guidance for managing AI risk across four core functions:

| Function | Purpose | AI Firewall Relevance |
|----------|---------|----------------------|
| **GOVERN** | Establish AI governance policies, roles, accountability | Firewalls enforce governance policies at runtime |
| **MAP** | Identify AI assets, system boundaries, risk scenarios | Discovery capabilities (model/endpoint inventory) |
| **MEASURE** | Quantify risk via metrics, bias audits, stress testing | Firewall logs and metrics feed risk measurement |
| **MANAGE** | Implement and monitor mitigation strategies | Runtime enforcement is a core mitigation control |

Source: [Cloudsine](https://www.cloudsine.tech/making-sense-of-ai-security-frameworks-owasp-mitre-atlas-and-the-nist-rmf/)

NIST's AI RMF emphasizes six pillars of trustworthy AI: robustness, reliability, fairness, explainability, privacy, and security. AI firewalls directly contribute to **robustness** (blocking adversarial inputs), **privacy** (PII redaction and DLP), and **security** (prompt injection and jailbreak prevention). In late 2024, NIST released a GenAI-specific AI RMF profile addressing content authenticity, hallucinations, and model misuse [Cloudsine](https://www.cloudsine.tech/making-sense-of-ai-security-frameworks-owasp-mitre-atlas-and-the-nist-rmf/).

Securiti explicitly positions its LLM Firewall as protecting "against the threats covered under the OWASP Top 10 List for LLM Applications and NIST AI RMF v.1" [Securiti](https://securiti.ai/what-is-llm-firewall/).

### 4.4 Gartner AI TRiSM and Forrester AEGIS

**Gartner AI TRiSM** (AI Trust, Risk and Security Management) is the analyst framework most directly relevant to AI firewalls. Gartner's Market Guide for AI TRiSM states that "the AI TRiSM market is distinguished by its independence from frontier model provider solutions" and that enterprises "must retain independence from any single AI model or hosting provider" [F5 citing Gartner](https://www.f5.com/company/news/press-releases/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises). Gartner defines runtime inspection and enforcement as a core TRiSM component: inputs and outputs are inspected for policy violations, with anomalies "blocked, autoremediated or forwarded to humans" [F5 citing Gartner](https://www.f5.com/company/blog/ai-security-through-the-analyst-lens-insights-from-gartner-forrester-and-kuppingercole). Lakera is listed as a representative GenAI TRiSM vendor in Gartner's 2024 Innovation Guide for Generative AI [Lakera](https://www.lakera.ai/).

**Forrester's AEGIS** (Agentic AI Guardrails for Information Security) is a six-domain framework published in August 2025 specifically for securing agentic AI. Its domains -- GRC, IAM, Data Security and Privacy, Application Security, Threat Management, and Zero Trust Architecture -- provide the governance model that AI firewalls enforce at runtime. AEGIS introduces principles of **least agency** (analog of least privilege for agents), **continuous assurance**, and **explainable outcomes** [Forrester](https://www.forrester.com/blogs/introducing-aegis-the-guardrails-cisos-need-for-the-agentic-enterprise/). Forrester VP Jeff Pollard argues that "CISOs must now secure intent, not just infrastructure" and that "agentic AI introduces emergent behavior that can bypass entitlements and escalate privileges" [Forrester](https://www.forrester.com/blogs/introducing-aegis-the-guardrails-cisos-need-for-the-agentic-enterprise/).

### 4.5 How the Frameworks Work Together

The three primary security frameworks are complementary rather than competing:

| Framework | Perspective | Primary Audience | Role |
|-----------|------------|-----------------|------|
| **OWASP LLM Top 10** | Developer-centric vulnerability list | Developers, AppSec teams | Identifies what to defend against |
| **MITRE ATLAS** | Adversary-centric attack matrix | Security analysts, SOC teams, red teams | Models how attackers operate |
| **NIST AI RMF** | Governance and risk management | CISOs, risk managers, executives | Defines organizational controls |

Source: [Cloudsine](https://www.cloudsine.tech/making-sense-of-ai-security-frameworks-owasp-mitre-atlas-and-the-nist-rmf/)

An AI firewall operationalizes all three: it implements defenses against OWASP vulnerabilities, detects ATLAS-catalogued attack techniques, and provides the runtime enforcement component of NIST AI RMF's MANAGE function. Cisco AI Defense explicitly publishes an AI security taxonomy that maps to standards from MITRE, NIST, and OWASP [Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/ai-runtime/index.html).

---

## 5. Vendor Landscape

The AI firewall market in 2026 is characterized by rapid consolidation and convergence between AI security startups and established security platform vendors:

| Vendor/Product | Type | Key Differentiator | Funding/Status |
|----------------|------|-------------------|----------------|
| **Cisco AI Defense** (ex-Robust Intelligence) | Enterprise platform | First mover with "AI Firewall" branding; network-level enforcement; integrates with NeMo Guardrails | Acquired by Cisco (Oct 2024) |
| **F5 AI Guardrails** (ex-CalypsoAI) | Enterprise platform | Governance-focused; government/defense contracts; integrated with F5 ADSP | Acquired by F5 for $180M (Sep 2025) |
| **Lakera Guard** | Startup | Prompt injection specialist; largest AI red team dataset (Gandalf); MCP security | ~$20M raised |
| **Protect AI (LLM Guard)** | Open source + enterprise | Broadest platform (model scanning, SBOM, runtime guardrails, bug bounty); open source toolkit | ~$108M raised |
| **NVIDIA NeMo Guardrails** | Open source framework | GPU-accelerated; framework-level integration (LangChain, LlamaIndex); programmable policies | NVIDIA-backed |
| **CrowdStrike Falcon AIDR** | Enterprise security | 75+ built-in classification rules; integrates with NeMo; tool invocation policies | Part of CrowdStrike platform |
| **Cloudflare Firewall for AI** | CDN/WAF provider | Leverages existing WAF infrastructure; edge deployment; score-based verdicts | Part of Cloudflare platform |
| **HiddenLayer** | Startup | ML Detection & Response (MLDR); model supply chain security | ~$56M raised |
| **Adversa AI** | Startup | Adversarial testing and red teaming specialist | ~$5M raised |

Sources: [WeCompareAI](https://www.wecompareai.com/blog/daily-ai-compare-2026-03-29-ai-security-platforms-in-2026-whos-really-protecting-your-models), [Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/robust-intelligence-is-part-of-cisco/index.html), [F5](https://www.f5.com/company/news/press-releases/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises), [Lakera](https://www.lakera.ai/), [LLM Guard](https://github.com/protectai/llm-guard), [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/)

**Key market observation**: No single vendor covers every capability equally. Model supply chain security is only offered by Protect AI and HiddenLayer. LLM guardrails are core to Protect AI, Cisco, Lakera, and CalypsoAI/F5. Adversarial red teaming at depth is Adversa AI's specialty. Organizations typically need to combine capabilities [WeCompareAI](https://www.wecompareai.com/blog/daily-ai-compare-2026-03-29-ai-security-platforms-in-2026-whos-really-protecting-your-models).

---

## 6. Limitations and Open Challenges

AI firewalls are not a silver bullet. Several important limitations must be acknowledged:

**Evasion risk**: Sophisticated attackers continuously develop novel prompt injection techniques that evade current detection models. AI firewalls require continuous updates to their threat intelligence and detection models to remain effective [Radware](https://www.radware.com/cyberpedia/ai-firewall/).

**False positives**: Strict filtering may block legitimate use cases or degrade user experience. Calibrating detection thresholds involves ongoing trade-offs between security and usability [Radware](https://www.radware.com/cyberpedia/ai-firewall/).

**Latency overhead**: Inspecting every prompt and response adds processing latency. Vendors report sub-50ms (Lakera), ~500ms for five parallel guardrails (NVIDIA NeMo), and sub-100ms (CrowdStrike AIDR), but these add up in multi-agent, multi-hop workflows [Lakera](https://www.lakera.ai/), [NVIDIA](https://developer.nvidia.com/nemo-guardrails), [CrowdStrike](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/).

**Limited standards maturity**: As Radware notes, "as an emerging category, AI firewalls lack widely adopted benchmarks, making product selection and integration challenging" [Radware](https://www.radware.com/cyberpedia/ai-firewall/).

**Model-specific behavior**: Effectiveness varies across different models and deployments, requiring per-model configuration and ongoing adaptation [Radware](https://www.radware.com/cyberpedia/ai-firewall/).

**Enterprise guardrails vs. model-provider guardrails**: Cisco's blog makes a critical architectural point: "Enterprises shouldn't rely on the built-in guardrails created by model developers, as they are different for each model, largely optimized for performance over security, and alignment is easily broken when changes to the model are made" [Cisco Blog](https://blogs.cisco.com/ai/cisco-ai-defense-integrates-with-nvidia-nemo-guardrails). This argues for independent, third-party AI firewalls rather than relying solely on model-native safety features.

---

## Conclusion

AI firewalls represent a necessary evolution of the security perimeter for the age of AI agents and LLM-powered applications. They are not merely rebranded WAFs -- they operate at a fundamentally different layer (semantic content rather than network packets), address a fundamentally different threat model (prompt injection and data leakage rather than SQL injection and DDoS), and enforce policies that traditional security tools cannot understand (topic relevance, toxicity, factual grounding, tool-call authorization).

The reason AI firewalls have emerged so rapidly is the **structural difference** between traditional and AI applications: LLMs are non-deterministic, their control plane and data plane are inseparable, and agentic AI systems have the ability to take real-world actions autonomously. These properties create attack surfaces that no amount of network-layer security can address. The category is being validated by major acquisitions (Cisco/Robust Intelligence, F5/CalypsoAI), enterprise security vendor investment (CrowdStrike, Cloudflare), analyst frameworks (Gartner TRiSM, Forrester AEGIS), and open-source adoption (NVIDIA NeMo at 6K GitHub stars, LLM Guard at 2.8K stars).

For organizations deploying AI systems, the practical recommendation is: treat AI firewalls as a **required infrastructure component**, not an optional enhancement. The OWASP LLM Top 10, MITRE ATLAS, and NIST AI RMF all point to the same conclusion -- runtime inspection and enforcement of AI interactions is a fundamental security control for production AI deployments.

---

## Sources

1. [Radware - AI Firewall: 5 Key Functions, Pros/Cons & Best Practices](https://www.radware.com/cyberpedia/ai-firewall/)
2. [Securiti - What is an LLM Firewall: Navigating Unprecedented AI Threats](https://securiti.ai/what-is-llm-firewall/)
3. [Cloudflare - Firewall for AI Announcement](https://blog.cloudflare.com/firewall-for-ai/)
4. [Cisco AI Defense - AI Runtime Protection](https://www.cisco.com/site/us/en/products/security/ai-defense/ai-runtime/index.html)
5. [Cisco - Robust Intelligence Is Now Part of Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/robust-intelligence-is-part-of-cisco/index.html)
6. [Cisco Blog - AI Defense Integrates with NVIDIA NeMo Guardrails](https://blogs.cisco.com/ai/cisco-ai-defense-integrates-with-nvidia-nemo-guardrails)
7. [Lakera - The AI-Native Security Platform](https://www.lakera.ai/)
8. [F5 - F5 to Acquire CalypsoAI](https://www.f5.com/company/news/press-releases/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises)
9. [F5 - AI Security Through the Analyst Lens (Gartner, Forrester, KuppingerCole)](https://www.f5.com/company/blog/ai-security-through-the-analyst-lens-insights-from-gartner-forrester-and-kuppingercole)
10. [Forrester - Introducing AEGIS: Agentic AI Guardrails for Information Security](https://www.forrester.com/blogs/introducing-aegis-the-guardrails-cisos-need-for-the-agentic-enterprise/)
11. [WeCompareAI - AI Security Platforms in 2026](https://www.wecompareai.com/blog/daily-ai-compare-2026-03-29-ai-security-platforms-in-2026-whos-really-protecting-your-models)
12. [Cloudsine - Making Sense of AI Security Frameworks: OWASP, MITRE ATLAS, NIST RMF](https://www.cloudsine.tech/making-sense-of-ai-security-frameworks-owasp-mitre-atlas-and-the-nist-rmf/)
13. [NVIDIA - NeMo Guardrails Developer Page](https://developer.nvidia.com/nemo-guardrails)
14. [NVIDIA NeMo Guardrails - GitHub](https://github.com/NVIDIA-NeMo/Guardrails)
15. [CrowdStrike - Secure Homegrown AI Agents with Falcon AIDR and NVIDIA NeMo Guardrails](https://www.crowdstrike.com/en-us/blog/secure-homegrown-ai-agents-with-crowdstrike-falcon-aidr-and-nvidia-nemo-guardrails/)
16. [Protect AI - LLM Guard Product Page](https://protectai.com/llm-guard)
17. [Protect AI - LLM Guard GitHub](https://github.com/protectai/llm-guard)
18. [Vectra AI - MITRE ATLAS: AI Security Framework Guide](https://www.vectra.ai/topics/mitre-atlas)
19. [NIST CSRC - MITRE ATLAS Overview Presentation (Sept 2025)](https://csrc.nist.gov/csrc/media/Presentations/2025/mitre-atlas/TuePM2.1-MITRE%20ATLAS%20Overview%20Sept%202025.pdf)
20. [VentureBeat - Adversaries Hijacked AI Security Tools at 90+ Organizations](https://venturebeat.com/security/adversaries-hijacked-ai-security-tools-at-90-organizations-the-next-wave-has-write-access-to-the-firewall)
21. [OWASP Top 10 for LLM Applications 2025 (PDF)](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf)
22. [ML Journey - How to Build LLM Guardrails for Production Applications](https://mljourney.com/how-to-build-llm-guardrails-for-production-applications/)
