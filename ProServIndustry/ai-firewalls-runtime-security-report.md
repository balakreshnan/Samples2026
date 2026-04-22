# AI Firewalls and AI Runtime Security: Vendor Landscape Report

**Date:** April 22, 2026
**Scope:** Commercial products and open-source frameworks for protecting AI models, agents, and applications at runtime

---

## Executive Summary

The AI runtime security market has rapidly matured from a nascent category in 2023 to a multi-billion-dollar segment with products spanning cloud platform providers, traditional security vendors, AI-native startups, and open-source frameworks. The market is characterized by significant M&A consolidation (Palo Alto Networks acquiring Protect AI, Cisco acquiring Robust Intelligence, Check Point acquiring Lakera, F5 acquiring CalypsoAI, SentinelOne acquiring Prompt Security), a universal expansion from LLM-focused guardrails toward agentic AI security (covering MCP servers, tool use, and autonomous agent behavior), and a convergence on common threat categories (prompt injection, data leakage, jailbreaks, toxic content) with differentiation emerging in deployment models and ecosystem integration.

**Key market dynamics:**
- **Consolidation wave:** Nearly every major security vendor has acquired an AI-native startup to accelerate their AI security capabilities.
- **Agentic AI shift:** Since late 2025, every vendor has pivoted messaging and capabilities toward securing autonomous AI agents, MCP servers, and tool-mediated interactions.
- **Deployment model divergence:** Cloud providers offer native API/SDK integration; network security vendors leverage inline/gateway enforcement; startups favor proxy and browser extension models.

---

## Cloud / Platform Providers

### Azure AI Content Safety / Prompt Shields (Microsoft)

**What it protects:** LLM inputs and outputs across Azure OpenAI and non-OpenAI models. Covers prompt injection (direct and indirect/XPIA), harmful content (violence, hate, sexual, self-harm), groundedness/hallucination detection, and protected material identification [Azure AI Content Safety Catalog](https://ai.azure.com/catalog/models/Azure-AI-Content-Safety).

**Deployment model:** Cloud API, on-premises container, and embedded (on-device) deployment options. Integrates natively with Azure OpenAI content filters and Azure AI Foundry [Azure Blog](https://azure.microsoft.com/en-us/blog/enhance-ai-security-with-azure-prompt-shields-and-azure-ai-content-safety/).

**Notable capabilities:**
- **Spotlighting** (announced at Microsoft Build 2025): Distinguishes trusted from untrusted inputs to improve indirect prompt injection detection in documents, emails, and web content [Azure Blog](https://azure.microsoft.com/en-us/blog/enhance-ai-security-with-azure-prompt-shields-and-azure-ai-content-safety/).
- **Real-time response** with contextual awareness that reduces false positives by understanding prompt intent [Azure Blog](https://azure.microsoft.com/en-us/blog/enhance-ai-security-with-azure-prompt-shields-and-azure-ai-content-safety/).
- **Custom categories** allowing users to train custom content filters with minimal examples [Azure AI Content Safety Catalog](https://ai.azure.com/catalog/models/Azure-AI-Content-Safety).
- Integration with **Microsoft Defender for Cloud** for AI security posture management [Zecker McDecker](https://www.zeckermcdecker.com/technology-news/enhance-ai-security-with-azure-prompt-shields-and-azure-ai-content-safety/).
- Supports 100+ languages, deployable on cloud, on-prem, and on devices [Azure AI Content Safety Catalog](https://ai.azure.com/catalog/models/Azure-AI-Content-Safety).

**Distinct advantage:** Deepest integration with the Azure/Microsoft ecosystem, including Defender for Cloud AIPM; one of the few products offering on-device deployment for edge scenarios.

---

### AWS Bedrock Guardrails (Amazon)

**What it protects:** Foundation model inputs and outputs (prompts and responses, excluding reasoning content blocks) across Amazon Bedrock models. Covers content filtering, denied topics, sensitive information (PII) redaction, contextual grounding checks, and automated reasoning policies [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).

**Deployment model:** Cloud-native API integrated directly into the Amazon Bedrock service. Applied via API calls (ApplyGuardrail) that can be used with any custom or third-party model, not just Bedrock-hosted models [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).

**Notable capabilities:**
- **Configurable content filters** with adjustable severity thresholds for hate, insults, sexual, violence, misconduct, and prompt attack categories [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Denied topic policies** using natural language definitions -- users describe topics to block without training data [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Sensitive information filters** with built-in PII types and custom regex patterns, with options to block or mask [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Contextual grounding checks** that detect hallucinations by verifying responses against source content and relevance to the query [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Automated reasoning policies** (preview) that use deterministic verification against pre-defined logical policies [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- Can be applied to any model via the standalone ApplyGuardrail API, extending protection beyond Bedrock-hosted models [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).

**Distinct advantage:** Tightest integration with the AWS ecosystem; ApplyGuardrail API enables use with any model (including self-hosted); unique automated reasoning capability for deterministic verification.

---

### Google Cloud Model Armor

**What it protects:** AI prompts, responses, and agent interactions. Covers prompt injection, jailbreaking, malicious URL detection, harmful content (hate, harassment, sexually explicit, dangerous topics), sensitive data (PII, financial, credentials), and malware detection [Google Cloud Model Armor](https://cloud.google.com/security/products/model-armor).

**Deployment model:** Cloud API service with integration into Google Cloud's Vertex AI platform. Part of the Security Command Center AI Protection suite. Offers a free tier [Google Cloud Model Armor](https://cloud.google.com/security/products/model-armor).

**Notable capabilities:**
- **Hybrid defense-in-depth approach** using multiple detection layers [Google Cloud Model Armor](https://cloud.google.com/security/products/model-armor).
- **Integrated with Google Cloud Sensitive Data Protection** service, specifically adapted for AI-generated text unpredictability [Google Cloud Model Armor](https://cloud.google.com/security/products/model-armor).
- **Granular content safety** with adjustable confidence thresholds per content category [Google Cloud Model Armor](https://cloud.google.com/security/products/model-armor).
- **Malware detection and safe browsing** -- detects malicious files and URLs in prompts/responses [Google Cloud Model Armor](https://cloud.google.com/security/products/model-armor).
- **Malicious URL detection** embedded in prompts or responses [Google Cloud Model Armor](https://cloud.google.com/security/products/model-armor).

**Distinct advantage:** Unique combination of content safety with malware/URL scanning; free tier available; integrated with Google's broader Sensitive Data Protection infrastructure.

---

### Cloudflare Firewall for AI / AI Gateway

**What it protects:** AI applications, employee AI usage (including Shadow AI), and model interactions. Covers prompt injection, toxic content, PII leakage, and unauthorized AI app usage [Cloudflare Press Release](https://www.cloudflare.com/en-gb/press/press-releases/2025/cloudflare-introduces-cloudflare-for-ai/).

**Deployment model:** Cloud-based proxy/gateway running on Cloudflare's global edge network (190+ cities). Part of the broader Cloudflare for AI suite within the Zero Trust platform [Cloudflare Press Release](https://www.cloudflare.com/en-gb/press/press-releases/2025/cloudflare-introduces-cloudflare-for-ai/).

**Notable capabilities:**
- **AI Security Posture Management (AI-SPM)** with Shadow AI Report for discovering all AI applications in use across an organization [Lushbinary](https://lushbinary.com/blog/cloudflare-ai-week-2025-everything-released/).
- **Application Confidence Scores** (1-5 rating) evaluating SaaS/GenAI app trustworthiness based on compliance, data policies, security controls, and vendor stability [Lushbinary](https://lushbinary.com/blog/cloudflare-ai-week-2025-everything-released/).
- **Prompt-level DLP** controls for GenAI interactions [Lushbinary](https://lushbinary.com/blog/cloudflare-ai-week-2025-everything-released/).
- **CASB integrations** with ChatGPT, Claude, and Gemini [Lushbinary](https://lushbinary.com/blog/cloudflare-ai-week-2025-everything-released/).
- **Llama Guard integration** in AI Gateway for unsafe content moderation [Cloudflare Press Release](https://www.cloudflare.com/en-gb/press/press-releases/2025/cloudflare-introduces-cloudflare-for-ai/).
- **AI Gateway** provides unified endpoint routing to 350+ models, usage analytics, and traffic controls [Lushbinary](https://lushbinary.com/blog/cloudflare-ai-week-2025-everything-released/).
- FedRAMP Moderate and High authorization planned for 2026 [Lushbinary](https://lushbinary.com/blog/cloudflare-ai-week-2025-everything-released/).

**Distinct advantage:** Unique edge-network deployment model with global reach; combines AI security with broader Zero Trust platform; strong Shadow AI discovery via network-level visibility.

---

## Security Vendors

### Palo Alto Networks Prisma AIRS (AI Runtime Security)

**What it protects:** The entire AI ecosystem -- AI applications, AI models, AI data, and AI agents (including those on low-code/no-code platforms). Covers prompt injection, malicious code, toxic content, sensitive data leakage, model DoS, hallucination, identity impersonation, memory manipulation, tool misuse, and MCP server security [Prisma AIRS Docs](https://docs.paloaltonetworks.com/ai-runtime-security).

**Deployment model:** Multiple modalities: **AI Runtime Firewall** (inline network-layer enforcement for cloud/on-prem), **AI Runtime API** (SDK/Security-as-Code embedded in source code), and cloud-based management. Supports multi-cloud deployment including VM-Series [Prisma AIRS Docs](https://docs.paloaltonetworks.com/ai-runtime-security).

**Notable capabilities:**
- **Five-pillar platform:** AI Model Scanning, Posture Management, AI Red Teaming, Runtime Security, and AI Agent Security [NAND Research](https://nand-research.com/research-note-palo-alto-networks-prisma-airs-for-ai-protection/).
- **Prisma AIRS 2.0** (Oct 2025) integrates acquired Protect AI technology for deep model inspection (architectural backdoors, data poisoning, malicious code in model layers) [Help Net Security](https://www.helpnetsecurity.com/2025/10/29/palo-alto-networks-launches-prisma-airs-2-0-to-deliver-end-to-end-security-across-the-ai-lifecycle/).
- **AI Red Teaming** with 500+ specialized attacks, continuous autonomous testing, and multi-turn attack support [Help Net Security](https://www.helpnetsecurity.com/2025/10/29/palo-alto-networks-launches-prisma-airs-2-0-to-deliver-end-to-end-security-across-the-ai-lifecycle/).
- **MCP Server security** with standalone MCP Server for agent protection (Sept 2025) and MCP threat detection [Prisma AIRS Docs](https://docs.paloaltonetworks.com/ai-runtime-security).
- **Lifecycle coverage** from development-time (shift-left) through deployment governance to runtime monitoring [OpenClawAI](https://openclawai.io/blog/palo-alto-prisma-airs-agentic-ai-security-rsac-2026).
- **Secure browser** for agentic workflows within Prisma SASE [OpenClawAI](https://openclawai.io/blog/palo-alto-prisma-airs-agentic-ai-security-rsac-2026).

**Distinct advantage:** Most comprehensive single-vendor AI security platform, covering model supply chain through runtime; acquired Protect AI (closed July 2025) for model-level scanning capabilities; network-layer enforcement unique among pure AI security offerings.

---

### Cisco AI Defense (formerly Robust Intelligence)

**What it protects:** AI models, applications, agents, and data across distributed cloud environments. Covers prompt injection, jailbreaking, supply chain compromise (malicious models, poisoned data, unsafe tools), tool misuse, privilege escalation, deceptive agent behavior, memory poisoning, and intent hijacking [Cisco AI Defense Data Sheet](https://www.cisco.com/c/en/us/products/collateral/security/ai-defense/ai-defense-ds.html).

**Deployment model:** Network-layer enforcement leveraging Cisco's networking/security footprint across cloud, VPC, and on-premises AI POD deployments. Embeds into Cisco Security Cloud [Cisco AI Defense Data Sheet](https://www.cisco.com/c/en/us/products/collateral/security/ai-defense/ai-defense-ds.html).

**Notable capabilities:**
- **AI BOM (Bill of Materials)** for centralized AI asset governance including MCP servers and third-party dependencies [PR Newswire - Cisco](https://www.prnewswire.com/news-releases/cisco-redefines-security-for-the-agentic-era-with-ai-defense-expansion-and-ai-aware-sase-302683205.html).
- **MCP Catalog** that discovers and inventories MCP servers across public and private platforms [PR Newswire - Cisco](https://www.prnewswire.com/news-releases/cisco-redefines-security-for-the-agentic-era-with-ai-defense-expansion-and-ai-aware-sase-302683205.html).
- **Algorithmic red teaming** evaluating 200+ threat subcategories with adaptive single and multi-turn testing in multiple languages [Cisco AI Defense Data Sheet](https://www.cisco.com/c/en/us/products/collateral/security/ai-defense/ai-defense-ds.html).
- Builds on **Robust Intelligence** acquisition (Oct 2024), which pioneered algorithmic red teaming and the first AI Firewall [Cisco - Robust Intelligence](https://www.cisco.com/site/us/en/products/security/ai-defense/robust-intelligence-is-part-of-cisco/index.html).
- Integration with **NVIDIA NeMo Guardrails** open-source framework for modular runtime protection [PR Newswire - Cisco](https://www.prnewswire.com/news-releases/cisco-redefines-security-for-the-agentic-era-with-ai-defense-expansion-and-ai-aware-sase-302683205.html).
- Aligns with **MITRE ATLAS, OWASP Top 10 for LLMs, and NIST AI-RMF** [Cisco AI Defense Data Sheet](https://www.cisco.com/c/en/us/products/collateral/security/ai-defense/ai-defense-ds.html).
- **AI-aware SASE** with MCP visibility, logging, and intent-aware inspection [Enterprise AI World](https://www.enterpriseaiworld.com/Articles/News/News/Cisco-Rolls-Out-Suite-of-Capabilities-with-AI-Defense-and-AI-Aware-SASE-173507.aspx).

**Distinct advantage:** Leverages Cisco's unmatched networking footprint for broad visibility and enforcement across east-west and north-south AI traffic; AI BOM and MCP Catalog provide unique supply chain governance; threat intelligence from Cisco Talos.

---

### F5 AI Gateway / AI Guardrails / AI Red Team

**What it protects:** AI models, applications, agents, and APIs. Covers prompt injection, jailbreaks, data leakage, toxic content, and OWASP Top Ten for LLM Applications risks [F5 Press Release](https://www.f5.com/company/news/press-releases/f5-ai-gateway-manage-secure-traffic-application-demands).

**Deployment model:** Containerized gateway deployable in any cloud or data center. Natively integrates with F5's NGINX and BIG-IP platforms. Part of the F5 Application Delivery and Security Platform (ADSP) [F5 Press Release](https://www.f5.com/company/news/press-releases/f5-ai-gateway-manage-secure-traffic-application-demands).

**Notable capabilities:**
- **F5 AI Guardrails** (GA Jan 2026): Model-agnostic runtime security with consistent policy enforcement across all AI models, apps, and agents. Provides observability and auditability of AI inputs/outputs [Security MEA - F5](https://securitymea.com/2026/01/28/f5-unveils-ai-guardrails-and-ai-red-team-to-secure-enterprise-ai-systems/).
- **F5 AI Red Team** (GA Jan 2026): Automated adversarial testing with 10,000+ new attack techniques added monthly. Insights feed back into AI Guardrails policies [Security MEA - F5](https://securitymea.com/2026/01/28/f5-unveils-ai-guardrails-and-ai-red-team-to-secure-enterprise-ai-systems/).
- **F5 AI Remediate** (March 2026): Automates creation and validation of targeted guardrail packages from red team findings [Nasdaq - F5](https://www.nasdaq.com/press-release/f5-advances-enterprise-application-security-ai-and-post-quantum-era-2026-03-11).
- **CalypsoAI acquisition** ($180M, Sept 2025): Brings inference-layer security with red teaming, real-time defense, and observability [Intelligent CISO](https://www.intelligentciso.com/2025/09/15/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises/).
- **Semantic caching** for cost optimization and **intelligent traffic routing** for LLMs [F5 Press Release](https://www.f5.com/company/news/press-releases/f5-ai-gateway-manage-secure-traffic-application-demands).
- **Data leakage detection** via LeakSignal acquisition technology [Intelligent CISO](https://www.intelligentciso.com/2025/08/05/f5-adds-data-protection-features-for-ai-workloads-and-encrypted-traffic/).

**Distinct advantage:** Red team-to-guardrail feedback loop (identify vulnerabilities, auto-generate protections); strong application delivery heritage (NGINX, BIG-IP) for performance-optimized AI traffic handling; comprehensive ADSP integration.

---

### Fortinet FortiAI

**What it protects:** AI models, LLM data, AI workloads, and employee AI usage. Covers network-based threats, LLM data leakage, cloud AI workload protection, and unauthorized GenAI application usage [Fortinet FortiAI](https://www.fortinet.com/products/fortiai-secure).

**Deployment model:** Embedded across the Fortinet Security Fabric platform, leveraging existing FortiGate firewalls, FortiWeb WAF, FortiCNAPP, and FortiSASE infrastructure. Inline network-layer enforcement [Fortinet FortiAI](https://www.fortinet.com/products/fortiai-secure).

**Notable capabilities:**
- **Three-pillar approach:** FortiAI-Assist (GenAI + agentic AI for security operations), FortiAI-Protect (AI-driven threat detection + GenAI usage control), FortiAI-SecureAI (AI model/infrastructure/data protection) [MSSP Alert](https://www.msspalert.com/news/fortinet-expands-ai-capabilities-across-security-fabric-platform).
- **AI application monitoring** detecting 6,500+ AI-related URLs with context on model types, data paths, and purposes [MSSP Alert](https://www.msspalert.com/news/fortinet-expands-ai-capabilities-across-security-fabric-platform).
- **Zero-trust policy enforcement** to block shadow AI and high-risk AI applications [Security Info Watch](https://www.securityinfowatch.com/cybersecurity/press-release/55280685/fortinet-fortinet-expands-fortiai-across-its-security-fabric-platform).
- **DLP for LLMs** with web app and API-layer security [Fortinet FortiAI](https://www.fortinet.com/products/fortiai-secure).
- **ZTNA** for safeguarding AI applications from unauthorized access [Fortinet FortiAI](https://www.fortinet.com/products/fortiai-secure).
- **Deception technology** for early detection of attacks against AI infrastructure [Fortinet FortiAI](https://www.fortinet.com/products/fortiai-secure).
- 500+ AI patents and 15+ years of AI innovation [Security Info Watch](https://www.securityinfowatch.com/cybersecurity/press-release/55280685/fortinet-fortinet-expands-fortiai-across-its-security-fabric-platform).

**Distinct advantage:** Leverages the broad Fortinet Security Fabric installed base; AI security is not a standalone product but layered across existing firewall, WAF, CNAPP, and SASE infrastructure, minimizing new deployment requirements for existing Fortinet customers.

---

### Check Point AI Security (with Lakera acquisition)

**What it protects:** Workforce AI usage (browsers, SaaS, copilots), AI applications (model inputs/outputs), and autonomous AI agents (tool calls, file access, runtime behavior) [Check Point AI Security](https://www.checkpoint.com/ai-security/).

**Deployment model:** Integrated into the Check Point Infinity platform, leveraging existing Check Point infrastructure (CloudGuard WAF, GenAI Protect). API-based and cloud-delivered with on-prem options via Lakera technology [CSO Online](https://www.csoonline.com/article/4058653/check-point-acquires-lakera-to-build-a-unified-ai-security-stack.html).

**Notable capabilities:**
- **Lakera acquisition** (Sept 2025, ~$190M): Brings AI-native runtime protection, continuous red teaming via the Gandalf adversarial engine (80M+ attack patterns from 1M+ players), and multilingual defenses with 98%+ detection rates at sub-50ms latency [CSO Online](https://www.csoonline.com/article/4058653/check-point-acquires-lakera-to-build-a-unified-ai-security-stack.html).
- **Cyata acquisition** (Feb 2026): AI agent identity management -- discovers active agents, maps permissions, monitors behavior, enforces automated security policies [Security Solutions Media](https://www.securitysolutionsmedia.com/2026/02/16/check-point-unveils-ai-security-strategy-reinforces-platform-with-three-strategic-acquisitions/).
- **AI Defense Plane** architecture: Discover (map AI usage including Shadow AI), Protect (stop threats), Govern (enforce policy with audit trails) [Check Point AI Security](https://www.checkpoint.com/ai-security/).
- Covers 100+ languages, guarantees <50ms latency at runtime [Check Point AI Security](https://www.checkpoint.com/ai-security/).
- **Four-pillar strategy:** Hybrid Mesh Network Security, Workspace Security, Exposure Management, and AI Security [Security Solutions Media](https://www.securitysolutionsmedia.com/2026/02/16/check-point-unveils-ai-security-strategy-reinforces-platform-with-three-strategic-acquisitions/).

**Distinct advantage:** Gandalf crowdsourced adversarial intelligence (over 85M prompts from real human attackers) provides uniquely rich threat data; Cyata acquisition adds AI agent identity management not widely available elsewhere.

---

## AI-Native Security Startups

### Prompt Security (acquired by SentinelOne, Aug 2025, ~$250M)

**What it protects:** Employee GenAI usage, homegrown LLM applications, AI code assistants, and agentic AI (MCP security). Covers prompt injection, jailbreaks, data leakage, Shadow AI, and system prompt extraction [AppSecSanta](https://appsecsanta.com/prompt-security).

**Deployment model:** SaaS, on-premises, browser extension (Chrome via Intune/MDM), and API/SDK integration. Browser extension for employee AI monitoring; API integration for homegrown apps [AppSecSanta](https://appsecsanta.com/prompt-security).

**Notable capabilities:**
- **Sub-200ms detection latency** across 250+ LLM models [AppSecSanta](https://appsecsanta.com/prompt-security).
- **MCP Gateway** -- described as the first comprehensive solution for MCP security [Prompt Security](https://prompt.security/).
- **Semantic data leakage prevention** using contextual analysis (not just pattern matching) for PII, PHI, financial data, and source code [AppSecSanta](https://appsecsanta.com/prompt-security).
- **Chrome browser extension** with DOM analysis and user action tracking for shadow AI discovery [AppSecSanta](https://appsecsanta.com/prompt-security).
- Built-in **red teaming** capabilities [AppSecSanta](https://appsecsanta.com/prompt-security).
- Now integrated into **SentinelOne Singularity** platform [AppSecSanta](https://appsecsanta.com/prompt-security).
- Core OWASP research team members as co-founders [Prompt Security](https://prompt.security/).

**Distinct advantage:** Full-stack approach covering both employee AI usage (via browser extension) and homegrown app security (via API); now benefits from SentinelOne's endpoint security distribution.

---

### Lakera Guard (acquired by Check Point, Sept 2025)

**What it protects:** LLM applications and AI agents. Covers prompt attacks, data leakage, inappropriate interactions, and content safety across multiple languages and modalities [Lakera Guard Docs](https://docs.lakera.ai/guard).

**Deployment model:** Cloud API (single API call integration in ~5 minutes). Centralized policy management with out-of-the-box and customizable policies [Lakera Guard Docs](https://docs.lakera.ai/guard).

**Notable capabilities:**
- **Real-time threat intelligence** updated daily from Lakera threat research, including insights from 100K+ Gandalf attacks per day [Lakera Guard Docs](https://docs.lakera.ai/guard).
- **Gandalf adversarial engine** -- a gamified AI security challenge that has processed 85M+ prompts from 1M+ players, generating real-world attack data [Check Point AI Security](https://www.checkpoint.com/ai-security/).
- **Ultra-low latency** with few false positives (<0.5%) [CSO Online](https://www.csoonline.com/article/4058653/check-point-acquires-lakera-to-build-a-unified-ai-security-stack.html).
- Multilingual and multimodal support [Lakera Guard Docs](https://docs.lakera.ai/guard).
- Centralized control with consistent policy enforcement across applications without code changes [Lakera Guard Docs](https://docs.lakera.ai/guard).
- Now being integrated into Check Point Infinity (CloudGuard WAF and GenAI Protect) [CSO Online](https://www.csoonline.com/article/4058653/check-point-acquires-lakera-to-build-a-unified-ai-security-stack.html).

**Distinct advantage:** Gandalf crowdsourced threat intelligence is unique in the industry -- real human adversarial data rather than synthetic attacks; extremely low false positive rate.

---

### Protect AI (acquired by Palo Alto Networks, July 2025)

**What it protects:** AI/ML model supply chain, LLM applications, and AI data. Products included Guardian (runtime security), Layer (model risk management), and Recon (AI asset discovery) [Palo Alto Networks Press Release](https://www.paloaltonetworks.com/company/press/2025/palo-alto-networks-completes-acquisition-of-protect-ai).

**Deployment model:** Cloud platform and API. Also maintained open-source tools (LLM Guard, Rebuff). Now integrated into Prisma AIRS [Palo Alto Networks Press Release](https://www.paloaltonetworks.com/company/press/2025/palo-alto-networks-completes-acquisition-of-protect-ai).

**Notable capabilities:**
- **Model scanning** for architectural backdoors, data poisoning, malicious code, and deserialization attacks [Help Net Security](https://www.helpnetsecurity.com/2025/10/29/palo-alto-networks-launches-prisma-airs-2-0-to-deliver-end-to-end-security-across-the-ai-lifecycle/).
- **AI BOM (Bill of Materials)** including model architecture, training datasets, open-source licenses, and software dependencies [Help Net Security](https://www.helpnetsecurity.com/2025/10/29/palo-alto-networks-launches-prisma-airs-2-0-to-deliver-end-to-end-security-across-the-ai-lifecycle/).
- Maintained **LLM Guard** (open source) and **Rebuff** (prompt injection detector) [Protect AI - LLM Guard](https://protectai.com/llm-guard).
- Technology now forms the model security backbone of Prisma AIRS 2.0 [Help Net Security](https://www.helpnetsecurity.com/2025/10/29/palo-alto-networks-launches-prisma-airs-2-0-to-deliver-end-to-end-security-across-the-ai-lifecycle/).

**Distinct advantage:** Deepest model-level inspection capabilities (detecting threats within model weights/layers); open-source roots (LLM Guard) provided community validation; now powered by Palo Alto Networks scale.

---

### HiddenLayer

**What it protects:** AI models across the full lifecycle -- from supply chain (pre-deployment scanning) through production runtime. Covers model tampering, adversarial attacks, prompt injection, data leakage, backdoored weights, vulnerable dependencies, and agent misuse/escalation [HiddenLayer - Model Scanning](https://www.hiddenlayer.com/solutions/model-scanning).

**Deployment model:** Cloud platform with API integration. Operates without requiring direct model access -- non-invasive scanning and monitoring [HiddenLayer](https://www.hiddenlayer.com/innovation-hub/reports-and-guides).

**Notable capabilities:**
- **AISec Platform** with four pillars: AI Discovery (visibility into AI assets to eliminate shadow AI), AI Red Teaming (continuous adversarial simulation), Model Scanning (supply chain integrity), and MLDR (Machine Learning Detection and Response in production) [HiddenLayer - Model Scanning](https://www.hiddenlayer.com/solutions/model-scanning).
- **Runtime policy enforcement** preventing prompt injection, data leakage, and unsafe AI behavior [HiddenLayer - Model Scanning](https://www.hiddenlayer.com/solutions/model-scanning).
- Focuses on enterprise, financial services, government/defense, and developer use cases [HiddenLayer - Model Scanning](https://www.hiddenlayer.com/solutions/model-scanning).
- Publishes annual **AI Threat Landscape Report** [HiddenLayer](https://www.hiddenlayer.com/innovation-hub/reports-and-guides).

**Distinct advantage:** Pioneer in MLDR (ML Detection and Response) -- applying endpoint detection-style protection to AI models; non-invasive approach that doesn't require model modification; strong government/defense positioning.

---

### Lasso Security

**What it protects:** AI applications, AI agents, and AI tools across the enterprise. Covers shadow AI, adversarial manipulation, agent guardrail bypass, and supply chain risks from foundational model updates [Lasso Security](https://www.lasso.security/).

**Deployment model:** Platform integration via CI/CD pipeline connections and platform connectors. Discovers homegrown applications and AI agents automatically [Lasso Security](https://www.lasso.security/).

**Notable capabilities:**
- **Intent Security for AI Agents** -- analyzes the intent behind agent actions rather than relying on fixed patterns, addressing the non-deterministic nature of AI [Lasso Security](https://www.lasso.security/).
- **Full lifecycle governance:** Discover (agents and apps via CI integration), Analyze (security posture), Protect (runtime defenses), and Govern (policy enforcement) [Lasso Security](https://www.lasso.security/).
- **Supply chain awareness** -- monitors for behavior changes when foundational model providers push updates [Lasso Security](https://www.lasso.security/).

**Distinct advantage:** Intent-based security approach for agents; emphasizes the behavioral/intent analysis gap that traditional rule-based systems miss.

---

### CalypsoAI (acquired by F5, Sept 2025, $180M)

**What it protects:** AI models and applications at the inference layer. Covers prompt injection, jailbreaks, data leakage, policy violations, and evolving AI adversaries [Gartner Peer Insights - CalypsoAI](https://www.gartner.com/reviews/market/ai-security-and-anomaly-detection/vendor/calypsoai).

**Deployment model:** The "Inference Perimeter" -- sits at the inference layer across public cloud, private cloud, and on-premises deployments. Model and vendor agnostic [NSFOCUS](https://nsfocusglobal.com/rsac-2025-innovation-sandbox-calypsoai-a-practical-path-and-trust-foundation-for-forging-ai-system-security-protection-systems/).

**Notable capabilities:**
- **Three-module architecture:** Red-Team (automated adversarial testing), Defend (real-time guardrails), Observe (monitoring and compliance) [NSFOCUS](https://nsfocusglobal.com/rsac-2025-innovation-sandbox-calypsoai-a-practical-path-and-trust-foundation-for-forging-ai-system-security-protection-systems/).
- **Agentic Warfare** defense capabilities [AI Tools Info](https://ai.toolsinfo.com/tool/calypsoai).
- Originally designed for **national security** use cases (US DHS, US Air Force) before commercial expansion [NSFOCUS](https://nsfocusglobal.com/rsac-2025-innovation-sandbox-calypsoai-a-practical-path-and-trust-foundation-for-forging-ai-system-security-protection-systems/).
- 10,000+ new attack prompts tested monthly [Intelligent CISO](https://www.intelligentciso.com/2025/09/15/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises/).
- Founded in 2018, making it one of the earliest AI security companies [NSFOCUS](https://nsfocusglobal.com/rsac-2025-innovation-sandbox-calypsoai-a-practical-path-and-trust-foundation-for-forging-ai-system-security-protection-systems/).
- Now integrated into F5's ADSP as the backbone of F5 AI Guardrails and AI Red Team [Intelligent CISO](https://www.intelligentciso.com/2025/09/15/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises/).

**Distinct advantage:** Government/defense heritage provides high-assurance security pedigree; inference-layer focus is model/vendor agnostic; now gains F5's application delivery performance capabilities.

---

### WitnessAI

**What it protects:** Enterprise LLMs, AI applications, AI agents, and employee AI usage. Covers prompt injection, jailbreaks, multi-turn attacks, data leakage, toxic content, and unsafe agent behavior. Monitors MCP servers and tool interactions [WitnessAI](https://witness.ai/).

**Deployment model:** Network-level proxy with single-tenant architecture for data sovereignty. Developer API for model protection. Covers both employee-facing AI usage and application-level security [WitnessAI](https://witness.ai/).

**Notable capabilities:**
- **Witness Protect**: AI firewall with 99%+ prompt injection detection accuracy, behavioral runtime defense, intention-based response control, and support for 100+ LLM types [PR Newswire - WitnessAI](https://www.prnewswire.com/news-releases/witnessai-announces-automated-red-teaming-and-next-generation-ai-firewall-protection-for-enterprise-llms-and-ai-applications-302534128.html).
- **Witness Attack**: Automated red teaming using multimodal attacks, multi-step jailbreaks, fuzzing, and reinforcement-learning attacks [PR Newswire - WitnessAI](https://www.prnewswire.com/news-releases/witnessai-announces-automated-red-teaming-and-next-generation-ai-firewall-protection-for-enterprise-llms-and-ai-applications-302534128.html).
- **Agentic Security** (Jan 2026): Extends protection to AI agents with real-time observability, human-to-agent identity linking, and MCP server monitoring [TechIntelPro](https://techintelpro.com/news/ai/enterprise-ai/witnessai-raises-58m-launches-agentic-ai-security-features).
- **Intent-based behavioral policies** that analyze meaning and purpose of prompts, not just keywords [TechEdge AI](https://techedgeai.com/witnessai-raises-58m-to-lock-down-the-agentic-ai-era-before-enterprises-lose-control/).
- **500%+ ARR growth** over 12 months; $58M funding led by Sound Ventures (Jan 2026) [TechIntelPro](https://techintelpro.com/news/ai/enterprise-ai/witnessai-raises-58m-launches-agentic-ai-security-features).
- Intelligent **prompt routing** based on risk, cost, or purpose [WitnessAI](https://witness.ai/).

**Distinct advantage:** Single-tenant architecture provides data sovereignty guarantees unusual in SaaS offerings; intent-based behavioral detection for both prompts and agent actions; unified platform for both human employee and agent governance.

---

### Arthur Shield / Arthur AI

**What it protects:** LLM applications (inputs and outputs), plus broader model monitoring across LLMs, tabular, NLP, and computer vision models. Covers PII leakage, hallucinations, prompt injection, and toxic language [Arthur Shield](https://www.arthur.ai/product/shield).

**Deployment model:** Sits between the application layer and deployment layer. Supports SaaS, managed cloud, and on-premises deployment. Model and platform agnostic (works with any LLM and any cloud provider) [Arthur Shield](https://www.arthur.ai/product/shield).

**Notable capabilities:**
- **Arthur Shield** (launched May 2023): One of the first LLM firewalls. Validates user prompts and model responses on two endpoints in real time [Arthur Shield](https://www.arthur.ai/product/shield).
- **Hallucination detection** using secondary ML classifiers that evaluate response factual consistency [AppSecSanta - Arthur AI](https://appsecsanta.com/arthur-ai).
- **Bias detection** via active probing across subgroups with configurable fairness thresholds [AppSecSanta - Arthur AI](https://appsecsanta.com/arthur-ai).
- **Agent Discovery & Governance (ADG)** platform (Dec 2025) for discovering, monitoring, and governing agentic AI in production [AppSecSanta - Arthur AI](https://appsecsanta.com/arthur-ai).
- **Open-source tools:** Arthur Bench (LLM evaluation) and Arthur Engine (monitoring and guardrails) [AppSecSanta - Arthur AI](https://appsecsanta.com/arthur-ai).
- **Explainability** via LIME and SHAP algorithms for both LLM and traditional models [AppSecSanta - Arthur AI](https://appsecsanta.com/arthur-ai).
- Free tier available with paid Premium ($60/mo) and Enterprise plans [Toolradar](https://toolradar.com/tools/arthur-ai).

**Distinct advantage:** Broadest model type coverage (LLMs + tabular + NLP + computer vision); combines security with observability, fairness monitoring, and explainability; one of the first movers in LLM firewalls.

---

## Frameworks / Open Source

### NVIDIA NeMo Guardrails

**What it protects:** LLM and agentic AI applications. Provides orchestration for topic control, PII detection, RAG grounding, jailbreak prevention, and multilingual/multimodal content safety [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails).

**Deployment model:** Open-source Python library (Apache 2.0). Embeds directly into application code. Integrates with LangChain, LangGraph, and LlamaIndex. Supports multi-agent deployments. GPU-accelerated for low-latency performance [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails).

**Notable capabilities:**
- **Colang** -- a domain-specific language for defining conversational guardrails and dialog flows [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails).
- Works out of the box with **NVIDIA Nemotron safety models** packaged as NIM microservices [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails).
- Supports **content safety, topic control, and jailbreak detection** guardrails [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails).
- Part of the larger **NVIDIA NeMo software suite** for building, monitoring, and optimizing AI agents [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails).
- **Cisco AI Defense integration** -- used as a modular runtime protection component [PR Newswire - Cisco](https://www.prnewswire.com/news-releases/cisco-redefines-security-for-the-agentic-era-with-ai-defense-expansion-and-ai-aware-sase-302683205.html).

**Distinct advantage:** Most widely adopted open-source guardrails framework for production use; Colang DSL provides declarative guardrail definition; GPU acceleration via NVIDIA NIM ecosystem; endorsed by Cisco for enterprise integration.

---

### Guardrails AI

**What it protects:** LLM inputs and outputs. Focuses on risk detection/mitigation and structured data generation from LLMs [Guardrails AI](https://guardrailsai.com/).

**Deployment model:** Open-source Python framework (Apache 2.0, 6.7k GitHub stars). Can be deployed as a standalone Flask-based REST API server. Also offers commercial platform (Guardrails Hub, Snowglobe) [GitHub - Guardrails AI](https://github.com/guardrails-ai/guardrails).

**Notable capabilities:**
- **Guardrails Hub** -- a collection of pre-built validators (called "validators") for specific risk types: toxicity, competitor mentions, regex matching, PII detection, and more [PyPI - guardrails-ai](https://pypi.org/project/guardrails-ai/).
- **Guardrails Index** (Feb 2025): First benchmark comparing performance and latency of 24 guardrails across 6 categories [PyPI - guardrails-ai](https://pypi.org/project/guardrails-ai/).
- **Composable Guards** -- multiple validators can be combined into Input and Output Guards [PyPI - guardrails-ai](https://pypi.org/project/guardrails-ai/).
- **Structured data generation** from LLMs using Pydantic models [PyPI - guardrails-ai](https://pypi.org/project/guardrails-ai/).
- **Snowglobe** (commercial): Synthetic data generation for fine-tuning, eval dataset creation, and edge case simulation [Guardrails AI](https://guardrailsai.com/).

**Distinct advantage:** Developer-first approach with composable, community-contributed validators; dual focus on safety/security and structured output reliability; active benchmarking community.

---

### LLM Guard (by Protect AI / now Palo Alto Networks)

**What it protects:** LLM prompts and responses. Covers prompt injection, data leakage, harmful language, PII, secrets, toxicity, code leaks, and more [Protect AI - LLM Guard](https://protectai.com/llm-guard).

**Deployment model:** Open-source Python library (MIT license, 2.8k GitHub stars). Can be deployed as a library or API. Model-agnostic, works with any LLM framework (Azure OpenAI, Bedrock, LangChain, etc.) [GitHub - LLM Guard](https://github.com/protectai/llm-guard).

**Notable capabilities:**
- **Extensive scanner library:** 15+ input scanners (Anonymize, BanCode, BanCompetitors, BanTopics, Gibberish, InvisibleText, Language, PromptInjection, Regex, Secrets, Sentiment, TokenLimit, Toxicity) and 17+ output scanners (Bias, Deanonymize, FactualConsistency, JSON, MaliciousURLs, NoRefusal, Relevance, Sensitive, URLReachability) [GitHub - LLM Guard](https://github.com/protectai/llm-guard).
- **CPU-optimized** with 5x lower inference costs on CPU vs GPU [Protect AI - LLM Guard](https://protectai.com/llm-guard).
- **2.5M+ downloads** [Protect AI - LLM Guard](https://protectai.com/llm-guard).
- Highly modular -- scanners can be independently selected and configured [GitHub - LLM Guard](https://github.com/protectai/llm-guard).

**Distinct advantage:** Most comprehensive set of individual scanners in any open-source tool; CPU-optimized for cost-effective deployment; battle-tested with 2.5M+ downloads.

---

### Rebuff (by Protect AI -- Archived)

**What it protects:** Focused specifically on prompt injection detection. Multi-layered approach to detecting prompt injection attacks [GitHub - Rebuff](https://github.com/protectai/rebuff).

**Deployment model:** Open-source (Apache 2.0). Python SDK and JavaScript SDK. Self-hostable with Supabase, OpenAI, and Pinecone/Chroma backend [GitHub - Rebuff](https://github.com/protectai/rebuff).

**Notable capabilities:**
- **Multi-layered detection:** Heuristic analysis, LLM-based detection, and vector database similarity search against known injection patterns [GitHub - Rebuff](https://github.com/protectai/rebuff).
- **Self-hardening:** Learns from detected attacks to improve future detection [GitHub - Rebuff](https://github.com/protectai/rebuff).
- **Archived in May 2025** -- no longer actively maintained; functionality absorbed into broader Protect AI / Prisma AIRS platform [GitHub - Rebuff](https://github.com/protectai/rebuff).

**Distinct advantage:** Innovative self-hardening concept where the detector improves from each attack; however, now archived and superseded by commercial alternatives.

---

### Vigil

**What it protects:** LLM prompts. Focused on prompt injection detection. Note: Vigil is a smaller community project with limited recent activity compared to the other tools listed here. It provides a simple prompt injection scanner using heuristics and ML models. Due to its limited scope and community size, detailed sourcing is limited. Its last significant activity was in 2024.

**Deployment model:** Open-source Python library/API.

**Notable capabilities:**
- Prompt injection detection using multiple detection methods (heuristic, model-based).
- Lightweight and self-hostable.

**Distinct advantage:** Minimal footprint for basic prompt injection detection; suitable for prototyping but less comprehensive than alternatives like LLM Guard or NeMo Guardrails.

---

## Comparison Table

| Vendor/Product | Category | Protects | Deployment Model | Key Differentiator |
|---|---|---|---|---|
| **Azure AI Content Safety / Prompt Shields** | Cloud Platform | Model I/O, content | API, container, on-device | Spotlighting for XPIA; on-device deployment; Defender integration |
| **AWS Bedrock Guardrails** | Cloud Platform | Model I/O, content, PII | Cloud API (ApplyGuardrail) | ApplyGuardrail API for any model; automated reasoning policies |
| **Google Cloud Model Armor** | Cloud Platform | Model I/O, agents, content | Cloud API | Malware/URL scanning in prompts; free tier; SDP integration |
| **Cloudflare for AI** | Cloud Platform | Apps, employees, models | Edge proxy/gateway | Edge network deployment; Shadow AI discovery; AI-SPM |
| **Palo Alto Prisma AIRS** | Security Vendor | Apps, models, agents, data | Inline firewall, API/SDK, cloud | Most comprehensive platform; model scanning; Protect AI integration |
| **Cisco AI Defense** | Security Vendor | Models, apps, agents, data | Network-layer, cloud | AI BOM; MCP Catalog; Cisco networking footprint; Talos TI |
| **F5 AI Gateway / Guardrails** | Security Vendor | Models, apps, agents, APIs | Containerized gateway | Red team-to-guardrail loop; CalypsoAI integration; ADSP |
| **Fortinet FortiAI** | Security Vendor | Models, LLM data, workloads | Fabric-integrated (inline) | Embedded in existing Security Fabric; 6,500+ AI URL detection |
| **Check Point AI Security** | Security Vendor | Workforce, apps, agents | Infinity platform (inline/API) | Gandalf adversarial engine (85M+ real attacks); Lakera + Cyata |
| **Prompt Security** | Startup (acquired) | Employee AI, apps, code, agents | SaaS, on-prem, browser, API | Browser extension for Shadow AI; MCP Gateway; now in SentinelOne |
| **Lakera Guard** | Startup (acquired) | LLM apps, agents | Cloud API | Gandalf crowdsourced TI; <0.5% false positives; now in Check Point |
| **Protect AI** | Startup (acquired) | Model supply chain, apps | Cloud platform, API | Model-level scanning; AI BOM; now in Prisma AIRS |
| **HiddenLayer** | Startup | Models, agents | Cloud API (non-invasive) | MLDR (ML Detection & Response); gov/defense focus |
| **Lasso Security** | Startup | Apps, agents, tools | Platform/CI integration | Intent Security for agents; behavioral analysis |
| **CalypsoAI** | Startup (acquired) | Models, apps (inference) | Inference-layer perimeter | Defense/national security heritage; now in F5 ADSP |
| **WitnessAI** | Startup | LLMs, apps, agents, employees | Network proxy (single-tenant) | 99%+ injection detection; single-tenant; intent-based policies |
| **Arthur AI** | Startup | LLMs, tabular, NLP, CV models | SaaS, on-prem, cloud-agnostic | Broadest model type coverage; fairness/explainability; free tier |
| **NVIDIA NeMo Guardrails** | Open Source | LLM/agent apps | Python library (app-embedded) | Colang DSL; GPU acceleration; NVIDIA NIM integration |
| **Guardrails AI** | Open Source | LLM I/O | Python library, REST API | Composable validators; Hub ecosystem; Guardrails Index benchmark |
| **LLM Guard** | Open Source | LLM prompts/responses | Python library, API | 30+ scanners; CPU-optimized; 2.5M+ downloads |
| **Rebuff** | Open Source (archived) | Prompt injection only | Self-hosted | Self-hardening detection; archived May 2025 |
| **Vigil** | Open Source | Prompt injection only | Python library | Lightweight; limited scope |

---

## M&A Consolidation Summary

The AI security startup landscape has undergone dramatic consolidation since mid-2025:

| Acquirer | Target | Date | Price | Integration |
|---|---|---|---|---|
| Palo Alto Networks | Protect AI | July 2025 | Undisclosed | Prisma AIRS model scanning & AI BOM |
| SentinelOne | Prompt Security | Aug 2025 | ~$250M | Singularity platform GenAI security |
| F5 | CalypsoAI | Sept 2025 | $180M | ADSP AI Guardrails & Red Team |
| Check Point | Lakera | Sept 2025 | ~$190M | Infinity platform AI security |
| Cisco | Robust Intelligence | Oct 2024 | Undisclosed | AI Defense (algorithmic red teaming) |

Sources: [Help Net Security](https://www.helpnetsecurity.com/2025/10/29/palo-alto-networks-launches-prisma-airs-2-0-to-deliver-end-to-end-security-across-the-ai-lifecycle/), [AppSecSanta](https://appsecsanta.com/prompt-security), [Intelligent CISO](https://www.intelligentciso.com/2025/09/15/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises/), [CSO Online](https://www.csoonline.com/article/4058653/check-point-acquires-lakera-to-build-a-unified-ai-security-stack.html), [Cisco](https://www.cisco.com/site/us/en/products/security/ai-defense/robust-intelligence-is-part-of-cisco/index.html).

---

## Key Market Observations

**1. Universal agentic AI pivot:** Every vendor -- without exception -- has expanded or pivoted toward agentic AI security capabilities since late 2025. This includes MCP server scanning, tool-use governance, agent identity management, and multi-step behavioral monitoring. This reflects the rapid enterprise adoption of AI agents (65% of organizations researching, piloting, or deploying agentic AI per Futurum Group's 1H 2026 survey) [OpenClawAI](https://openclawai.io/blog/palo-alto-prisma-airs-agentic-ai-security-rsac-2026).

**2. Deployment model spectrum:** The market spans five primary deployment patterns:
- **API/SDK** (Azure, AWS, Lakera, Arthur): Lowest friction, developer-integrated
- **Cloud proxy/gateway** (Cloudflare, F5, WitnessAI): Network-level interception
- **Inline network firewall** (Palo Alto, Cisco, Fortinet): Deepest traffic visibility
- **Browser extension** (Prompt Security, Cloudflare): Employee AI usage monitoring
- **Embedded library** (NeMo Guardrails, Guardrails AI, LLM Guard): Application-code level

**3. Red teaming as table stakes:** Automated AI red teaming has become a baseline capability. Every major vendor now offers continuous adversarial testing (Palo Alto with 500+ attacks, F5 with 10,000+ monthly, Cisco with 200+ subcategories, WitnessAI with multimodal/reinforcement-learning attacks).

**4. Open source as proving ground:** Open-source projects (LLM Guard, NeMo Guardrails, Guardrails AI) serve as community-validated foundations. Several have been absorbed into commercial products (LLM Guard into Prisma AIRS, NeMo Guardrails integrated with Cisco AI Defense).

**5. Remaining independent startups under pressure:** With the major acquisitions complete, independent startups like HiddenLayer, WitnessAI, Lasso Security, and Arthur AI face a market where most enterprises will prefer platform-integrated solutions. HiddenLayer ($50M+ raised, strong gov/defense positioning) and WitnessAI ($85M+ total raised, 500%+ ARR growth) appear best positioned to remain independent or command premium acquisition prices.

---

## Conclusion

The AI firewall and runtime security market in 2026 is defined by three forces: **consolidation** (major security vendors have absorbed nearly every leading AI-native startup), **convergence** (all vendors protect against the same core threats -- prompt injection, data leakage, jailbreaks, toxic content), and **expansion** (the category is rapidly growing from LLM guardrails to full agentic AI lifecycle security).

For organizations evaluating solutions, the primary decision axis is ecosystem alignment: cloud platform customers benefit most from native tools (Azure, AWS, GCP), organizations with existing security vendor relationships can extend their stack (Palo Alto, Cisco, F5, Fortinet, Check Point), and those wanting vendor-agnostic or developer-centric approaches should consider independent startups (HiddenLayer, WitnessAI, Arthur AI) or open-source frameworks (NeMo Guardrails, Guardrails AI, LLM Guard).

The market remains highly dynamic, with agentic AI security capabilities evolving quarterly and no single vendor yet commanding a dominant market share in this emerging category.

---

## Sources

1. [Azure AI Content Safety Blog](https://azure.microsoft.com/en-us/blog/enhance-ai-security-with-azure-prompt-shields-and-azure-ai-content-safety/)
2. [Azure AI Content Safety Catalog](https://ai.azure.com/catalog/models/Azure-AI-Content-Safety)
3. [AWS Bedrock Guardrails Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
4. [Google Cloud Model Armor](https://cloud.google.com/security/products/model-armor)
5. [Cloudflare for AI Press Release](https://www.cloudflare.com/en-gb/press/press-releases/2025/cloudflare-introduces-cloudflare-for-ai/)
6. [Cloudflare AI Week 2025 - Lushbinary](https://lushbinary.com/blog/cloudflare-ai-week-2025-everything-released/)
7. [Prisma AIRS Documentation](https://docs.paloaltonetworks.com/ai-runtime-security)
8. [NAND Research - Prisma AIRS](https://nand-research.com/research-note-palo-alto-networks-prisma-airs-for-ai-protection/)
9. [Help Net Security - Prisma AIRS 2.0](https://www.helpnetsecurity.com/2025/10/29/palo-alto-networks-launches-prisma-airs-2-0-to-deliver-end-to-end-security-across-the-ai-lifecycle/)
10. [OpenClawAI - Prisma AIRS at RSAC 2026](https://openclawai.io/blog/palo-alto-prisma-airs-agentic-ai-security-rsac-2026)
11. [Cisco AI Defense Data Sheet](https://www.cisco.com/c/en/us/products/collateral/security/ai-defense/ai-defense-ds.html)
12. [PR Newswire - Cisco AI Defense Expansion](https://www.prnewswire.com/news-releases/cisco-redefines-security-for-the-agentic-era-with-ai-defense-expansion-and-ai-aware-sase-302683205.html)
13. [Cisco - Robust Intelligence](https://www.cisco.com/site/us/en/products/security/ai-defense/robust-intelligence-is-part-of-cisco/index.html)
14. [F5 AI Gateway Press Release](https://www.f5.com/company/news/press-releases/f5-ai-gateway-manage-secure-traffic-application-demands)
15. [Security MEA - F5 AI Guardrails](https://securitymea.com/2026/01/28/f5-unveils-ai-guardrails-and-ai-red-team-to-secure-enterprise-ai-systems/)
16. [Nasdaq - F5 AppWorld 2026](https://www.nasdaq.com/press-release/f5-advances-enterprise-application-security-ai-and-post-quantum-era-2026-03-11)
17. [Intelligent CISO - F5 CalypsoAI Acquisition](https://www.intelligentciso.com/2025/09/15/f5-to-acquire-calypsoai-to-bring-advanced-ai-guardrails-to-large-enterprises/)
18. [Fortinet FortiAI Product Page](https://www.fortinet.com/products/fortiai-secure)
19. [MSSP Alert - Fortinet FortiAI](https://www.msspalert.com/news/fortinet-expands-ai-capabilities-across-security-fabric-platform)
20. [Check Point AI Security](https://www.checkpoint.com/ai-security/)
21. [CSO Online - Check Point Lakera Acquisition](https://www.csoonline.com/article/4058653/check-point-acquires-lakera-to-build-a-unified-ai-security-stack.html)
22. [Security Solutions Media - Check Point Acquisitions](https://www.securitysolutionsmedia.com/2026/02/16/check-point-unveils-ai-security-strategy-reinforces-platform-with-three-strategic-acquisitions/)
23. [AppSecSanta - Prompt Security](https://appsecsanta.com/prompt-security)
24. [Prompt Security Website](https://prompt.security/)
25. [Lakera Guard Documentation](https://docs.lakera.ai/guard)
26. [Palo Alto - Protect AI Acquisition](https://www.paloaltonetworks.com/company/press/2025/palo-alto-networks-completes-acquisition-of-protect-ai)
27. [HiddenLayer - Model Scanning](https://www.hiddenlayer.com/solutions/model-scanning)
28. [Lasso Security Website](https://www.lasso.security/)
29. [Gartner Peer Insights - CalypsoAI](https://www.gartner.com/reviews/market/ai-security-and-anomaly-detection/vendor/calypsoai)
30. [NSFOCUS - CalypsoAI RSAC 2025](https://nsfocusglobal.com/rsac-2025-innovation-sandbox-calypsoai-a-practical-path-and-trust-foundation-for-forging-ai-system-security-protection-systems/)
31. [WitnessAI Website](https://witness.ai/)
32. [PR Newswire - WitnessAI Products](https://www.prnewswire.com/news-releases/witnessai-announces-automated-red-teaming-and-next-generation-ai-firewall-protection-for-enterprise-llms-and-ai-applications-302534128.html)
33. [TechIntelPro - WitnessAI Funding](https://techintelpro.com/news/ai/enterprise-ai/witnessai-raises-58m-launches-agentic-ai-security-features)
34. [Arthur Shield Product Page](https://www.arthur.ai/product/shield)
35. [AppSecSanta - Arthur AI](https://appsecsanta.com/arthur-ai)
36. [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails)
37. [Guardrails AI Website](https://guardrailsai.com/)
38. [GitHub - Guardrails AI](https://github.com/guardrails-ai/guardrails)
39. [Protect AI - LLM Guard](https://protectai.com/llm-guard)
40. [GitHub - LLM Guard](https://github.com/protectai/llm-guard)
41. [GitHub - Rebuff](https://github.com/protectai/rebuff)
42. [Intelligent CISO - F5 Data Protection](https://www.intelligentciso.com/2025/08/05/f5-adds-data-protection-features-for-ai-workloads-and-encrypted-traffic/)
43. [Security Info Watch - Fortinet](https://www.securityinfowatch.com/cybersecurity/press-release/55280685/fortinet-fortinet-expands-fortiai-across-its-security-fabric-platform)
