# Microsoft AI Security: Tooling, Operating Model, and RACI

*A reference guide for enterprises adopting Azure AI / Foundry, covering Content Safety, PII protection, Evaluations, and Red Teaming — plus the org structure, lifecycle integration, and accountability model needed to make it real.*

> **Audience:** AI platform leads, CISOs, Responsible AI champions, cloud architects.
> **Frameworks referenced:** Microsoft Responsible AI Standard v2, NIST AI RMF 1.0 + GenAI Profile (AI 600-1), ISO/IEC 42001:2023, OWASP LLM Top 10.

---

## Part 1 — Executive Summary

Microsoft's AI security stack is organized around **four operational pillars** wrapped by a **governance layer** and a **posture-management layer**:

| Layer | Capability | Microsoft product / tool |
|---|---|---|
| **Runtime guardrail** | Detect and block harmful, jailbreaking, ungrounded, copyrighted content | **Azure AI Content Safety** (Prompt Shields, Groundedness Detection, Protected Material, Custom Categories) |
| **Data guardrail** | Discover, classify, redact PII; prevent sensitive-data exfil into prompts | **Azure AI Language PII**, **Microsoft Purview DSPM for AI**, **Presidio** (OSS) |
| **Quality gate** | Score model output for groundedness, relevance, safety; gate CI; sample production | **Azure AI Foundry Evaluations** (`azure-ai-evaluation` SDK), **continuous evaluation** in Foundry Observability |
| **Adversarial gate** | Probe the system with jailbreak / injection / exfiltration attacks | **PyRIT** (OSS), **Azure AI Foundry AI Red Teaming Agent** |
| **Governance** | Principles, impact assessments, accountability | **Microsoft Responsible AI Standard v2**, **NIST AI RMF**, **ISO/IEC 42001** |
| **Posture & SOC** | AI BOM discovery, attack-path analysis, prompt-injection alerts into XDR | **Microsoft Defender for Cloud AI-SPM**, **Threat Protection for AI Services** |

The architectural pattern: **models run in Foundry / Azure OpenAI; Content Safety + Prompt Shields are the runtime guardrail; Purview + Language PII are the data guardrail; `azure-ai-evaluation` is the quality gate; PyRIT and the AI Red Teaming Agent are the adversarial gate; Defender for Cloud is the posture and SOC integration.**

---

## Part 2 — The Four Pillars in Detail

### Pillar 1 — Azure AI Content Safety

**What it is.** A managed Azure AI service that screens text, images, and multimodal inputs for harmful, jailbreaking, ungrounded, or copyrighted content via REST/SDK. It is the runtime layer used by Azure OpenAI's built-in content filters and by custom apps that need policy enforcement on inputs and outputs.

**Detection capabilities:**

- **Four harm categories** — Hate, Sexual, Violence, Self-Harm — scored 0–7, surfaced at default discrete levels 0/2/4/6.
- **Prompt Shields** — detects (a) **user prompt attacks** (jailbreaks against the system prompt) and (b) **indirect (document) prompt injection attacks** (XPIA) embedded in retrieved content.
- **Groundedness Detection** — flags model outputs not supported by provided sources; optional **reasoning** mode (LLM explanation) and **correction** mode (rewrites the ungrounded passage).
- **Protected Material detection** — copyrighted text (lyrics, news, recipes) and code (GitHub-style license-risk matches).
- **Custom Categories** — customer-defined policies trained from a few hundred examples.

**Deployment patterns:**
1. **REST/SDK direct** on inputs and outputs from any app.
2. **Azure OpenAI built-in content filters** — applied automatically at default Medium severity, configurable per prompt and per completion.
3. **Foundry-integrated guardrails** — applied to deployed models and agents.

**Pricing.** Standard tier ≈ **$0.38 per 1,000 text records**, **$0.75 per 1,000 image records**; free tier 5,000 records/month per feature. Groundedness reasoning is billed as Azure OpenAI tokens.

**Lifecycle slot:** Runtime — every prompt and every completion.
**Primary owner:** AI platform team, with security co-owning the policy thresholds.

---

### Pillar 2 — PII Detection and Protection

**Azure AI Language — PII detection.** Built-in entity recognizer for tens of PII entity types (Person, Phone, Email, Address, ABA Routing, IBAN, IP, SSN, plus health and government IDs in the conversational PII model). The 2025-11-15 preview API introduces explicit **`redactionPolicies`**:
- `NoMask` — detect-only.
- `CharacterMask` — replace each character with `*` (default).
- `EntityMask` — replace with entity-type token, e.g. `[PERSON]`.
- `SyntheticReplacement` — replace with realistic synthetic data so downstream parsers still work.

**Microsoft Purview — DSPM for AI.** Discovers AI usage across Microsoft 365 Copilot, Azure AI Foundry, and third-party AI apps. Chains together **Sensitivity Labels**, **DLP for Generative AI**, **Insider Risk Management**, and **Communication Compliance** — blocks sensitive prompts, audits prompt/response activity, and enforces retention. Foundry projects ingest Purview policies so label-based DLP travels with data into RAG indexes and agent tools.

**PII handling in Azure OpenAI / Foundry.** Customer prompts and completions are processed in the customer's Azure tenant and region. Abuse-monitoring data retention can be opted out of via the Limited Access form. Data residency follows the resource region.

**Microsoft Presidio (open source, MIT-licensed).** Python SDK for de-identification:
- **Analyzer** — built-in + custom recognizers, NER + regex + checksum + context.
- **Anonymizer** — mask, replace, hash, redact, encrypt.
- **Image Redactor** — for screenshots and scanned documents.
Use Presidio for on-prem / pre-cloud scrubbing pipelines and for building deterministic test corpora that don't carry real PII into Foundry.

**Lifecycle slot:** Data ingestion (Presidio, Purview classification) → runtime (Language PII, Purview DLP) → post-hoc audit (Purview activity explorer).
**Primary owner:** Data security / compliance, with platform team operating the connectors.

---

### Pillar 3 — AI Evaluations

The `azure-ai-evaluation` Python SDK (`pip install azure-ai-evaluation`) is the supported integration point. Foundry surfaces an **Evaluations** tab and exposes APIs for CI/CD and continuous production sampling.

**Built-in evaluator families:**

| Family | Evaluators |
|---|---|
| Quality | Coherence, Fluency, QA |
| RAG | Groundedness, GroundednessPro (Content-Safety-backed), Relevance, Retrieval, Response Completeness, Document Retrieval |
| Similarity | Similarity (LLM-judge) + classical F1, BLEU, ROUGE, METEOR, GLEU |
| Agent | Intent Resolution, Tool Call Accuracy, Task Adherence |
| **Risk & Safety** | Violence, Sexual, Hate/Unfairness, Self-Harm, Protected Material, Indirect Attack (XPIA), Code Vulnerability, Ungrounded Attributes; preview: Prohibited Actions, Sensitive Data Leakage |

Risk/safety evaluators emit both a categorical label and a 0–7 severity, mirroring Content Safety taxonomy.

**Custom evaluators.** Three first-class authoring models:
1. **Callable Python evaluators** — any function returning a score.
2. **Prompt-based evaluators** — Jinja templates with a model judge.
3. **Composite evaluators** — combinations of the above.

**CI/CD integration.** Microsoft publishes **GitHub Actions** and **Azure DevOps** templates that run evaluation suites on PRs and gate merges on score thresholds. Evals become unit tests for LLM apps.

**Continuous (production) evaluation.** Foundry Observability scores sampled live traces against the same evaluator catalog and publishes dashboards / alerts. Required for drift detection and for catching regressions caused by model swaps, prompt edits, or upstream data changes.

**Lifecycle slot:** Pre-deployment (CI gates) → pre-release (offline benchmark suites) → continuously in production (sampling).
**Primary owner:** AI engineering / quality, with thresholds co-owned by the Responsible AI champion.

---

### Pillar 4 — AI Red Teaming

**PyRIT (Python Risk Identification Tool).** Open-source framework from the Microsoft AI Red Team. Component model:

- **Targets** — wrappers for the system under test (Azure OpenAI, Foundry, OpenAI, Anthropic, HuggingFace, custom HTTP, even GUI/voice).
- **Datasets** — seed prompts (HarmBench, AdvBench, custom).
- **Converters** — obfuscation transforms (Base64, ROT13, ASCII art, translation, persuasion).
- **Scorers** — LLM-judge or Content-Safety-backed labels for the response.
- **Attack strategies** — multi-turn algorithms including **Crescendo**, **Tree of Attacks with Pruning (TAP)**, **Skeleton Key**, **PAIR**.
- **Memory** — stores conversations for replay and analysis.

**Foundry AI Red Teaming Agent.** GA managed service that runs PyRIT-backed scans against a Foundry deployment or agent and produces an **Attack Success Rate (ASR)** report aligned to NIST AI RMF **Govern / Map / Measure / Manage**. Scans run cloud-side at scale or locally via the SDK.

**Methodology — what to attack.** From the Microsoft AI Red Team's published lessons across 100+ generative AI products, the relevant attack categories beyond simple jailbreaks:
- Direct **prompt injection**
- **Indirect (cross-prompt) prompt injection (XPIA)**
- **Data exfiltration via tool-use chains**
- **Model denial-of-service** via amplification prompts
- **Supply-chain risks** in fine-tunes and adapters
- **Traditional security flaws** in the surrounding agent harness (SSRF, credential leakage, broken authZ on tool calls)
- **Spotlighting** — explicitly tagging untrusted retrieved content — is a documented prompt-engineering defense for XPIA.

**Pre-deployment vs. continuous.** Pre-deployment uses exhaustive PyRIT scans, manual red-team exercises, and a Responsible AI Impact Assessment. Continuous testing schedules Red Teaming Agent scans against staging on every model swap or system-prompt change, and feeds signals into Defender XDR.

**Lifecycle slot:** Pre-release (mandatory gate) + continuous on every change.
**Primary owner:** AI Red Team / product security, with platform team running the harness.

---

### Cross-cutting — Governance and Posture

**Microsoft Responsible AI Standard v2.** Six principles — **Fairness, Reliability & Safety, Privacy & Security, Inclusiveness, Transparency, Accountability** — operationalized via Impact Assessments, sensitive-uses review, and "Responsible AI by Design" gates.

**NIST AI RMF + GenAI Profile (AI 600-1, July 2024).** 200+ recommended actions under **Govern / Map / Measure / Manage**, with explicit attention to confabulation, dangerous content, data privacy, and information security. Foundry's Red Teaming Agent and risk/safety evaluators emit reports labeled by RMF function.

**ISO/IEC 42001:2023.** First international, certifiable AI Management System (AIMS) standard, modeled on ISO 27001's plan-do-check-act. For regulated customers, mapping the AI program to 42001 controls is increasingly an audit expectation.

**Microsoft Defender for Cloud — AI Security Posture Management (AI-SPM).** Defender CSPM (paid) discovers an **AI bill of materials** across Azure OpenAI, Foundry, Azure ML, **Amazon Bedrock**, and **Google Vertex AI**, surfaces vulnerabilities, and computes attack paths from internet-exposed endpoints to grounding data.

**Threat Protection for AI Services (GA).** Pipes Prompt Shields signals into **Microsoft Defender XDR**, raises alerts on jailbreak attempts and credential theft. Ships with a 30-day free trial up to 75 billion tokens scanned.

---

## Part 3 — Setting Up the Organization

### 3.1 Roles & Resources

A defensible AI security program needs the following roles. In smaller orgs, one person may wear several hats; the responsibilities still need a name on them.

| # | Role | Core responsibility |
|---|---|---|
| 1 | **Chief AI Officer / AI Governance Lead** | Owns the AI strategy, policy, and the Responsible AI council. Final accountability for risk acceptance. |
| 2 | **Responsible AI Champion / Officer** | Day-to-day owner of the RAI Standard. Runs Impact Assessments, sensitive-use triage, transparency notes. |
| 3 | **AI / ML Engineer** | Builds the model integration, prompts, agent workflows. Implements evaluators in CI. |
| 4 | **Data Scientist** | Owns dataset curation, training/fine-tuning experiments, metric definitions. |
| 5 | **MLOps / LLMOps Engineer** | Owns the deployment pipeline, monitoring, continuous evaluation, model registry, drift detection. |
| 6 | **AI Security Engineer (Red Team)** | Adversarial testing, PyRIT runs, jailbreak research, manual probes, threat modeling for AI features. |
| 7 | **Privacy Officer / DPO** | PII discovery, DPIA, data residency, cross-border issues, opt-outs, retention. |
| 8 | **Legal & Compliance** | Regulatory fit (EU AI Act, GDPR, sectoral rules), contracts, IP indemnity, copyright. |
| 9 | **Security Architect** | Reference architectures, IAM, network controls, key management, secrets in agent tools. |
| 10 | **SOC / Detection Engineering** | Operates Defender XDR alerts for AI, writes detections, runs IR for AI incidents. |
| 11 | **Product Manager** | Owns the use case, business value, acceptance criteria, user-facing transparency, opt-outs. |
| 12 | **Business / Domain SME** | Provides ground-truth labels, evaluates outputs in domain language, signs off on accuracy bar. |
| 13 | **End-user feedback channel** | Thumbs up/down + structured incident report path; closes the loop into eval datasets. |

### 3.2 Operating Model

Three patterns are common; **hub-and-spoke is the default recommendation** for enterprises with more than a handful of AI teams:

- **Centralized.** A single AI platform team builds and approves everything. Scales poorly past ~10 use cases.
- **Federated.** Each business unit builds its own. Fast, but governance and red-team coverage are uneven.
- **Hub-and-spoke (recommended).** A central **AI Center of Excellence (CoE)** owns the platform, the RAI Standard, the eval/red-team tooling, and the intake/triage process. Spokes (business unit AI teams) build use cases on the platform under CoE guardrails. This is how Microsoft itself operates internally.

**Intake / triage process** (the entry point):
1. **Use-case proposal** (1-page) submitted by the product manager.
2. **Risk classification** — Low / Medium / High / Restricted, based on impact (people affected, decisions made, sensitive data, regulated domain).
3. **Lightweight gate for Low**, **full RAI Impact Assessment for Medium+**, **sensitive-use review board for High/Restricted**.
4. **Output:** approved scope, mandatory controls, eval thresholds, red-team requirement, monitoring plan.

**Phased rollout (Crawl / Walk / Run):**

| Phase | Scope | Controls in place |
|---|---|---|
| **Crawl** (months 0–3) | One pilot use case, low risk | Manual RAI Impact Assessment; default Azure OpenAI content filters; manual eval; one-shot red team before launch. |
| **Walk** (months 3–9) | 3–5 use cases | CoE stood up; standardized RAI Impact Assessment; `azure-ai-evaluation` in CI; Purview DSPM for AI; Defender CSPM AI-SPM enabled; quarterly red team. |
| **Run** (9+ months) | Multiple BUs, regulated workloads | ISO 42001 alignment; continuous evaluation in production; AI Red Teaming Agent on every promotion; Threat Protection for AI in Defender XDR with documented IR runbooks; model card registry. |

### 3.3 Lifecycle Stages and Who Is Involved

Mapped to NIST AI RMF (**Govern / Map / Measure / Manage**) and the Microsoft AI development lifecycle.

| # | Stage | Primary owner | Key activities |
|---|---|---|---|
| 1 | **Ideation / intake & risk classification** | Product Manager + RAI Champion | Use-case proposal, business case, risk tier |
| 2 | **Design — threat model + RAI Impact Assessment** | Security Architect + RAI Champion + AI Engineer | Threat model (incl. OWASP LLM Top 10), data flow, failure modes, mitigations |
| 3 | **Data preparation** | Data Scientist + Privacy Officer | PII scan (Presidio / Language PII), licensing review, sensitivity labels |
| 4 | **Model selection / fine-tuning** | AI Engineer + Data Scientist | Base model choice, fine-tune dataset review, capability vs. risk trade-off |
| 5 | **Offline evaluation** | AI Engineer + Domain SME | Quality + RAG + safety evaluators in CI; thresholds approved by RAI Champion |
| 6 | **Pre-release red teaming** | AI Security Engineer | PyRIT scans, AI Red Teaming Agent run, manual probes, ASR report |
| 7 | **Release approval** | Chief AI Officer / RAI council | Sign-off on residual risk, transparency note, model card |
| 8 | **Deployment** | MLOps + Security Architect | Guardrails wired (Content Safety, Prompt Shields), DLP active, network/IAM, secret hygiene |
| 9 | **Production monitoring & continuous evaluation** | MLOps + SOC | Continuous eval sampling, Defender XDR alerts, drift dashboards |
| 10 | **Incident response** | SOC + AI Security Engineer + Legal | IR runbook, customer comms, root cause, remediation, lessons-learned |
| 11 | **Decommissioning** | MLOps + Privacy Officer | Data deletion, model archival, audit log preservation |

### 3.4 Required Policies & Artifacts

Produce these once at the org level; reuse per use case:

- **AI Acceptable Use Policy** — what employees and apps can/can't do with AI.
- **RAI Impact Assessment template** — risk tiering, intended use, stakeholders, harms analysis, mitigations.
- **Model Card template** — purpose, training data summary, eval scores, known limitations, contacts.
- **AI Incident Response Plan** — definitions (jailbreak, harmful output, exfiltration, hallucination harm), severity matrix, comms tree, regulator notification triggers.
- **Red Team Rules of Engagement** — scope, allowed targets, data handling, reporting cadence.
- **Transparency Notes / User-facing disclosures** — what the system is, what it isn't, opt-out path.
- **Data Classification + Sensitivity Label Map** — bound to Purview labels, drives DLP policy.

---

## Part 4 — RACI Chart

**Legend.** **R** = Responsible (does the work) · **A** = Accountable (one per row, owns outcome) · **C** = Consulted (input before decision) · **I** = Informed (after the fact).

Roles compressed for table width:

| Code | Role |
|---|---|
| CAIO | Chief AI Officer / AI Governance Lead |
| RAI | Responsible AI Champion |
| PM | Product Manager |
| AIE | AI / ML Engineer |
| DS | Data Scientist |
| MLO | MLOps / LLMOps Engineer |
| RT | AI Security Engineer (Red Team) |
| SEC | Security Architect |
| SOC | SOC / Detection Eng |
| PO | Privacy Officer / DPO |
| LEG | Legal & Compliance |
| SME | Business / Domain SME |

| # | Activity | CAIO | RAI | PM | AIE | DS | MLO | RT | SEC | SOC | PO | LEG | SME |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Define org AI strategy & risk appetite | A | R | C | I | I | I | C | C | C | C | C | I |
| 2 | Maintain AI Acceptable Use Policy | A | R | C | I | I | I | C | C | I | C | C | I |
| 3 | Use-case intake & risk classification | I | A | R | C | C | I | C | C | I | C | C | C |
| 4 | RAI Impact Assessment | I | A | R | C | C | I | C | C | I | C | C | C |
| 5 | Threat model the AI feature (OWASP LLM Top 10) | I | C | I | R | I | C | C | A | C | I | I | I |
| 6 | Data classification & sensitivity labelling (Purview) | I | C | C | C | R | C | I | C | I | A | C | I |
| 7 | PII scan of training / RAG data (Presidio, Language PII) | I | C | I | C | R | C | I | C | I | A | C | I |
| 8 | Configure Content Safety filters & Prompt Shields | I | C | C | R | I | C | C | A | I | I | I | I |
| 9 | Configure Purview DSPM for AI / DLP for GenAI | I | C | I | C | I | C | I | C | I | A | C | I |
| 10 | Author offline evaluators in `azure-ai-evaluation` | I | C | C | A/R | C | C | C | I | I | I | I | C |
| 11 | Approve eval thresholds for release gating | I | A | C | R | C | C | C | C | I | I | C | C |
| 12 | Pre-release red team (PyRIT + Foundry Red Team Agent) | I | C | I | C | I | C | A/R | C | I | I | I | I |
| 13 | Release approval / sign-off | A | R | C | I | I | I | C | C | I | C | C | C |
| 14 | Deploy with guardrails (IAM, network, secrets, filters) | I | I | I | C | I | A/R | I | C | I | I | I | I |
| 15 | Run continuous evaluation in production | I | C | I | C | I | A/R | I | I | C | I | I | C |
| 16 | Operate Defender for Cloud AI-SPM | I | I | I | I | I | C | C | A | R | I | I | I |
| 17 | Triage alerts from Threat Protection for AI / XDR | I | I | I | C | I | C | C | C | A/R | I | I | I |
| 18 | Triage AI incident & invoke IR runbook | I | C | C | C | I | C | R | C | A | C | C | I |
| 19 | Regulator / customer notification post-incident | A | C | C | I | I | I | I | I | C | C | R | I |
| 20 | Periodic red team & control re-test | I | C | I | C | I | C | A/R | C | C | I | I | I |
| 21 | Model card maintenance | I | C | C | A/R | C | C | I | I | I | I | I | C |
| 22 | Decommission model & purge data | I | C | I | C | C | A/R | I | C | I | C | C | I |

---

## Part 5 — Reference Architecture (one-paragraph version)

Deploy models in **Azure AI Foundry / Azure OpenAI**. Wire **Azure AI Content Safety** (with Prompt Shields, Groundedness Detection, and Protected Material) as the runtime guardrail on every input and output. Layer **Azure AI Language PII** and **Microsoft Purview DSPM for AI** as the data guardrail on ingestion, retrieval, and prompt paths. Gate every PR with `azure-ai-evaluation` running quality + risk/safety evaluators in **GitHub Actions / Azure DevOps**, and run **continuous evaluation** on sampled production traffic. Run **PyRIT** locally in pre-merge and the **Foundry AI Red Teaming Agent** before every promotion; archive ASR reports as release evidence. Surface the whole estate in **Defender for Cloud AI-SPM**, and route **Threat Protection for AI** alerts into **Microsoft Defender XDR** where SOC owns triage.

Each layer addresses a distinct failure mode — content policy, data leakage, quality drift, adversarial coercion, posture drift — and they share a common taxonomy (Content Safety harm categories) so signals correlate cleanly across the stack.

---

## Part 6 — Sources

### Microsoft product docs

1. [Azure AI Content Safety overview](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)
2. [Prompt Shields concepts](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)
3. [Groundedness Detection](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness)
4. [Protected Material detection](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/protected-material)
5. [Harm categories](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories)
6. [Azure OpenAI content filtering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter)
7. [Azure AI Content Safety pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/content-safety/)
8. [Azure AI Language PII detection](https://learn.microsoft.com/en-us/azure/ai-services/language-service/personally-identifiable-information/overview)
9. [PII redaction policies (preview)](https://learn.microsoft.com/en-us/azure/ai-services/language-service/personally-identifiable-information/how-to-call)
10. [Microsoft Purview DSPM for AI](https://learn.microsoft.com/en-us/purview/dspm-for-ai)
11. [Microsoft Presidio (OSS)](https://microsoft.github.io/presidio/)
12. [Azure AI Foundry built-in evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/evaluators)
13. [Risk and safety evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/risk-safety-evaluators)
14. [`azure-ai-evaluation` SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
15. [Continuous evaluation in Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/continuous-evaluation-agents)
16. [PyRIT (Microsoft AI Red Team)](https://github.com/Azure/PyRIT)
17. [Azure AI Foundry — AI Red Teaming Agent](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/run-scans-ai-red-teaming-agent)
18. [Microsoft AI Red Team — Lessons from 100 GenAI Products](https://airedteamwhitepapers.blob.core.windows.net/lessonswhitepaper/MS_AIRT_Lessons_eBook.pdf)
19. [Microsoft Responsible AI principles and approach](https://www.microsoft.com/en-us/ai/principles-and-approach)
20. [Defender for Cloud — AI Security Posture Management](https://learn.microsoft.com/en-us/azure/defender-for-cloud/ai-security-posture)
21. [Defender for Cloud — Threat Protection for AI Services](https://learn.microsoft.com/en-us/azure/defender-for-cloud/ai-threat-protection)

### Standards & frameworks

22. [NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework)
23. [NIST AI 600-1: Generative AI Profile (July 2024)](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.600-1.pdf)
24. [ISO/IEC 42001:2023 — AI management systems](https://www.iso.org/standard/81230.html)
25. [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

*Document prepared 2026-05-04. Microsoft AI services iterate quickly — verify preview-feature status (Sensitive Data Leakage evaluator, PII redaction policies API version) on Microsoft Learn before production reliance.*
