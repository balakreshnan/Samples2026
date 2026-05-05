# Microsoft AI Security Tooling: A Principal Architect's Field Guide

## Executive Summary

Microsoft's AI security stack is organized around four operational pillars — **runtime content safety**, **PII discovery and protection**, **pre-production and continuous evaluation**, and **adversarial red teaming** — anchored by a governance layer (Responsible AI Standard, NIST AI RMF, ISO/IEC 42001) and a posture-management layer (Microsoft Defender for Cloud AI-SPM and Threat Protection for AI). For a Principal CSA, the architectural pattern to internalize is: **Foundry / Azure OpenAI is where models run; Content Safety is the runtime guardrail; Purview + Language PII are the data guardrail; `azure-ai-evaluation` is the quality gate; PyRIT and the AI Red Teaming Agent are the adversarial gate; Defender for Cloud is the posture and SOC integration.** Confidence in the technical claims below is **HIGH** — every load-bearing fact is sourced to learn.microsoft.com, microsoft.com, NIST, or ISO primary references.

![Architecture](https://github.com/balakreshnan/Samples2026/blob/main/ProServIndustry/images/microsoft-ai-security-four-pillars-16x9.jpg, "Architecture")
---

## Pillar 1 — Azure AI Content Safety

**What it is.** Azure AI Content Safety is a managed Azure AI service that screens text and images (and now multimodal inputs) for harmful, jailbreaking, ungrounded, or copyrighted content via REST APIs and SDKs ([Azure AI Content Safety overview](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)). It is the runtime guardrail layer used by Azure OpenAI's built-in content filters and by custom apps that need policy enforcement on inputs/outputs.

**Detection capabilities.**

- **Four harm categories** — Hate, Sexual, Violence, Self-Harm — scored on a 0–7 severity scale, commonly surfaced at the default discrete levels 0/2/4/6 ([Harm categories](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories)).
- **Prompt Shields** — detects two attack classes: (a) **user prompt attacks** (jailbreaks against the system prompt) and (b) **indirect (document) prompt injection attacks** embedded in retrieved content ([Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)).
- **Groundedness Detection** — flags model outputs that are not supported by provided sources, with an optional **reasoning** mode (LLM-based explanation of why content is ungrounded) and a **correction** capability that rewrites the ungrounded passage ([Groundedness Detection](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness)).
- **Protected Material detection** — for text (copyrighted lyrics, news, recipes) and code (GitHub-style license-risk matches) ([Protected Material](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/protected-material)).
- **Custom Categories** — customer-defined policies trained from a few hundred examples.

**Deployment options.** Three patterns dominate: (1) **REST/SDK direct calls** on inputs and outputs from any app; (2) **Azure OpenAI built-in content filters**, which apply Content Safety automatically at default Medium severity threshold and are configurable per prompt and per completion ([Azure OpenAI content filtering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter)); (3) **Foundry-integrated guardrails**, applied to deployed models and agents.

**Pricing.** Standard tier is **$0.38 per 1,000 text records** and **$0.75 per 1,000 image records**, with a free tier of 5,000 records/month per feature ([Content Safety pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/content-safety/)). Prompt Shields, Groundedness Detection (without reasoning), and Protected Material are billed under the same record model; Groundedness reasoning is priced as Azure OpenAI tokens.

**When in the lifecycle.** Runtime — every prompt and every completion. **Owner.** Application platform / AI platform team; co-owned with Security for policy thresholds.

---

## Pillar 2 — PII Detection and Protection

**Azure AI Language — PII detection.** The Language service exposes a PII entity recognizer covering tens of entity types (Person, PhoneNumber, Email, Address, ABA Routing, IBAN, IP, SSN, plus health and government IDs in the conversational PII model) ([PII detection overview](https://learn.microsoft.com/en-us/azure/ai-services/language-service/personally-identifiable-information/overview)). The current preview API (`api-version=2025-11-15-preview`) introduces explicit **`redactionPolicies`** with four modes — **`NoMask`**, **`CharacterMask`** (default \*), **`EntityMask`** (replace with entity-type token), and **`SyntheticReplacement`** (replace with realistic synthetic data so downstream parsers still work) ([PII redaction policies](https://learn.microsoft.com/en-us/azure/ai-services/language-service/personally-identifiable-information/how-to-call)).

**Microsoft Purview — Data Security Posture Management for AI.** Purview DSPM for AI provides discovery of AI usage across Microsoft 365 Copilot, Foundry, and third-party AI apps, and chains together **Sensitivity Labels**, **Data Loss Prevention (DLP) for Generative AI**, **Insider Risk Management**, and **Communication Compliance** to block sensitive prompts, audit prompt/response activity, and apply retention ([Purview DSPM for AI](https://learn.microsoft.com/en-us/purview/dspm-for-ai)). Foundry projects ingest Purview policies so that label-based DLP travels with the data into RAG indexes and agent tools.

**PII handling in Azure OpenAI / Foundry.** Customer prompts and completions are processed in the customer's Azure tenant and region; abuse-monitoring data retention can be opted out of via the Limited Access form. Data residency follows the Foundry/Azure OpenAI resource region selection.

**Microsoft Presidio (open source).** Presidio is Microsoft's MIT-licensed Python SDK for de-identification: an **Analyzer** (built-in + custom recognizers, NER + regex + checksum + context), **Anonymizer** (mask, replace, hash, redact, encrypt), and **Image Redactor** for screenshots ([Presidio](https://microsoft.github.io/presidio/)). Use it for on-prem / pre-cloud scrubbing pipelines and for building deterministic test corpora that don't carry real PII into Foundry.

**When in the lifecycle.** Data ingestion (Presidio, Purview classification), runtime (Language PII, Purview DLP), and post-hoc audit (Purview activity explorer). **Owner.** Data security / compliance, with platform team operating the connectors.

---

## Pillar 3 — AI Evaluations (Foundry)

**Built-in evaluators.** The `azure-ai-evaluation` SDK ships several evaluator families ([Foundry built-in evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/evaluators)):

- **General-purpose quality** — Coherence, Fluency, QA.
- **RAG quality** — Groundedness (and GroundednessPro, which calls the Content Safety service), Relevance, Retrieval, Response Completeness, Document Retrieval.
- **Textual similarity** — Similarity (LLM-judge) and classical metrics F1, BLEU, ROUGE, METEOR, GLEU.
- **Agent evaluators** — Intent Resolution, Tool Call Accuracy, Task Adherence.

**Risk and safety evaluators.** A separate evaluator family runs adversarial probes and AI-judge scoring for: **Violence, Sexual, Hate/Unfairness, Self-Harm, Protected Material, Indirect Attack (XPIA), Code Vulnerability, Ungrounded Attributes**, plus **Prohibited Actions** and **Sensitive Data Leakage** in preview ([Risk and safety evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/risk-safety-evaluators)). Each emits both a categorical label and a 0–7 severity, mirroring Content Safety.

**Custom evaluators.** Three authoring models are first-class: **callable Python evaluators** (any function that returns a score), **prompt-based evaluators** (Jinja templates with a model judge), and **composite evaluators** that combine the above. Evaluation results are logged to the Foundry project and can be visualized in the **Evaluations** tab.

**SDK and CI/CD.** The `azure-ai-evaluation` Python SDK (`pip install azure-ai-evaluation`) is the supported integration point ([Evaluate with the SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)). Microsoft publishes **GitHub Actions** and **Azure DevOps** templates that run evaluations on PRs and gate merges on score thresholds, treating eval suites like unit tests for LLM apps.

**Continuous (production) evaluation.** Foundry Observability supports **continuous evaluation** of sampled live traces, scoring them against the same evaluator catalog and publishing dashboards/alerts ([Continuous evaluation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/continuous-evaluation-agents)). This is the production analog of CI evals and is required for drift detection.

**When in the lifecycle.** Pre-deployment (CI gates), pre-release (offline benchmark suites), and continuously in production (sampling). **Owner.** AI engineering / quality, with thresholds co-owned by Responsible AI champions.

---

## Pillar 4 — AI Red Teaming

**PyRIT (Python Risk Identification Tool).** PyRIT is the open-source framework published by the **Microsoft AI Red Team** ([PyRIT on GitHub](https://github.com/Azure/PyRIT)). Its component model is the canonical mental model for automated AI red teaming:

- **Targets** — wrappers for the system under test (Azure OpenAI, Foundry deployments, OpenAI, Anthropic, HuggingFace, custom HTTP, even GUI/voice).
- **Datasets** — seed prompts (HarmBench, AdvBench, custom).
- **Converters** — transformations that obfuscate prompts (Base64, ROT13, ASCII art, translation, persuasion).
- **Scorers** — LLM-judge or Content-Safety-backed labels for the response.
- **Attack strategies** — multi-turn algorithms including **Crescendo**, **Tree of Attacks with Pruning (TAP)**, **Skeleton Key**, and PAIR.
- **Memory** — stores conversations for replay and analysis.

**Foundry AI Red Teaming Agent.** A managed, GA service that runs PyRIT-backed scans against a Foundry deployment or agent and produces an **Attack Success Rate (ASR)** report aligned to NIST AI RMF **Govern / Map / Measure / Manage** functions ([AI Red Teaming Agent](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/run-scans-ai-red-teaming-agent)). Scans can run **cloud-side** (large-scale) or **locally** via the SDK; risk categories follow the Content Safety taxonomy.

**Methodology — what to attack.** The Microsoft AI Red Team's published lessons from testing **100+ generative AI products** ([MS AI Red Team Lessons](https://airedteamwhitepapers.blob.core.windows.net/lessonswhitepaper/MS_AIRT_Lessons_eBook.pdf)) emphasizes attack categories beyond simple jailbreaks: **direct prompt injection**, **indirect (cross-prompt) prompt injection (XPIA)**, **data exfiltration via tool-use chains**, **model denial-of-service via amplification prompts**, **supply-chain risks** in fine-tunes and adapters, and **traditional security flaws** in the surrounding agent harness (SSRF, credential leakage). **Spotlighting** — explicitly tagging untrusted retrieved content — is a documented prompt-engineering defense for XPIA.

**Pre-deployment vs. continuous.** Pre-deployment uses exhaustive PyRIT scans, manual red-team exercises, and Responsible AI Impact Assessments. **Continuous** testing is enabled by scheduling Red Teaming Agent scans against staging endpoints on every model swap or system-prompt change, and feeding signals into Defender XDR.

**Owner.** Product security / AI Red Team, with platform team running the harness.

---

## Supplementary — Governance and Posture

**Microsoft Responsible AI Standard.** Microsoft's internal standard codifies **six principles** — Fairness, Reliability & Safety, Privacy & Security, Inclusiveness, Transparency, Accountability — and operationalizes them via Impact Assessments, sensitive-uses review, and Responsible AI by Design gates ([Microsoft RAI principles and approach](https://www.microsoft.com/en-us/ai/principles-and-approach)). It maps cleanly to the controls described in Pillars 1–4.

**NIST AI RMF + Generative AI Profile.** NIST AI 600-1 (the **Generative AI Profile**, July 2024) provides **200+ recommended actions** organized under the four AI RMF functions Govern / Map / Measure / Manage, with explicit attention to confabulation, dangerous content, data privacy, and information security ([NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.600-1.pdf)). The Foundry Red Teaming Agent and risk/safety evaluators emit reports that label findings by RMF function.

**ISO/IEC 42001:2023.** The first international **AI Management System (AIMS)** standard, certifiable, modeled on ISO 27001's plan-do-check-act ([ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html)). Microsoft has publicly committed to alignment; for regulated customers, mapping their AI program to 42001 controls is increasingly an audit expectation.

**Microsoft Defender for Cloud — AI-SPM.** Defender CSPM (paid plan) extends posture management to AI workloads. **AI Security Posture Management** discovers an **AI bill of materials** across Azure OpenAI, Azure AI Foundry, Azure Machine Learning, **Amazon Bedrock**, and **Google Vertex AI**, surfaces vulnerabilities, and computes attack paths from internet-exposed endpoints to grounding data ([Defender for Cloud AI-SPM](https://learn.microsoft.com/en-us/azure/defender-for-cloud/ai-security-posture)). Sister capability **Threat Protection for AI Services** is GA, integrates **Prompt Shields** signals into Defender XDR, raises alerts on jailbreak attempts and credential theft, and ships with a **30-day free trial up to 75 billion tokens** scanned ([Threat Protection for AI Services](https://learn.microsoft.com/en-us/azure/defender-for-cloud/ai-threat-protection)).

**When/Owner.** AI-SPM is steady-state SecOps + Cloud Security; governance is the Responsible AI council / Compliance.

---

## Conclusion

Microsoft's AI security portfolio is now coherent enough that a CSA can prescribe a default reference architecture without hedging: deploy models in **Foundry / Azure OpenAI**, wire **Content Safety + Prompt Shields + Groundedness Detection** as the runtime guardrail, layer **Azure AI Language PII** and **Purview DSPM for AI** as the data guardrail, gate releases with **`azure-ai-evaluation`** in CI plus **continuous evaluation** in production, run **PyRIT / Foundry AI Red Teaming Agent** scans against every promotion, and roll the entire posture into **Defender CSPM AI-SPM** with **Threat Protection for AI** alerts piping into Defender XDR. The reason this stack works is that each layer addresses a *different* failure mode — content policy violations, data leakage, quality drift, adversarial coercion, and posture drift — and the layers share a common taxonomy (the Content Safety harm categories) so signals cross-correlate. The principal residual risks: (a) several risk/safety evaluators are still in preview, so production reliance on Prohibited Actions or Sensitive Data Leakage scoring should be paired with manual review; (b) AI-SPM coverage of non-Azure model hosting (Bedrock, Vertex) is dependent on the customer connecting those clouds; and (c) governance frameworks (RAI Standard, NIST AI RMF, ISO 42001) remain organizational disciplines — tooling cannot substitute for the impact-assessment and accountability work they prescribe.

**Confidence: HIGH** for the tool capabilities and pricing (vendor-primary sources, cross-validated). **MODERATE** for preview-feature timelines, which Microsoft is iterating on quickly.

---

## Sources

1. [Azure AI Content Safety overview — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)
2. [Prompt Shields concepts — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)
3. [Groundedness Detection concepts — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness)
4. [Protected Material concepts — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/protected-material)
5. [Harm categories — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories)
6. [Azure OpenAI content filtering — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter)
7. [Azure AI Content Safety pricing — Azure](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/content-safety/)
8. [Azure AI Language PII detection overview — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/language-service/personally-identifiable-information/overview)
9. [Azure AI Language PII redaction policies — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/language-service/personally-identifiable-information/how-to-call)
10. [Microsoft Purview DSPM for AI — Microsoft Learn](https://learn.microsoft.com/en-us/purview/dspm-for-ai)
11. [Microsoft Presidio — Microsoft GitHub](https://microsoft.github.io/presidio/)
12. [Foundry built-in evaluators — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/evaluators)
13. [Risk and safety evaluators — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/risk-safety-evaluators)
14. [`azure-ai-evaluation` SDK — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
15. [Continuous evaluation in Foundry — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/continuous-evaluation-agents)
16. [PyRIT — Microsoft AI Red Team (GitHub)](https://github.com/Azure/PyRIT)
17. [AI Red Teaming Agent in Foundry — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/run-scans-ai-red-teaming-agent)
18. [Microsoft AI Red Team — Lessons from 100 GenAI Products](https://airedteamwhitepapers.blob.core.windows.net/lessonswhitepaper/MS_AIRT_Lessons_eBook.pdf)
19. [Microsoft Responsible AI principles and approach](https://www.microsoft.com/en-us/ai/principles-and-approach)
20. [Defender for Cloud AI Security Posture Management — Microsoft Learn](https://learn.microsoft.com/en-us/azure/defender-for-cloud/ai-security-posture)
21. [Threat Protection for AI Services — Microsoft Learn](https://learn.microsoft.com/en-us/azure/defender-for-cloud/ai-threat-protection)
22. [NIST AI 600-1: Generative AI Profile (July 2024)](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.600-1.pdf)
23. [ISO/IEC 42001:2023 — Information technology — AI management system](https://www.iso.org/standard/81230.html)
