# AI Security

## AI Security Lifecycle: Dev/Training vs. Inferencing Pipelines

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/aisecurity-1.jpg 'AI Security Lifecycle')

## AI Security: Dev/Training vs. Inferencing Pipelines

- Diagram provides an excellent high-level view of the AI security lifecycle, splitting into two interconnected phases: 
  - Dev/Training (building and hardening the system)
  - Inferencing (runtime execution)
- This aligns closely with industry frameworks like OWASP LLM Top 10 (2025), NIST AI RMF, and MITRE ATLAS, which emphasize "shift-left" security (catch issues early in dev) combined with runtime defenses.

- The Dev/Training flow is iterative and feedback-driven (e.g., Red Team → Guardrails loop), focusing on preventing vulnerabilities from being baked in. The Inferencing flow is linear and real-time, emphasizing detection and blocking at runtime. Below, I explain each block based on current best practices (sourced from OWASP, NIST, MITRE ATLAS, AWS/Azure guides, and red-teaming research as of 2026).
- For each, I cover:
  - What it is.
  - Why it matters for security.
  - Key risks if ignored.
  - Core mitigations.

## AI Security - Dev/Training Pipeline

This phase secures the foundation: data, models, and components before deployment.

### Data

Raw or curated datasets used for pre-training or fine-tuning.
- Why needed: Training data is the root of all model behavior. Poisoned or biased data leads to backdoors, hallucinations, or discriminatory outputs.
- Key risks: Training Data Poisoning (OWASP LLM03/LLM04), supply-chain tampering.
- Mitigations: Vet sources, use WORM storage, data versioning, anomaly detection, and synthetic/anonymized data for sensitive domains.

### Models (with Content Safety feedback)

Base or pre-trained foundation models (e.g., Llama, GPT variants). Content Safety here refers to safety-aligned models or filters applied during selection.
- Why needed: Models inherit supply-chain risks and must be chosen or hardened for safety.
- Key risks: Supply Chain Vulnerabilities (OWASP LLM03), model extraction/theft, unsafe base behavior.
- Mitigations: Use trusted repositories (e.g., Hugging Face with scanning), verify provenance, and apply initial content safety filters (e.g., Azure AI Content Safety or Llama Guard).

### Fine tuning

Adapting a base model on domain-specific data.
- Why needed: Customizes performance but can degrade built-in safety alignments if not done securely.
- Key risks: Fine-tuning can break refusal mechanisms, introduce backdoors, or leak PII from training data.
- Mitigations: Use LoRA/DoRA for efficiency, anonymize/redact data, post-tune safety testing, and differential privacy techniques.

### Evaluations

Systematic testing of model outputs against safety, fairness, and robustness benchmarks.
- Why needed: Quantifies risks before agents/tools are built on top.
- Key risks: Undetected jailbreaks, bias, or hallucinations propagating downstream.
- Mitigations: Use frameworks like Inspect (UK AISI), DeepEval, or RAGAS; test for OWASP risks.

### Red Team (feedback to Guardrails)

Adversarial simulation of attacks (jailbreaks, prompt injection, poisoning).
- Why needed: Finds edge cases that automated evals miss; it's the "human hacker" layer.
- Key risks: Real-world exploits (e.g., indirect prompt injection via tools).
- Mitigations: Tools like PyRIT, Garak; automated + human red teaming; feed results back to refine Guardrails.

### Guardrails (built from Evaluations + Red Team)

Policy-enforcement layers (filters, rules, classifiers) that wrap models/agents.
- Why needed: Acts as the "safety net" derived from testing; prevents unsafe behavior at runtime.
- Key risks: Without them, even tested models fail in production.
- Mitigations: NeMo Guardrails, AWS Bedrock Guardrails, or custom (input/output validation, topic blocking).

### Agents

Autonomous systems that reason, plan, and act using tools.
- Why needed: Agents amplify capabilities (and risks) like excessive agency.
- Key risks: Tool misuse, memory poisoning, autonomous harmful actions (OWASP Excessive Agency).
- Mitigations: Least-privilege design, sandboxing.

### Tools

External functions/APIs agents can call (e.g., databases, search).
- Why needed: Tools give agents real-world impact but expand attack surface.
- Key risks: Tool injection, privilege escalation.
- Mitigations: Schema validation, allowlists, execution isolation.

### Knowledge Base

Vector/RAG store for retrieval-augmented generation.
- Why needed: Provides context but is vulnerable to poisoning.
- Key risks: Retrieval poisoning, sensitive data leakage.
- Mitigations: RBAC, chunk-level encryption/redaction, source verification, WORM storage.

### Infrastructure

Compute, storage, pipelines, and deployment environment.
- Why needed: Underpins everything; insecure infra compromises the entire chain.
- Key risks: Model theft, DoS, supply-chain attacks.
- Mitigations: Zero-trust access, container isolation, AI gateways, continuous monitoring, encrypted channels.

## AI Security - Inferencing Pipeline

This is runtime protection for live queries and agent actions.

### Data → Content Safety

User inputs/queries filtered before reaching the model.
- Why needed: Blocks malicious prompts in real time.
- Key risks: Prompt Injection (OWASP LLM01).
- Mitigations: Prompt Shields, input sanitization, Azure/Google content filters.

### Models

Inference calls on deployed models.
- Why needed: Even safe models can be manipulated at runtime.
- Key risks: Model exploitation, prompt injection.
- Mitigations: Rate limiting, output validation.

### Agents → Tools

Runtime agent reasoning and tool execution.
- Why needed: Controls autonomous behavior
- Key risks: Memory/context poisoning, excessive agency.
- Mitigations: Step-by-step validation, human-in-the-loop for high-risk actions.

### Memory

Short/long-term agent state or conversation history.
- Why needed: Persistent memory enables useful agents but creates stealth attack vectors
- Key risks: Memory poisoning, cross-session leakage.
- Mitigations: Isolation per user/session, expiration, sanitization before storage, cryptographic checks.

### Guardrails

Runtime enforcement (input/output filters, policy checks).
- Why needed: Last line of defense; catches what dev testing missed
- Mitigations: Multi-layer (pre/post-model), groundedness checks, sensitive data redaction.

### Knowledge Base → Infrastructure

Retrieval + hosting layer.
- Why needed: Ensures secure, private context at scale
- Mitigations: As in training (RBAC, auditing) plus real-time retrieval guardrails.

## Security Choke Points & Where to Put Mitigation Steps

diagram already highlights natural choke points (Content Safety, Guardrails, Red Team/Evaluations loops). These are the highest-ROI places for controls—insert mitigations here to create defense-in-depth.

- Data (both pipelines) – Entry point for poisoning.
 - Mitigation: Automated validation + provenance tracking + redaction before ingestion/storage.
- Content Safety (pre-Models in both flows) – Primary input filter.
 - Mitigation: Prompt Shields + harm-category classifiers (hate, violence, PII); block or rewrite before model.
- Evaluations + Red Team (Dev/Training) – Pre-deployment gate.
 - Mitigation: Automated adversarial testing + human red teaming; fail builds if thresholds breached; feed directly into Guardrails.
- Guardrails (both pipelines, especially post-Red Team and in Inferencing) – Runtime + feedback choke point.
 - Mitigation: Layered (NeMo/AWS Bedrock style): input validation → model call → output filtering → memory sanitization. Use as a feedback loop for continuous improvement.
- Agents/Tools/Memory (Inferencing) – Behavioral choke points.
 - Mitigation: Least-privilege tool calls, schema validation, memory isolation/expiration, step-wise approval for high-risk actions.
- Knowledge Base (both) – Retrieval choke point.
 - Mitigation: Chunk attribution, trust scoring, RBAC per query, redaction of sensitive chunks.
- Infrastructure (final layer) – Deployment choke point.
 - Mitigation: AI Gateway (rate limits, auth, logging), zero-trust networking, anomaly detection, immutable infra-as-code.

## Recommended Implementation Order (Practical Roadmap)

- Start with Content Safety + Guardrails (quick wins).
- Add Red Team/Evaluations loop in CI/CD.
- Layer memory/tool isolation for agents.
- Enforce Infrastructure zero-trust last (foundational).

This structure turns your diagram into a resilient, auditable security program. It directly maps to OWASP risks (e.g., Prompt Injection blocked at Content Safety/Guardrails; Data Poisoning prevented upstream). By following this roadmap, you can build a secure AI system that is robust against both known and emerging threats.

## Conclusion

- The above security model is evolving rapidly, but this framework provides a comprehensive starting point. It emphasizes the importance of both proactive (Dev/Training) and reactive (Inferencing) controls, with clear choke points for mitigation. By implementing these layers thoughtfully, organizations can significantly reduce the risk of AI-related security incidents while still enabling powerful capabilities.