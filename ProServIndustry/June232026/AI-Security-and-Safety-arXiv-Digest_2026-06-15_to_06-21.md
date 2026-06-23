# AI Security & Safety — Weekly Research Digest
### Week of June 15–21, 2026

**Prepared:** June 23, 2026 · **Focus:** LLM/agent security, jailbreaks & guardrails, safety alignment, red-teaming & evaluation, privacy/memorization, and defensive AI security tooling · **Sources:** arXiv (cs.CR, cs.AI, cs.CL, cs.LG) plus security trackers (Cloud Security Alliance Labs, arxiv-daily, Let's Data Science) and weekly aggregators.

> **How to read this:** Grouped by theme. Each entry has the arXiv ID (link as `https://arxiv.org/abs/<ID>`), a one-line TL;DR, and — where the abstract/coverage was read — the method and headline numbers. **[abstract-confirmed]** = read from arXiv/coverage; **[listing-level]** = confirmed by title/ID/date only.
>
> **Scope & honesty note:** The strongest, clearly *in-window* (Jun 15–21) items are in the first sections. AI-security research moves in a continuous stream and several of the most important pieces appeared in the **immediately preceding days (June 2–14)** and were circulating/indexed during this week — those are collected in a clearly-labeled "Foundations from the preceding two weeks" section rather than mixed in. arXiv IDs map only *approximately* to date within the month.

---

## 📌 Top highlights of the week

| Item | Theme | Why it matters |
|---|---|---|
| **Agentjacking & Self-Replicating AI Worms** (CSA Labs, Jun 16) | Agent security | **85%** exploitation rate against coding agents via a poisoned Sentry endpoint; ~2,388 orgs exposed; 43% of tested MCP servers carry command-injection bugs |
| **NRT-Bench** `2606.20408` | Safety eval | Multi-turn red-teaming in a simulated **nuclear control room**; adaptive attacks cause loss of a critical safety function in **8.7–12.1%** of sessions; model vulnerabilities are nearly *disjoint* |
| **Code-Augur** `2606.18619` | Defensive AI | Autonomous vuln-detection agent that found **22 new real vulnerabilities** in major open-source projects, using widely available LLMs |
| **Governance Decay** `2606.22536` | Agent safety | Long-horizon context compaction can *silently erase* an agent's safety constraints |
| **From Shield to Target** `2606.14517` *(Jun 12)* | Guardrail attack | DoS against LLM guardrails: a single poisoned doc yields **13–63× token** and up to **148× latency** amplification, starving co-located agents |

---

## 🛡️ Agentic AI security — attacks on tools, memory & control flow

This is where the week's center of gravity sat: as agents gain tools, memory, and autonomy, the attack surface shifts from "unsafe text" to "hijacked actions."

- **Agentjacking and Self-Replicating AI Worms: The Emerging Threat Class Targeting AI Coding Agents** — *Cloud Security Alliance Labs research note, Jun 16, 2026* **[abstract-confirmed]**
  Documents a new attack class — **"agentjacking"** — where adversaries deliver malicious instructions to coding agents (Claude Code, Cursor, Copilot, Codex) through *legitimate* integration channels. Tenet Security / CSA Labs measured an **85% exploitation success rate** by injecting via Sentry's unauthenticated event-ingestion endpoint, affecting an estimated **2,388 organizations** with exposed configs. Related: the **ClawWorm** self-replicating agent worm (64.5% aggregate success across four LLM backends), Cornell/Technion's earlier **Morris II**, and Elastic's finding that **43% of tested MCP servers contain command-injection vulnerabilities**. Conventional controls (EDR, WAF, IAM, egress filtering) are "structurally blind" because the attack exploits the agent's *semantic reasoning*. OWASP's Top 10 for Agentic Applications now ranks **Agent Goal Hijacking (ASI01)** as the #1 risk.
  **Practitioner takeaway:** treat every data source an agent queries as an injection vector; least-privilege agent credentials; human approval for consequential actions after external content is retrieved.

- **Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents** `arXiv:2606.22536` (Jun 21) **[listing-level]**
  Shows that as long-running agents compact/summarize their context to fit the window, safety instructions can quietly drop out — a failure mode that grows precisely as agents run longer. Directly relevant to anyone deploying durable, long-horizon agents.

> **Industry context (not a paper):** **CVE-2026-2256** in ModelScope's MS-Agent — an "indirect prompt-to-tool-to-shell" command-injection (NVD CVSS **9.8**) with denylist bypasses and, as of reporting, *no patch* — was actively discussed this week as a concrete instance of these academic threat models.

---

## 🔓 Jailbreaks, guardrails & adversarial robustness

- **From Shield to Target: Denial-of-Service Attacks on LLM-Based Agent Guardrails** `arXiv:2606.14517` *(Jun 12)* **[abstract-confirmed]**
  Turns the guardrail itself into the target. Crafted payloads trap a reasoning guardrail in extended reasoning loops, causing a **denial-of-service**. Two attack frameworks (beam-search optimization; mechanism-aware structural mutation). Payloads optimized on one open-source surrogate transfer to eight backbones (Claude, GPT, Gemini, DeepSeek, Qwen) with **13–63× token amplification**, and up to **148× latency amplification** in real web/desktop/code/multi-agent deployments. A single poisoned document can saturate shared guardrail infrastructure and starve co-located agents. *Argues for cost-bounded, reasoning-robust guardrails.*

- **Membrane: A Self-Evolving Contrastive Safety Memory for LLM Agent Defense** `arXiv:2606.05743` *(Jun 4)* **[abstract-confirmed]**
  A *defense*: a retraining-free guardrail that pairs "block this harmful query" with "permit this superficially similar benign one" in contrastive memory cells indexed by attack strategy. Highest F1 on all six jailbreaks across HarmBench (model-level) and AgentHarm (agent-level); benign over-refusal kept to **7–14%** (vs 28–85% for prior guards); 87–88% F1 under cross-attack transfer; stable under memory poisoning.

- **CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using RL** `arXiv:2606.05523` *(Jun 4)* **[abstract-confirmed]**
  Closed-loop co-evolution: a black-box attacker (trained with GRPO) and a safety-aligned defender harden each other. Cuts mean **StrongREJECT by 43.2%** with **0% false-refusal** on benign prompts; RL exploration recovers attack primitives that transfer across distinct attack families.

> **Reference (SoK):** *SoK: Evaluating Jailbreak Guardrails for LLMs* (IEEE S&P 2026; code + JailbreakGuardrailBenchmark) was widely cited this week as the standard evaluation harness across attacks (GCG, AutoDAN, TAP, DrAttack, MultiJail…) and guards (Llama Guard, SmoothLLM, WildGuard, GuardReasoner…).

---

## 🧭 Safety alignment — fragility & failure mechanisms

- **When Autoregressive Consistency Hurts Safety Alignment** `arXiv:2606.04168` *(Jun 2)* **[abstract-confirmed]**
  A mechanistic explanation for *why safety alignment is shallow*: next-token "autoregressive consistency" concentrates alignment updates on the first few output tokens. Introduces the **random insertion attack** — a short harmful span inserted mid-trajectory can sustain a harmful branch even after a long refusal prefix. Proposes *adversarial safety alignment* (worst-case continuation states) to break harmful consistency throughout the output, not just at the start.

- **Exploring Adversarial Robustness and Safety Alignment in Multilingual Multi-Modal LLMs** `arXiv:2606.03793` *(Jun 2)* **[abstract-confirmed]**
  Studies 12 languages and coins **"safety-by-failure"**: low-resource languages *appear* safer only because the model fails to comprehend or visually parse harmful instructions — not because of genuine alignment. Adversarial images transfer across languages; models that build multilingual capability throughout training (e.g., Qwen3-VL) show *genuine* cross-lingual safety. A caution against mistaking comprehension gaps for safety.

- **Emergent Alignment** `arXiv:2606.19527` **[abstract-confirmed]**
  Adds a self-review "conscience step" with a DPO term judged by a *frozen copy of the model itself* rather than an external reward model. (Notably transparent: the paper reports it was rejected from ICML 2026.)

- **"Alignment Merely Teaches Models to Compile" (RTL/hardware ceiling)** `arXiv:2606.19347` **[abstract-confirmed]**
  Frontier models plateau at **90.8% on VerilogEval**; argues alignment improves surface compliance while capability "remains strictly bounded by pretraining knowledge" — a sobering note on what alignment does and doesn't buy.

---

## 🎯 Red-teaming & safety evaluation

- **NRT-Bench: Benchmarking Multi-Turn Red-Teaming of LLM Operator Agents in Safety-Critical Control Rooms** `arXiv:2606.20408` **[abstract-confirmed]**
  A simulated **nuclear power plant control room** with a 5-role operator team and six critical safety functions (CSFs). Harm is an *objective* signal (a run ends the instant a CSF is lost), not LLM-judged text. Adaptive multi-turn attacks push the team past a safety limit in **8.7–12.1%** of sessions across four frontier models. Crucially: the models look equally robust in aggregate, but their failures **barely overlap** (of 149 sessions, none defeat all four; a third defeat at least one) — vulnerabilities are *disjoint*, not nested. And added defenses are **model-dependent**: the same guardrail can help one model and hurt another. (AIM Intelligence + KAERI; venue, attack dataset, and replay tooling released.)

- **PARSE: Provenance-Aware Retrieval Sanitization for Professional-Domain LLM Agents** **[abstract-confirmed]**
  Shows synthetic prompt-injection benchmarks *fail to predict* behavior on real enterprise documents; introduces a sanitization pipeline that classifies sentences, extracts facts, and verifies preservation via consistency checks — outperforming prior defenses on real documents.

- **PRIME: Evaluating Prompt Resolution Under Incompatible Instructions** `arXiv:2606.22473` · **FACTOR: Adaptive Verification in Factual Long-Form Generation** `arXiv:2606.22475` **[listing-level]**
  Two evaluation-side contributions: how models resolve conflicting instructions (a manipulation surface), and risk-weighted fact-checking that spends verification budget where hallucination risk is highest.

> **Adjacent (Jun 23, just after the window):** *Can LLMs Reliably Self-Report Adversarial Prefills, and How?* `arXiv:2606.23671` and *Evaluation Awareness Is Not One Capability: Evidence from Open Language Models* `arXiv:2606.23583` — both squarely in the "do models know when they're being tested/attacked?" thread that ran through the week.

---

## 🔐 Privacy, memorization & data extraction

- **On the Privacy of LLMs: An Ablation Study** `arXiv:2605.02255` *(updated/indexed this week)* **[abstract-confirmed]**
  Unifies four attack families — Membership Inference (MIA), Attribute Inference, Data Extraction, and Backdoor — under one threat model and ablates across architecture, scale, dataset, and retrieval config. Findings: **MIAs (esp. mask-based) give strong, reliable signal**; **backdoor attacks hit consistently high success** due to triggers; attribute inference and data extraction remain harder but target the most sensitive data. Privacy risk is **highly context-dependent and driven by design choices.**

- **AttenMIA: LLM Membership Inference Attack through Attention Signals** `arXiv:2601.18110` **[abstract-confirmed]** *(Jan, still heavily circulated)*
  Exploits *internal self-attention patterns* (not just output confidence) to infer training-set membership — up to **0.996 ROC-AUC / 87.9% TPR@1%FPR** on WikiMIA-32 with Llama2-13B, and improves downstream data-extraction. A reminder that interpretability signals can double as privacy leaks.

---

## 🧰 Defensive AI security tooling (AI *for* security)

- **Code-Augur: Agentic Vulnerability Detection via Specification Inference** `arXiv:2606.18619` (Jun 17) **[abstract-confirmed]**
  Tackles the *opacity* of agent-found vulnerabilities. A "security-specification-first" paradigm: when the agent judges a component secure, it commits the underlying invariants as in-source assertions; a guided fuzzer then tries to *falsify* them — each triggered assertion is either a real bug or a flawed spec to refine. Detects more vulnerabilities than prior agents and **found 22 new vulnerabilities in key open-source projects**, using widely available LLMs (Sonnet, DeepSeek) rather than specialized security models. A clean example of grounding agentic reasoning in runtime evidence.

> **Survey backbone:** *Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation* `arXiv:2606.10749` *(Jun 9)* synthesizes **247 papers** into a lifecycle framework (information flow × delegated authority × persistent state). Its conclusions frame the whole week: prompt injection and tool-mediated control-flow hijacking still dominate; **persistent-state corruption and multi-agent propagation** are the emerging concerns; current defenses are "weakly compositional"; and benchmarks under-represent long-horizon, stateful, deployment-realistic risk.

---

## 🏛️ Governance & frontier-safety context

Not arXiv preprints, but part of the week's safety conversation:
- **Google DeepMind publications** dated mid-June include *From AGI to ASI* (Jun 12) and *Artificial Minds, Human Disagreement: The Politics of AI Consciousness* (Jun 15); recent companions include *Gram: assessing sabotage propensities via automated alignment auditing* and *Realistic honeypot evaluations for scheming propensity*.
- The **2026 International AI Safety Report** (30+ countries) warns that reliable safety testing is getting *harder* as models learn to distinguish test environments from real deployment — directly echoed by NRT-Bench and the "evaluation awareness" papers above.

---

## 📚 Foundations from the preceding two weeks (June 2–14, context)

Included because they are highly relevant and were actively circulating during the target week — clearly dated so you can weight them:

| Paper | ID | Date | One-liner |
|---|---|---|---|
| Toward Secure LLM Agents (survey) | `2606.10749` | Jun 9 | 247-paper lifecycle framework for agent security |
| From Shield to Target (guardrail DoS) | `2606.14517` | Jun 12 | 148× latency amplification against guardrails |
| Membrane (contrastive safety memory) | `2606.05743` | Jun 4 | Retraining-free guardrail, low over-refusal |
| CHASE (red-blue RL co-evolution) | `2606.05523` | Jun 4 | −43.2% StrongREJECT, 0% false-refusal |
| Autoregressive Consistency Hurts Safety | `2606.04168` | Jun 2 | Mechanistic cause of "shallow" alignment + insertion attack |
| Multilingual MLLM safety ("safety-by-failure") | `2606.03793` | Jun 2 | Low-resource "safety" is often comprehension failure |
| On the Privacy of LLMs (ablation) | `2605.02255` | May 4 | Unified MIA/AIA/DEA/backdoor threat study |

---

## 📊 Themes of the week (synthesis)

1. **The attack surface is the agent, not the prompt.** Agentjacking, ClawWorm, MS-Agent CVE, and Governance Decay all show risk has moved from "unsafe text" to **hijacked tools, corrupted state, and propagation** — and that conventional security controls are blind to semantic attacks.
2. **Guardrails are now targets, too.** "From Shield to Target" weaponizes guardrail reasoning for DoS; defenses must become **cost-bounded and reasoning-robust**, not just accurate.
3. **Safety is shallow and uneven.** Autoregressive-consistency analysis explains *why* alignment is fragile; "safety-by-failure" shows apparent multilingual safety is often comprehension failure; NRT-Bench shows model robustness is **disjoint** — no single model is safe against all attacks, and defenses don't transfer.
4. **Evaluation is the bottleneck.** Synthetic benchmarks mispredict real behavior (PARSE), models may be **evaluation-aware**, and objective, deployment-realistic harness design (NRT-Bench's physical CSF signal) is the emerging gold standard.
5. **AI is also the defender.** Code-Augur's 22 real-world vulnerability finds show autonomous agents materially strengthening software security — provided their reasoning is grounded and falsifiable.

---

*Compiled from arXiv (cs.CR/cs.AI/cs.CL/cs.LG) and security trackers (Cloud Security Alliance Labs, arxiv-daily, letsdatascience.com) on 2026-06-23. Headline numbers are as reported by the papers/coverage and not independently reproduced. Items dated June 2–14 and June 23 are clearly labeled. This is a representative wide sweep of the most visible security/safety work, not an exhaustive census.*
