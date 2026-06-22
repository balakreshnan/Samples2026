# Software Engineering After Claude: The Agentic Era

**A research briefing on how software gets built in the post–AI-coding-agent world**

*Prepared for: Balamurugan Balakreshnan*
*Date: June 22, 2026*
*Method: Multi-source deep research — 25 sources fetched, 125 claims extracted, adversarially verified (3-vote) to 17 confirmed findings. Claims that failed verification are explicitly excluded.*

---

## How to read this report

Findings fall into three tiers, flagged inline:

- **[Verified]** — confirmed by adversarial multi-source verification with citations.
- **[Industry view]** — well-established practitioner consensus or framework reasoning, not a single hard statistic. Use as direction, not as a measured fact.
- **[Caveat]** — a limitation, bias, or time-sensitivity you should weigh before acting.

> **Headline:** The job is shifting from *writing* code to *specifying, reviewing, orchestrating, and verifying* code that agents write. The capability is real and accelerating, but it is a **supervised collaboration, not autonomy** — and the two biggest costs are **security** and **skill atrophy**, not productivity.

---

## Table of Contents

1. [The shift: what software engineering becomes](#1-the-shift-what-software-engineering-becomes)
2. [End-to-end process for production applications with agents](#2-end-to-end-process-for-production-applications-with-agents)
3. [Vendor comparison: who is best at what](#3-vendor-comparison-who-is-best-at-what)
4. [Career reinvention: where the workforce goes](#4-career-reinvention-where-the-workforce-goes)
5. [Pros and cons of the new approach](#5-pros-and-cons-of-the-new-approach)
6. [Where the room to innovate and grow is](#6-where-the-room-to-innovate-and-grow-is)
7. [Bottom line and a personal action plan](#7-bottom-line-and-a-personal-action-plan)
8. [Sources](#8-sources)

---

## 1. The shift: what software engineering becomes

### The core transition

**[Verified]** The software engineering role is moving *from writing code to specifying, reviewing, testing, and orchestrating AI agents that write code.* Human expertise re-concentrates on **architecture, system design, strategic decisions, and accountability** for the software's reliability, performance, and value. Thoughtworks frames the new identity as **"stewards of systems."** For greenfield work where code is generated directly from specifications, the developer's role moves *closer to product owner / business analyst.* (Sources: Anthropic "Eight trends defining how software gets built in 2026"; Thoughtworks, Dec 2025.)

**[Verified]** This is **supervised collaboration, not full automation.** At Anthropic — an early-adopter, AI-native shop — engineers use AI in roughly **60% of their work but can "fully delegate" only 0–20% of tasks.** Effective use *requires supervision, validation, and human judgment.* (Source: Anthropic Societal Impacts study, 132 engineers + 53 interviews + Claude Code usage data, Dec 2025.)

**[Verified]** Agents are gaining built-in testing and validation loops, but **still struggle with large context in complex projects** — they lose track of details, misinterpret project data, or make planning mistakes "an experienced software engineer would catch." This is precisely why **human verification remains a required checkpoint.** (Source: Fast Company, Feb 2026; corroborated by arXiv "Codified Context" and Anthropic's own "context rot" research.)

### What the role looks like in 1–5 years

**[Industry view]** A realistic trajectory:

| Horizon | What changes | What stays human |
|---|---|---|
| **Now → 12 months** | Agents handle the "first draft" of most features, tests, refactors, and glue code. IDEs and CLI agents (Claude Code, Codex) become the default surface. | Spec writing, architecture, review, security sign-off, production accountability. |
| **1–3 years** | Multi-agent workflows tackle multi-file, multi-hour tasks. "Spec-as-source-of-truth" workflows mature. Review becomes the bottleneck, so AI-automated review scales up. | Judgment on trade-offs, domain modeling, ambiguous-requirement resolution, incident command. |
| **3–5 years** | Larger share of routine CRUD/integration work fully delegated. Engineering org charts flatten; "one senior + a fleet of agents" patterns spread. Non-engineers ship software via agentic tools. | System-level correctness, novel design, cross-system integration, safety/compliance, and *deciding what to build.* |

**[Caveat]** The role-shift narrative leans heavily on Anthropic and tool vendors, who have an interest in the "human-in-the-loop, buy-our-tools" framing. Treat the *direction* as well-supported and the *pace* as uncertain.

### The skills that rise in value

- **Specification & decomposition** — turning fuzzy intent into precise, testable requirements an agent can execute.
- **Code review at scale** — reading and judging code you didn't write, fast, including security and architectural review.
- **System design & integration** — the parts agents are weakest at (cross-cutting concerns, data modeling, distributed-systems trade-offs).
- **Verification & evals** — designing the tests, harnesses, and acceptance criteria that *prove* agent output is correct.
- **Agent orchestration** — context engineering, tool/MCP wiring, multi-agent coordination, and knowing when *not* to delegate.

---

## 2. End-to-end process for production applications with agents

This is the **agentic SDLC**. At each stage, note what the agent now does and where the **human checkpoint (🔒)** stays.

### Stage-by-stage

| Stage | What the agent does | Human checkpoint 🔒 |
|---|---|---|
| **1. Ideation / Spec** | Drafts user stories, acceptance criteria, edge cases from a prompt; proposes a `spec.md` / `CLAUDE.md`. | 🔒 **Define the problem and the "definition of done."** This is where humans now spend *more* time. Ambiguity here multiplies downstream. |
| **2. Planning / Architecture** | Proposes architecture, data models, tech stack, task breakdown, file structure. | 🔒 **Approve the architecture.** Agents make planning mistakes a senior would catch; architectural flaws are the costliest to fix later (see §5). |
| **3. Scaffolding** | Generates project skeleton, configs, CI files, boilerplate, interfaces. | Light review — low risk, high time-savings. Good first thing to delegate. |
| **4. Implementation** | Writes features across multiple files, iterating against tests. | 🔒 **Review diffs in meaningful chunks.** Don't accept large opaque changes. Keep PRs small enough to actually read. |
| **5. Testing** | Generates unit/integration/e2e tests; runs suites in a sandbox; iterates until green. | 🔒 **Verify tests are *meaningful*, not just green.** Agents can write tests that pass but assert nothing real, or test the bug. |
| **6. Code review** | AI reviewer flags bugs, style, security; can self-review before human sees it. | 🔒 **Final human review on correctness, intent, and trade-offs.** AI review *augments*, doesn't replace, the accountable reviewer. |
| **7. Security** | Runs SAST/dependency scans; suggests fixes; can generate threat models. | 🔒 **Security sign-off is mandatory and non-delegable** (this is the #1 risk surface — see §5). Embed security *from the earliest stages*, not as a final gate. |
| **8. CI/CD** | Writes/updates pipelines, IaC, deployment configs; can open and shepherd PRs. | 🔒 **Approve production deploys and infra changes.** Gate merges to main behind human + automated checks. |
| **9. Deployment** | Executes deploys via pipeline, runs smoke tests, can roll back on signal. | 🔒 **Own the production change** — blast radius, rollout strategy, feature flags. |
| **10. Observability / Maintenance** | Watches logs/metrics, triages alerts, proposes fixes/PRs for incidents, keeps deps current. | 🔒 **Incident command and root-cause accountability stay human.** Agents assist triage; humans own the call. |

### Principles that make the agentic SDLC work

**[Industry view]**

1. **Spec is the new source code.** The clearer and more testable the spec, the better the output. "Spec-driven development" is the dominant emerging discipline.
2. **Tests and evals are the guardrail.** Agents should iterate against a strong test/acceptance harness. Without it, you're trusting vibes.
3. **Keep changes reviewable.** Small, scoped, frequently-merged diffs beat giant agent-generated dumps no human can audit.
4. **Context engineering matters more than prompt cleverness.** Curate what the agent sees (repo conventions, architecture docs, constraints). Context rot is a real failure mode in long sessions.
5. **Defense in depth on security.** Assume ~half of generated code may carry a vulnerability (§5) and design scanning + review accordingly.
6. **Human accountability never delegates.** Someone with a name owns every production change.

**[Caveat]** No single, independently-verified, end-to-end stage benchmark survived verification in this research — the stage table is synthesized from practitioner playbooks (Provectus agentic SDLC, GitHub agentic-sdlc-playbook, Anthropic guidance) plus the verified findings on context limits, testing loops, and security. Treat it as a well-grounded *operating model*, not a measured one.

---

## 3. Vendor comparison: who is best at what

### Benchmark context first

**[Verified]** **SWE-bench Verified** — the headline coding benchmark — is **500 human-validated real GitHub issues** requiring multi-function/multi-file patches in Python repos, scored 0–1. As of June 2026, llm-stats tracks **102 models** on it. (Source: llm-stats; OpenAI; swebench.com.)

**[Verified]** On **independent SWE-bench Verified leaderboards (mid-June 2026)**, Anthropic's Claude models lead:

| Model | SWE-bench Verified | Notes |
|---|---|---|
| Claude Fable 5 | **95.0%** | Top entry; non-standard reasoning loop; self-reported |
| Claude (Mythos-class) | 93.9% | Non-standard reasoning configuration |
| **Claude Opus 4.8** | **88.6%** | **Top *standard* production model** (with thinking) |
| OpenAI GPT 5.5 | 82.6% | |
| Google Gemini 3.5 Flash | 78.8% | |

(Sources: Vals AI, updated 6/17/2026; cross-confirmed by llm-stats and steel.dev.)

**[Caveat — read before quoting these numbers]**
- Frontier Claude scores (Fable 5, Opus 4.8) are **self-reported to the leaderboards.**
- By *raw* score, GPT 5.5 sits closer to ~#5 once second Opus runs and other models (e.g., GLM-class) are counted — Claude still leads overall, but the gap is narrower than the headline implies.
- Benchmarks carry **contamination caveats**; **OpenAI deprecated SWE-bench for frontier evaluation in Feb 2026.**
- These model names and scores **will date within weeks.** Treat the *ordering and shape*, not the decimals, as the takeaway.

### Qualitative differentiation

**[Verified]** Developers report **Claude is stronger at higher-level planning and understanding intent**, while **OpenAI's Codex is better at following specific instructions and matching an existing codebase.** (Source: Fast Company, Feb 2026; corroborated by MindStudio's 100-hour comparison, May 2026.) **[Caveat]** This is qualitative sentiment covering only two vendors, not a benchmark.

### "Best at which feature" — practitioner map

> **[Industry view]** The table below blends the *verified* leaderboard/qualitative data above with **established industry positioning** for vendors the verification pass did not cover head-to-head (Gemini, Copilot, Cursor, Amazon Q, Devin, Windsurf, DeepSeek, Mistral, Grok, Llama). For those, treat entries as *general positioning, not measured benchmarks.*

| Vendor / Tool | Strongest at | Typical use | Notes |
|---|---|---|---|
| **Anthropic Claude (Opus/Sonnet) + Claude Code** | **Top coding benchmarks; high-level planning, intent understanding, agentic long-horizon work; large context** | Complex feature work, refactors, autonomous multi-step tasks, terminal-native agentic coding | Leads SWE-bench [Verified]; "stronger at planning" [Verified] |
| **OpenAI GPT-5.x / Codex** | **Instruction-following, matching an existing codebase, sandboxed test-run-iterate loops** | Precise edits to established repos; test-driven agent loops | "Better at following specific instructions" [Verified] |
| **Google Gemini** | **Very large context windows; multimodal; tight Google Cloud/Workspace integration** | Huge-codebase reasoning, doc+code+image tasks, GCP shops | [Industry view] |
| **GitHub Copilot** | **Deepest IDE/GitHub-native integration; ubiquity; enterprise governance** | In-editor completion + chat + PR workflows for teams already on GitHub | [Industry view] |
| **Cursor** | **Best-in-class agentic *IDE* experience; multi-file edits; model-agnostic** | Developers who want an AI-first editor wrapping multiple frontier models | [Industry view] |
| **Amazon Q Developer** | **AWS ecosystem depth; enterprise security/compliance; code transformation/upgrades** | AWS-heavy enterprises, large-scale Java/.NET migrations | [Industry view] |
| **Cognition Devin / Windsurf** | **Autonomous "agent as teammate" / async task completion** | Delegating well-scoped tickets to run unattended | [Industry view] |
| **DeepSeek** | **Strong open(-weight) performance at very low cost** | Cost-sensitive teams, self-hosting, on-prem | [Industry view] |
| **Meta Llama / Code Llama** | **Open weights; self-hosting; customization/fine-tuning; data sovereignty** | Regulated orgs needing on-prem control | [Industry view] |
| **Mistral** | **Open/efficient European models; strong price-performance** | EU data-residency, lean deployments | [Industry view] |
| **xAI Grok** | **Fast iteration, real-time/X integration, large context** | Rapid prototyping, X-platform-adjacent work | [Industry view] |

### Choosing — a simple rubric

- **Pure coding capability / hard agentic tasks:** Claude (Opus/Sonnet) + Claude Code is the current benchmark leader. **[Verified]**
- **Precise edits inside a big existing codebase:** Codex. **[Verified]**
- **You live in GitHub:** Copilot for friction-free integration. **[Industry view]**
- **You want an AI-native editor:** Cursor. **[Industry view]**
- **AWS / Azure / GCP lock-in:** Amazon Q / (cloud-native) / Gemini respectively. **[Industry view]**
- **Cost / data sovereignty / on-prem:** DeepSeek, Llama, Mistral (open weights). **[Industry view]**
- **Don't single-source:** the leaderboard reshuffles every few weeks — build your workflow to **swap models**, and **run your own eval** on *your* codebase rather than trusting public benchmarks.

---

## 4. Career reinvention: where the workforce goes

**[Caveat]** No specific career-pivot statistics survived verification — this section is **[Industry view]**: a strategic framework grounded in the verified role-shift (§1) and skill-demand signals, not measured outcomes. The good news embedded in the data: agents *can't fully delegate* most work, so demand is **transforming, not vanishing** — but the *shape* of demand changes.

### The reframe

Don't think "leave software engineering." Think **"move up the value chain that agents can't reach yet"**: judgment, systems thinking, domain expertise, human coordination, and accountability.

### Eight realistic pivots

| Pivot | Why it's resilient | What to build | Difficulty from SWE |
|---|---|---|---|
| **AI / ML Engineering** | You build the tools doing the disrupting | ML fundamentals, model serving, fine-tuning, evals, RAG, MLOps | Medium |
| **AI Agent / Context Engineering** | The new core craft of the agentic SDLC | Prompt & context engineering, orchestration frameworks, MCP/tool design, multi-agent systems | Low–Medium |
| **Platform / DevOps / SRE** | Someone must run the systems agents generate; reliability is human-owned | Kubernetes, IaC, observability, incident response, cost/perf | Low |
| **Security / AppSec** | The #1 risk surface of AI code (§5) — *growing* demand | Threat modeling, SAST/DAST, secure design, supply-chain security | Medium |
| **Data Engineering** | AI runs on data pipelines; durable, less-automatable plumbing | Pipelines, warehousing, streaming, data quality, governance | Low–Medium |
| **Product Management / Eng-PM hybrid** | Spec-writing *is* the new high-value work; you already speak both languages | Discovery, prioritization, stakeholder management, writing crisp specs | Medium |
| **Solutions Architecture / Technical Sales** | Translate AI capability into business value; high human-trust premium | Architecture breadth, communication, customer empathy, demos | Low–Medium |
| **Entrepreneurship / Indie building** | Agents let small teams ship what once took dozens — leverage you couldn't get before | Product sense, distribution, the full agentic SDLC, business model | High (but high-upside) |

### Adjacent industries that absorb engineering talent

- **Robotics & embodied AI** — physical-world systems where software meets hardware (still deeply human-engineered).
- **Healthcare / biotech / climate / fintech / industrial** — *domain-specialist hybrid* roles. The winning profile is **"deep domain knowledge + AI fluency."** A SWE who learns a regulated domain (clinical, energy, manufacturing) becomes far harder to automate than a generic coder.
- **AI safety, governance, and compliance** — a brand-new field needing technical people who understand both the tech and the risk.
- **Developer tooling & DevRel** — building and evangelizing the agentic tools themselves.

### How to reinvent — a 12-month plan

1. **Months 0–2 — Become agent-native.** Use Claude Code / Codex / Cursor daily on real work. Learn to spec, review, and orchestrate, not just prompt.
2. **Months 2–4 — Pick a moat direction.** Choose one: a *domain* (e.g., healthcare), a *layer* (security, platform, data), or a *craft* (agent/context engineering).
3. **Months 4–8 — Build proof.** Ship 2–3 substantive projects in that direction. Public artifacts (repos, writeups) > certificates.
4. **Months 8–12 — Reposition.** Update title/narrative toward the new value ("I orchestrate and verify systems / I secure AI-generated code / I build in X domain"), and network into it.

**The durable meta-skill:** **deliberately avoid skill atrophy** (§5). The engineers who thrive are the ones who can still *read, debug, and judge* code an agent wrote — that oversight ability is exactly what the research shows is most at risk and most in demand.

---

## 5. Pros and cons of the new approach

### Pros

- **[Verified] Real, broad capability.** AI is now used in a *majority* of engineering work at AI-native firms (~60% at Anthropic), with rapidly rising benchmark performance.
- **[Verified] Built-in test/validate loops.** Modern agents (Claude Code, Codex) run test suites in sandboxes and iterate until acceptance criteria are met — automating a tedious cycle.
- **[Industry view] Leverage & speed for routine work.** Scaffolding, boilerplate, glue code, and first drafts get dramatically faster, freeing humans for design and judgment.
- **[Industry view] Lower barrier to building.** Non-engineers and small teams can ship software, expanding who can create.
- **[Industry view] Role elevation.** Work shifts toward higher-value architecture, product thinking, and systems stewardship.

### Cons

- **[Verified] Security is the dominant risk.** Across 100+ LLMs (Veracode), **~45% of generated code samples introduced an OWASP Top 10 vulnerability**, and the **~55% security pass rate stayed essentially flat from 2025 into 2026** even as syntax correctness hit 95%+.
- **[Verified] The error profile gets *more dangerous*.** As syntax errors fell 76% and logic bugs fell 60%, **privilege-escalation paths rose 322% and architectural design flaws rose 153%** (Apiiro, across Fortune-50 repos). Surface bugs down, *deep* bugs up. **[Caveat]** single security-vendor measurement, not peer-reviewed.
- **[Verified] Skill atrophy is measurable.** In an Anthropic RCT (n=52, mostly juniors), the **AI-assisted group scored ~17 points lower on a comprehension quiz (50% vs 67%, Cohen's d=0.738, p=0.01)** on concepts used minutes earlier — **worst on debugging,** the exact oversight skill the new workflow demands. The time saved (~2 min) was *not* statistically significant. **[Caveat]** small, mostly-junior cohort, immediate-recall quiz.
- **[Verified] Maintainability / tech-debt trade-off.** Agents can "solve an immediate problem while making a codebase harder to maintain over time" — *time saved today vs. cleanup tomorrow.*
- **[Verified] Context limits persist.** Agents lose track of details and make planning mistakes in large/complex projects, keeping human verification mandatory.
- **[Industry view] Job-shape disruption.** Routine coding demand compresses; junior-pipeline and "learn by writing lots of code" career paths are most exposed.

### Claims that did NOT survive verification — do not repeat as fact

The verification pass **refuted** several widely-circulated numbers. Treat these as **unsupported**:

- ❌ "AI devs commit 3–4× faster but introduce ~10× more security findings."
- ❌ "AI reduces time-to-PR by up to 58%."
- ❌ "Senior engineers capture ~5× the productivity gains of juniors."
- ❌ "Georgia Tech confirmed 74 CVEs attributable to AI coding tools."
- ❌ Specific tight-cluster SWE-bench numbers for DeepSeek/Gemini/GPT in the low-80s%.

---

## 6. Where the room to innovate and grow is

**[Verified]** Anthropic explicitly identifies **four 2026 frontiers** that "demand immediate attention":

1. **Mastering multi-agent coordination** — getting fleets of agents to collaborate reliably.
2. **Scaling human–agent oversight through AI-automated review** — using AI to make human review tractable at agent-generated volume.
3. **Extending agentic coding beyond engineering teams** — putting agentic build tools in non-engineers' hands.
4. **Embedding security architecture from the earliest stages** — security-by-design, not security-as-final-gate.

### Concrete opportunity map

| Frontier | The gap to fill | Where to build |
|---|---|---|
| **Agent orchestration** | Reliable multi-agent frameworks, handoffs, shared memory, conflict resolution | Orchestration platforms, coordination protocols |
| **Verification & evals** | Trustworthy, codebase-specific eval harnesses; "prove the agent was right" tooling | Eval-as-a-service, test-generation that asserts *real* behavior |
| **AI-automated code review** | Review that scales to agent output volume without rubber-stamping | Smart review tools that triage what humans must see |
| **Security-by-design for AI code** | Close the flat ~55% security pass rate; catch architectural flaws agents introduce | AppSec tooling tuned for AI-generated patterns; secure scaffolds |
| **Context engineering** | Beat "context rot"; manage what large agentic sessions see | Context/memory management, repo-knowledge systems, MCP servers |
| **Cross-functional agentic tools** | Let PMs/analysts/domain experts ship safely | Guard-railed no-/low-code agentic builders |
| **Domain-specific agents** | Healthcare, finance, legal, industrial — where generic agents fail on domain rules | Vertical agentic apps with embedded domain knowledge + compliance |
| **Maintainability tooling** | Detect and prevent AI-generated tech debt before it compounds | Architecture-drift detection, refactor agents, debt scoring |

### The strategic insight

The biggest opportunities sit **exactly where the cons are.** Security, verification, oversight, and maintainability are simultaneously the **top risks** and the **clearest places to build the next generation of tools and careers.** The market need is proven by the failure data; the supply is thin. **That gap is the opportunity.**

---

## 7. Bottom line and a personal action plan

### Bottom line

Software engineering isn't disappearing — it's **inverting**. The scarce, valuable work moves from *producing* code to *directing and guaranteeing* it: specs, architecture, review, verification, security, and accountability. The capability is real and Claude currently leads the benchmarks, but autonomy is not here — **humans stay in the loop because the failure modes (security, deep architectural bugs, skill atrophy) are exactly the ones agents are worst at and humans are best at.**

### For you (a Principal Cloud Solution Architect at Microsoft)

You sit in an unusually strong position — your role is *already* the high-value, agent-resistant work (architecture, judgment, customer/business translation, accountability). To compound that:

1. **Become agent-native in your own delivery** — use agentic coding tools on reference architectures, PoCs, and customer demos to move faster and show the future credibly.
2. **Own the "agentic SDLC + security-by-design" narrative** with customers — it's the §6 frontier *and* your differentiator as an SA.
3. **Lean into verification, governance, and multi-agent orchestration** on cloud platforms — the enterprise-grade version of the innovation frontiers.
4. **Protect your own oversight skills** — keep reading and judging generated code; that's the capability the data says is both most at risk and most in demand.

---

## 8. Sources

**Primary / high-quality:**
- Anthropic — *Eight trends defining how software gets built in 2026* — https://claude.com/blog/eight-trends-defining-how-software-gets-built-in-2026
- Anthropic Research — *How AI assistance affects coding skills* (RCT) — https://www.anthropic.com/research/AI-assistance-coding-skills
- Thoughtworks — *Software engineering skills, jobs & careers in the AI era* — https://www.thoughtworks.com/en-us/insights/articles/software-engineering-skills-jobs-careers-ai-era
- Vals AI — *SWE-bench leaderboard* — https://www.vals.ai/benchmarks/swebench
- Cloud Security Alliance — *AI-generated code vulnerability surge 2026* — https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-generated-code-vulnerability-surge-2026/

**Secondary / corroborating:**
- Fast Company — *Developers are still weighing the pros and cons of AI coding agents* — https://www.fastcompany.com/91491186/developers-are-still-weighing-the-pros-and-cons-of-ai-coding-agents
- llm-stats — *SWE-bench Verified* — https://llm-stats.com/benchmarks/swe-bench-verified
- steel.dev — *SWE-bench Verified leaderboard* — https://leaderboard.steel.dev/leaderboards/swe-bench-verified/
- Veracode — *Spring 2026 GenAI Code Security Update* — https://www.veracode.com/blog/spring-2026-genai-code-security/
- Apiiro — *4x Velocity, 10x Vulnerabilities* — https://apiiro.com/blog/4x-velocity-10x-vulnerabilities/

**Practitioner playbooks (agentic SDLC, §2):**
- Provectus — *Agentic SDLC* — https://provectus.com/programs/agentic-sdlc/
- agentic-sdlc-playbook (GitHub) — https://github.com/matew17/agentic-sdlc-playbook
- DevelopersVoice — *Claude Code 2026 end-to-end SDLC* — https://developersvoice.com/blog/ai/claude_code_2026_end_to_end_sdlc/

---

### Methodology & limitations

This report was produced by a fan-out research harness: 5 search angles → 25 sources fetched → 125 claims extracted → top 25 adversarially verified (3 independent verifier votes each; a claim needed majority confirmation to survive) → 17 confirmed, 8 refuted and excluded.

**Key limitations to keep in mind:**
- **Vendor specifics beyond Claude/OpenAI/Gemini** (Amazon Q, Devin, Cursor, Windsurf, DeepSeek, Mistral, Grok, Llama) and the **career-pivot section** are *[Industry view]* synthesis — directionally sound, not independently measured here.
- **Leaderboard scores are self-reported and date within weeks.** Model names will change. Use ordering, not decimals.
- The **role-shift and "60%/0–20%" framing** comes substantially from Anthropic (interested party); **security stats** from commercial security vendors (interested parties). Both are plausible and internally consistent but not neutral.
- **Skill-atrophy evidence** is from small, mostly-junior cohorts with immediate-recall testing; generalization to seniors and long-term retention is an open question.

*Generated via Copilot deep research.*
