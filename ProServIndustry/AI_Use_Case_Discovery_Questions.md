# AI Use Case Discovery Questions

**Purpose:** Comprehensive discovery questions to understand a customer's use case and align the right AI-based solution — agentic AI, generative AI, or a hybrid approach.

---

## 1. Strategic Questions

| # | Question |
|---|---------|
| 1 | What is the overarching business goal this initiative supports? (e.g., revenue growth, cost reduction, market expansion, customer retention) |
| 2 | Why now? What has changed in your business or market that makes this use case urgent? |
| 3 | How does this use case align with your company's 1–3 year digital transformation roadmap? |
| 4 | Who is the executive sponsor, and what does success look like from their perspective? |
| 5 | What happens if you do nothing — what is the cost of inaction? |
| 6 | Are there competitive pressures driving this? Are competitors already using AI for similar workflows? |
| 7 | Is this a net-new capability or an improvement to an existing process? |
| 8 | How does this initiative fit within your broader AI/ML strategy — is this a pilot, a scale-up, or a standalone effort? |
| 9 | What is your organization's risk appetite for AI adoption? (e.g., conservative/regulated vs. move-fast) |
| 10 | What does the long-term vision look like — if this succeeds, what do you want AI to do for you in 2–3 years? |

---

## 2. Business Process & Value Questions

| # | Question |
|---|---------|
| 11 | Walk me through the current end-to-end process this use case would affect — who does what, in what order? |
| 12 | Where are the biggest bottlenecks, manual steps, or pain points in that process today? |
| 13 | How many people are involved in this workflow, and how much time do they spend on it weekly? |
| 14 | What is the current error rate or quality issue in this process? |
| 15 | How do you measure success today — what KPIs or metrics matter most? (e.g., throughput, accuracy, time-to-resolution, cost per transaction) |
| 16 | What is the expected ROI or business case? Have you quantified the value of automating or augmenting this workflow? |
| 17 | Who are the end users of this solution — what are their roles, skill levels, and daily tools? |
| 18 | What is the volume and frequency of this task? (e.g., 500 documents/day, 10,000 customer inquiries/month) |
| 19 | Are there seasonal or demand spikes that the solution needs to handle? |
| 20 | What decisions are being made as part of this process, and what is the impact of a wrong decision? |
| 21 | Does this workflow require human-in-the-loop approval, or can decisions be fully automated? |
| 22 | Are there compliance, regulatory, or audit requirements tied to this process? (e.g., HIPAA, SOX, GDPR, industry-specific) |

---

## 3. Data & Content Questions

| # | Question |
|---|---------|
| 23 | What data sources feed into this use case? (e.g., databases, documents, APIs, emails, images, logs) |
| 24 | Where does this data live today — on-prem, cloud, SaaS, or a mix? |
| 25 | How structured or unstructured is the data? (e.g., tabular data vs. PDFs, emails, free-text notes) |
| 26 | What is the data quality like — is it clean, labeled, and consistent, or does it need significant preparation? |
| 27 | Do you have domain-specific knowledge bases, SOPs, or institutional knowledge that the AI would need to reference? |
| 28 | How sensitive or confidential is the data? Are there data residency, sovereignty, or classification requirements? |
| 29 | What is the expected input and output of the AI? (e.g., input = customer email, output = drafted response + ticket classification) |

---

## 4. Technical & Architecture Questions

| # | Question |
|---|---------|
| 30 | What is your current tech stack and cloud environment? (Azure, AWS, hybrid, on-prem) |
| 31 | Do you have existing AI/ML infrastructure — model serving, MLOps pipelines, vector databases, or prompt management? |
| 32 | What integration points are required? (e.g., CRM, ERP, ticketing systems, SharePoint, Teams, custom APIs) |
| 33 | Do you need the AI to take actions autonomously (agentic), or primarily generate content and recommendations (generative)? |
| 34 | Does this use case require multi-step reasoning, tool use, or orchestration across multiple systems — or is it a single-turn generation task? |
| 35 | What are your latency and throughput requirements? (e.g., real-time response < 2 seconds vs. batch processing overnight) |
| 36 | How do you plan to handle model updates, prompt versioning, and drift monitoring over time? |
| 37 | Do you need fine-tuning on proprietary data, or will retrieval-augmented generation (RAG) with your knowledge base suffice? |
| 38 | What authentication, authorization, and role-based access controls are needed? |
| 39 | Are there existing APIs or connectors to the systems the AI agent would need to interact with? |

---

## 5. Governance, Trust & Change Management Questions

| # | Question |
|---|---------|
| 40 | How will you evaluate whether the AI output is accurate and trustworthy before going live? |
| 41 | What is your plan for responsible AI — bias testing, fairness, transparency, and explainability? |
| 42 | Who owns the AI solution post-deployment — IT, the business unit, or a shared model? |
| 43 | How will you handle user adoption and change management — are end users excited, skeptical, or unaware? |
| 44 | What is your rollout plan — big bang, phased by region/team, or a controlled pilot first? |
| 45 | Do you have a feedback loop planned so end users can flag bad outputs and the system improves over time? |

---

## 6. Qualifying the AI Approach

These questions help determine whether the solution should lean toward **agentic AI**, **generative AI**, or a hybrid:

| # | Question | Signal |
|---|---------|--------|
| 46 | Does the use case require the AI to plan, decide, and execute across multiple steps with minimal human input? | → **Agentic AI** |
| 47 | Is the primary need content creation, summarization, or transformation of information? | → **Generative AI** |
| 48 | Does the AI need to interact with external tools, APIs, or systems to complete the task? | → **Agentic AI with tool use** |
| 49 | Is there a well-defined, repeatable workflow — or does the task require adaptive reasoning based on context? | → Helps size orchestration complexity |
| 50 | What level of autonomy are you comfortable giving the AI — fully autonomous, semi-autonomous with approvals, or human-supervised? | → Determines guardrail design |

---

## How to Use This Guide

- Use these as a **conversation guide**, not a rigid checklist — pick the questions most relevant to the customer's maturity and context.
- **Key questions for shaping your AI recommendation:** 11, 20, 29, 33, and 46–50 are the most telling for deciding between agentic vs. generative AI.
- **Start broad, go deep:** Open with strategic questions (1–10), then drill into the process (11–22) before getting technical (30–39).
- **Listen for signals:** The customer's answers will naturally reveal whether they need an autonomous agent, a content generation tool, or a combination of both.

---

*Prepared for use case discovery and AI solution alignment — Agentic AI & Generative AI*
