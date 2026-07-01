# Neoclouds: The AI-Native GPU Cloud Landscape

*A cited research report on who they are, what they do, where they're headed, and who they're hiring.*

**Prepared:** July 1, 2026
**Method:** Multi-source web research with adversarial claim verification (25 sources fetched, 125 claims extracted, 25 verified by 3-vote adversarial review — 24 confirmed, 1 refuted). Confidence levels and caveats are stated throughout because this is a fast-moving field built partly on proprietary analyst forecasts.

---

## Executive Summary

**Neoclouds are AI-native, GPU-centric cloud providers** that rent out high-performance GPU compute (GPU-as-a-Service), generative-AI platform services, and high-capacity AI data centers. They differentiate from hyperscalers (AWS, Microsoft Azure, Google Cloud) by a **tight focus on GPUs** rather than a broad portfolio of hundreds of cloud services — combined with flexible contracts, faster provisioning, and prices reportedly up to **85% below** hyperscalers.

- **Who:** More than 100 exist globally; only ~10–15 operate at meaningful US scale. The leading players by revenue are **CoreWeave, Crusoe, Lambda, Nebius, and OpenAI**. CoreWeave is the strongest direct hyperscaler competitor (Nasdaq: CRWV, IPO March 2025); OpenAI is the largest by revenue (though its inclusion as a "neocloud" is contested).
- **Future:** Explosive near-term growth on surging AI-compute demand (projected ~200 GW by 2030), but **fragile unit economics** (14–16% margins after depreciation) and **heavy customer concentration** (Microsoft was 62% of CoreWeave's 2024 revenue). Long-term survival likely hinges on moving "up the stack" into AI-native software — which pits neoclouds against the very hyperscalers who are their anchor customers today.
- **Talent:** Hiring aggressively across **two poles** — capital-intensive physical data-center buildout/manufacturing/operations, and specialized GPU-infrastructure software engineering. CoreWeave listed 272 open roles; senior AI-infra leadership roles pay **$250K–$400K base** plus bonus and equity.

---

## 1. What Is a Neocloud?

A neocloud is a **specialized, GPU-centric cloud platform built primarily for AI workloads.** Its three core offerings are:

1. **GPU-as-a-Service (GPUaaS)** — renting raw or managed GPU compute
2. **GenAI platform services** — training and inference tooling
3. **High-capacity AI data centers**

They differentiate from hyperscalers by a **tight focus on GPUs rather than a broad portfolio of cloud services**. As one comparison frames it, "AWS offers over 200 distinct services" while neoclouds "offer GPUs." *(Confidence: High. Sources: [Synergy Research](https://www.srgresearch.com/articles/neoclouds-currently-growing-by-over-200-per-year-will-reach-180-billion-in-revenues-by-2030), [McKinsey](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/the-evolution-of-neoclouds-and-their-next-moves))*

> **A note on definition:** One job-posting-derived definition claiming neoclouds are *exclusively* "early-stage venture-backed startups running live customer pilots" was **refuted (0–3 votes)** during verification — publicly traded CoreWeave and OpenAI directly contradict it. Neoclouds span startups through public companies.

---

## 2. The Major Players

More than **100 neoclouds exist globally**, but only about **10–15 operate at meaningful scale in the United States**, with a growing footprint across Europe, the Middle East, and Asia. They are typically **backed by venture capital, private equity, or sovereign-wealth capital.** *(Confidence: High. Source: [McKinsey](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/the-evolution-of-neoclouds-and-their-next-moves), Nov 2025)*

The **leading neoclouds by revenue** are CoreWeave, Crusoe, Lambda, Nebius, and OpenAI. CoreWeave is the strongest direct competitor to traditional hyperscalers, while OpenAI is the largest of the group. *(Confidence: High. Source: [Synergy Research](https://www.srgresearch.com/articles/neoclouds-currently-growing-by-over-200-per-year-will-reach-180-billion-in-revenues-by-2030))*

| Provider | Positioning | What they do | Notable facts |
|---|---|---|---|
| **CoreWeave** | "The Essential Cloud for AI" | GPU cloud platform for AI labs, startups, and enterprises; competes on infrastructure performance, not general-purpose services | Founded 2017; **Nasdaq: CRWV since March 2025**; ~$5.13B 2025 revenue; ~$2B NVIDIA stake; Nasdaq-100 component |
| **Crusoe** | "Energy-first AI factory" | **Vertically integrated** — builds and manufactures its own data centers and hardware "from electrons to tokens" | Runs in-house manufacturing (CNC, welding, modular data-center engineering); distinct from hyperscalers who rely on contract manufacturers |
| **Lambda** | "The Superintelligence Cloud" | Builds GPU supercomputers for **training and inference**; cloud GPUs, on-demand clusters, private cloud, and hardware | Explicitly positions as "not a general-purpose hyperscaler" but a "GPU-rich, AI-specific cloud" |
| **Nebius** | European-rooted AI cloud | GPU cloud and AI infrastructure | Frequently cited among the top revenue-generating neoclouds |
| **OpenAI** | AI lab / compute buyer | Classified by Synergy as the largest "neocloud" via its Stargate initiatives | **Contested classification** — most sources treat OpenAI as a compute *buyer*, not a neocloud (see Caveats) |

*(Company profiles confirmed via first-party careers/homepages fetched 2026-07-01, plus SEC filings and independent reporting. Confidence: High.)*

---

## 3. Business Model — and How It Differs From Hyperscalers

### The cost-advantage core
The defining commercial pitch is **pricing GPU compute far below hyperscalers — reportedly as much as 85% less** — which makes neoclouds attractive to smaller generative-AI startups. Concrete corroboration: one comparison put AWS H100 at ~$6.88/hr vs. a neocloud alternative at ~$2.39/hr; another found hyperscaler H100s cost 2.9–5.1× more. *(Confidence: Medium. The "85%" is an **upper bound**, not typical — real-world discounts are often quoted in the 40–85% range. Source: [McKinsey](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/the-evolution-of-neoclouds-and-their-next-moves))*

### Operational differentiators
Beyond price, neoclouds compete on **flexible contracts, faster provisioning, and specialized infrastructure configurations**. But McKinsey argues their **long-term viability hinges on moving up the stack** into AI-native software services — training orchestration, distributed inference platforms, domain-specific stacks (e.g., life sciences, financial services), developer tools, and managed ML. Doing so puts them in **direct competition with the hyperscalers who are, today, their anchor customers.** *(Confidence: High. Source: [McKinsey](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/the-evolution-of-neoclouds-and-their-next-moves))*

### Fragile unit economics
The dominant **bare-metal-as-a-service (BMaaS)** model has thin margins:
- **55–65% gross margin** *before* depreciation
- **only 14–16%** *after* labor, power, and depreciation — "lower margins than many nontech retail businesses"

*(Confidence: High on the reported figures, but note the 14–16% is McKinsey's own hedged "according to some reports" citation to The Information's reporting on internal Oracle data — not an audited industry average. Source: [McKinsey](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/the-evolution-of-neoclouds-and-their-next-moves))*

---

## 4. Market Trajectory & Future Outlook

### Demand is the tailwind; supply is the bottleneck
AI compute demand is **steep and accelerating** — training and inference workload demand is expected to reach roughly **200 gigawatts by 2030**, with **infrastructure supply, not demand, as the main bottleneck.** Bain independently projected total global compute requirements "could reach 200 gigawatts by 2030." *(Confidence: High. Sources: [McKinsey](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/the-evolution-of-neoclouds-and-their-next-moves), Bain. Nuance: sources vary on whether 200 GW is AI-specific vs. total global compute.)*

### Revenue growth (Synergy Research)
- Neocloud revenue **passed $5B in Q2 2025**, up **205% year-over-year**
- On track to **exceed $23B for full-year 2025**
- Forecast to reach **almost $180B by 2030**, at an average ~69% annual growth rate

*(Confidence: High as reported, corroborated by DataCenterDynamics, RCR Wireless, Fierce Network, and others. Caveats: pure geometric CAGR from $23B→$180B is ~51%, implying front-loaded growth; and Synergy's tally includes OpenAI, which inflates totals. Source: [Synergy Research](https://www.srgresearch.com/articles/neoclouds-currently-growing-by-over-200-per-year-will-reach-180-billion-in-revenues-by-2030))*

### Infrastructure & GPUaaS forecasts (ABI Research)
- Neocloud-operated data centers growing from **558 facilities (2025) to more than 2,200 (2035)**
- A **~$250B GPUaaS revenue opportunity by 2030**, driven by AI inference workloads and cloud-sovereignty demand

*(Confidence: Medium — these are single-firm, proprietary, non-auditable analyst projections; the revenue figure drew a split 2–1 verification vote. Source: [ABI Research](https://www.abiresearch.com/blog/neocloud-market-trends), Dec 2025)*

### The central risk: concentration & commoditization
Neoclouds currently depend heavily on the **traditional compute supply chain (semiconductors and hyperscalers) rather than enterprises**, creating customer-concentration, margin-pressure, and commoditization risk:
- **Microsoft was 62% of CoreWeave's total revenue in 2024** (up from 35% in FY2023, per CoreWeave's Form S-1)
- **NVIDIA is a key customer** for both Lambda and CoreWeave — while also being their supplier *and* investor, a "circular financing" concentration risk
- ABI's warning: neoclouds "could remain GPU brokers for hyperscalers and chipmakers, trapped in a commodity position and vulnerable to margin pressure."

*(Confidence: High. Source: [ABI Research](https://www.abiresearch.com/blog/neocloud-market-trends); underlying figures trace to CoreWeave's SEC filings.)*

**Bottom line on the future:** The demand backdrop is exceptional and near-term growth is real, but the business model is capital-intensive and thin-margined. McKinsey's likely endpoint is **consolidation into niche/specialized players** — the winners will be those who climb from renting GPUs to selling AI-native software and managed services.

---

## 5. Roles & Talent Neoclouds Are Hiring For

Neoclouds are **hiring aggressively**, reflecting hyper-growth. As of July 1, 2026:
- **CoreWeave** listed **272 open jobs** (first-party careers page)
- **Crusoe** listed **200+ open roles** across departments

*(Confidence: High — figures fetched live 2026-07-01. These are snapshots that fluctuate daily.)*

Hiring clusters around **two distinct poles**:

### Pole 1 — Physical data-center buildout & operations (asset-heavy)
A large share of hiring is for **capital-intensive infrastructure**, distinguishing neoclouds from asset-light software firms:
- **CoreWeave** roles: Data Center Construction Manager (Self-Perform), Data Center Energy Analyst, Data Center Technicians — across 8+ US sites (Afton TX, Cincinnati OH, Ellendale ND, Kenilworth NJ, Phoenix AZ, Volo IL, Mesa AZ), plus land acquisition and substation design across ~18 sites
- **Crusoe's #1 hiring category is Manufacturing** (67 roles): CNC operators, welders, modular data-center engineers — plus a Digital Infrastructure Group (51 roles)

*(Confidence: Medium on the "large share" quantifier for CoreWeave; High that Manufacturing is Crusoe's single largest category. Sources: [CoreWeave Careers](https://www.coreweave.com/careers), [Crusoe Careers](https://www.crusoe.ai/about/careers))*

### Pole 2 — Specialized GPU / AI-infrastructure software engineering
To compete on AI-optimized compute, neoclouds hire deep-specialty software engineers:
- **CoreWeave** roles: GPU Performance Engineer; Staff Software Engineer, Inference; Staff Software Engineer, Cluster Orchestration; Software Engineer II, AI Workload Orchestration; Sr GPU Infrastructure Software Engineer
- **Crusoe's largest software department is Cloud Engineering** (56 roles), spanning SDN networking, storage, cloud/hypervisor R&D, and managed Kubernetes
- Common tooling: **Slurm, Kubernetes, SUNK** for large-scale GPU-cluster performance

*(Confidence: High. Sources: [CoreWeave Careers](https://www.coreweave.com/careers), [Crusoe Careers](https://www.crusoe.ai/about/careers))*

### Skills & compensation
Senior neocloud talent commands specialized skills and premium pay:
- **Deep expertise in GPU-cluster interconnect/networking** — PCIe/NVLink topologies, InfiniBand/RoCE, distributed storage — and, for senior roles, **10+ years operating large GPU clusters**
- **Head of AI Infrastructure (Neocloud):** **$250,000–$400,000 base** plus performance bonus and meaningful equity
- **Senior GPU Infrastructure Software Engineer (CoreWeave):** **$153,000–$204,000 base** (Go/Python, production-scale Kubernetes, AI/ML training/inference)

*(Confidence: High on compensation figures, corroborated across Jobright, Levels.fyi, SimplyHired. Sources: [Head of AI Infrastructure posting](https://www.linkedin.com/jobs/view/head-of-ai-infrastructure-neocloud-at-blue-signal-search-4359895560), [CoreWeave Sr GPU Infra SWE posting](https://www.linkedin.com/jobs/view/senior-gpu-infrastructure-software-engineer-at-coreweave-4420774171))*

**The talent takeaway:** Neoclouds need an unusual blend — heavy industrial/construction/manufacturing labor to build the physical plant, plus elite distributed-systems and GPU-networking engineers to make the fleet perform. That dual demand is itself a signature of the asset-heavy, performance-differentiated neocloud model.

---

## 6. Open Questions

These remain genuinely unresolved and are worth watching:

1. **Which neoclouds move up the stack** into AI-native software and managed services versus remaining commoditized "GPU brokers" — and how much consolidation into niche players results?
2. **What is the realistic path to sustainable profitability** given thin post-depreciation BMaaS margins (14–16%), heavy debt-funded capex, and the depreciation risk of rapidly aging GPU fleets?
3. **How durable is the extreme customer/supplier concentration** and "circular financing" (NVIDIA as simultaneous supplier, investor, and customer; Microsoft/OpenAI as dominant revenue sources)? What happens if an anchor customer builds its own capacity or churns?
4. **What does the international and sovereign neocloud landscape** look like beyond US players (e.g., Nebius and providers across Europe, the Middle East, and Asia backed by sovereign-wealth capital), and how do their models, scale, and hiring differ?

---

## 7. Caveats & Confidence Notes

*(Included in the interest of accuracy, per the request.)*

- **Time-sensitivity:** Job-posting counts (CoreWeave 272; Crusoe department totals) and compensation ranges are live snapshots as of 2026-07-01 and fluctuate daily. The Head of AI Infrastructure posting was already "No longer accepting applications" when verified. Market data anchors to specific publication dates (McKinsey Nov 2025, Synergy Oct 2025, ABI Dec 2025).
- **Forecast reliability:** The most striking growth figures — Synergy's ~$180B by 2030 and ABI's $250B GPUaaS-by-2030 and 2,200 data-centers-by-2035 — are **single-firm, proprietary, non-auditable analyst projections.** ABI's revenue forecast and McKinsey's "85%-cheaper" framing each drew a split 2–1 verification vote.
- **Definitional contestation:** Synergy's inclusion of **OpenAI** as a "neocloud" (and as the group's largest member) is **not the conventional definition** — most sources treat OpenAI as a compute buyer — which materially inflates aggregate revenue figures.
- **Margin economics:** The alarming 14–16% post-depreciation margin is McKinsey's own hedged figure sourced to The Information's reporting on internal Oracle data, not an audited industry average.
- **Source mix:** Definitional, market-structure, and outlook claims rest on strong primary sources (McKinsey, Synergy, ABI, SEC filings); several are single-analyst-firm sources with secondary corroboration rather than multiple independent primary studies. Talent claims are weighted toward two US players (CoreWeave, Crusoe) plus recruiter postings, generalized to "neoclouds."

---

## 8. Sources

**Primary sources**
- McKinsey — *The evolution of neoclouds and their next moves* (Nov 19, 2025): https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/the-evolution-of-neoclouds-and-their-next-moves
- Synergy Research Group — *Neoclouds growing 200%+/yr, will reach $180B by 2030* (Oct 13, 2025): https://www.srgresearch.com/articles/neoclouds-currently-growing-by-over-200-per-year-will-reach-180-billion-in-revenues-by-2030
- ABI Research — *The State of Neocloud: Four Trends for 2026* (Dec 3, 2025): https://www.abiresearch.com/blog/neocloud-market-trends
- CoreWeave Careers: https://www.coreweave.com/careers
- Crusoe Careers: https://www.crusoe.ai/about/careers
- Lambda Careers: https://lambda.ai/careers
- LinkedIn — Head of AI Infrastructure (Neocloud): https://www.linkedin.com/jobs/view/head-of-ai-infrastructure-neocloud-at-blue-signal-search-4359895560
- LinkedIn — CoreWeave Senior GPU Infrastructure Software Engineer: https://www.linkedin.com/jobs/view/senior-gpu-infrastructure-software-engineer-at-coreweave-4420774171
- Kerrisdale Capital — CoreWeave short thesis (Sept 2025): https://www.kerrisdalecap.com/wp-content/uploads/2025/09/Kerrisdale-CoreWeave.pdf

**Secondary sources**
- RCR Tech — Neocloud explainer, main players: https://rcrtech.com/semiconductor-news/neocloud-explainer-main-players/
- Data Center Knowledge — Neoclouds vs. hyperscalers: https://www.datacenterknowledge.com/ai-data-centers/neoclouds-vs-hyperscalers-will-ai-s-specialized-clouds-prevail-
- Disintermediate — Hyperscalers vs. neoclouds: https://www.disintermediate.global/insights/hyperscalers-vs-neoclouds
- Forbes — Inside the neocloud economy / GPUaaS: https://www.forbes.com/sites/rscottraynovich/2025/11/06/inside-the-neocloud-economy-whats-next-for-gpu-as-a-service/
- FinChannel — Gartner: neoclouds to capture 20% of $267B AI cloud market by 2030: https://finchannel.com/gartner-predicts-neocloud-providers-will-capture-20-of-the-267-billion-ai-cloud-market-by-2030/131850/tech-2/2026/06/
- DataCenterDynamics — Chipping away at the economics of neoclouds: https://www.datacenterdynamics.com/en/analysis/chipping-away-at-the-economics-of-neoclouds/
- Quartz — GPU-collateralized debt & neocloud financing risks: https://qz.com/gpu-collateralized-debt-ai-neocloud-coreweave-financing-risks-050526

**Additional context (blog / opinion)**
- Forbes Business Council — Neoclouds: the rise of specialized AI-native platforms: https://www.forbes.com/councils/forbesbusinesscouncil/2025/12/01/neoclouds-the-rise-of-specialized-ai-native-platforms/
- Forbes Tech Council — How neoclouds are redefining GenAI economics: https://www.forbes.com/councils/forbestechcouncil/2025/12/11/beyond-hyperscalers-how-neoclouds-are-redefining-the-economics-of-genai/

---

*Report generated from adversarially verified research. Confidence ratings (High / Medium) and split-vote notes are preserved so you can weigh each claim. For anything you plan to cite externally, the primary sources above are the ones to quote directly.*
