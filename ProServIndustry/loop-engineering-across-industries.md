# Loop Engineering Across Industries

### Use Cases & Secure, Safety-Enabled Claude Skills

*A companion to the Loop Engineering guide. It applies the same loop-engineering discipline — iterative AI cycles bounded by the five precepts — to 30+ industries, each with concrete use cases and a ready-to-use, security-hardened Claude skill.*

---

## How to Read This Guide

Every loop in this document is built on the **five precepts** of loop engineering:

1. **Goal** — a clear, echoed-back objective.
2. **Assessment** — how each iteration decides to continue or stop.
3. **Checkpoint** — a human-in-the-loop touchpoint, mandatory before irreversible/regulated actions.
4. **Stoppage** — explicit end conditions (goal met, time, iteration cap, budget cap).
5. **Test** — a dry run before going live.

Each **Claude skill** below is written in `SKILL.md` format (YAML frontmatter + body) and inherits the shared **Security & Safety Contract** in the next section. Install one by saving its block as `…/.claude/skills/<name>/SKILL.md` in your Cowork skills folder (OneDrive `Documents/Cowork`).

---

## The Security & Safety Contract (inherited by every skill)

Because these loops touch regulated data and can take consequential actions, **every** skill in this guide operates under one shared contract. Each skill's own *Security & Safety* block lists only the deltas specific to its domain.

- **Data minimization & classification.** Identify and classify sensitive data on ingest — PII, PHI (HIPAA), cardholder data (PCI-DSS), MNPI/material non-public info, GDPR/CCPA personal data, trade secrets, ITAR/export-controlled. Pull the minimum needed; mask/tokenize where possible; **never write secrets, credentials, or raw sensitive fields into logs or summaries**.
- **Least privilege.** Default to **read-only**. Write/act scopes are granted explicitly, per-action, and are revocable. No standing access to money movement, production systems, or regulated actions.
- **Human-in-the-loop gates.** Any **irreversible, financial, clinical, legal, safety-critical, or customer-facing** action is **staged for human approval** — the loop *proposes*, a qualified human *disposes*. Autonomy is opt-in and bounded.
- **Hard stoppage & kill switch.** Every loop is bounded by time, iteration count, and a spend/compute budget, and exposes a one-command stop. No unbounded loops.
- **Audit & observability.** Log every iteration, decision, input source, and action with timestamps to an immutable trail (supports SOX, 21 CFR Part 11, GxP, FINRA, etc.). Surface a reviewable summary.
- **Grounding & anti-hallucination.** Ground claims in retrieved, cited sources; add a verification step before acting on a finding. Never fabricate data, prices, diagnoses, or legal conclusions.
- **Fairness & non-discrimination.** For loops that affect people (lending, hiring, insurance, healthcare, benefits), apply bias checks and never decide on protected characteristics.
- **Segregation of duties.** The agent that detects/recommends is not the agent that executes regulated actions; approval is by a different, authorized human.

> Throughout the skills, the phrase **"under the Security & Safety Contract"** means all of the above applies.

---

## Industry Index

| # | Industry | # | Industry |
|---|---|---|---|
| 1 | Finance & Banking | 10 | Insurance |
| 2 | Retail & E-commerce | 11 | Agriculture |
| 3 | Energy & Utilities | 12 | Food (Production, Safety, Service) |
| 4 | Transportation | 13 | Legal |
| 5 | Logistics | 14 | Strategy Consulting |
| 6 | Supply Chain | 15 | AI Labs |
| 7 | Health Care | 16 | Software Industry |
| 8 | Pharmacy | 17 | Construction |
| 9 | Life Sciences & Biotech | 18+ | Additional industries (telecom, media, real estate, education, public sector, travel, automotive, aerospace & defense, HR, marketing, cybersecurity, mining, water, nonprofit) |

---

## 1. Finance & Banking

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Fraud / AML transaction monitoring | Monitor → Compare → Act → Notify | Catch fraudulent / suspicious transactions early | Shift/window end | Analyst review before any account hold; SAR filing is human |
| Portfolio drift & rebalancing watch | Monitor → Compare → Act → Notify | Keep portfolios within mandate | Drift resolved / market close | PM approval before any trade |
| Loan covenant / credit monitoring | Poll-until-state | Detect covenant breach early | Breach handled / review cycle | Relationship manager escalation |
| Reconciliation (cash, trades, ledgers) | Poll-until-state | Match records to closure | All matched / cutoff | Controller sign-off on exceptions |

**Flagship skill**

```markdown
---
name: aml-fraud-monitoring-loop
description: >-
  Monitors transactions for fraud/AML risk and escalates suspicious activity to
  an analyst. Use when the user says "watch transactions for fraud", "monitor for
  suspicious activity", "AML monitoring", or "flag unusual transactions".
---

# AML / Fraud Monitoring Loop

## Goal
Detect potentially fraudulent or suspicious transactions and escalate for human
review — never to auto-block or file regulatory reports autonomously.

## Inputs to gather
- Transaction stream/source and account scope; risk rules & thresholds
- Poll cadence (default: near-real-time / 5 min); monitoring window
- Analyst/compliance contact for escalation

## Loop (five precepts)
1. GOAL — confirm scope and the risk typologies to watch.
2. ASSESS — score each transaction vs. rules + anomaly models.
3. ACT — on a hit, RECOMMEND a case for review and notify the analyst; do NOT
   freeze funds or file a SAR autonomously.
4. CHECKPOINT — human analyst dispositions every flag.
5. STOP — end of window; or all flags triaged.
6. TEST — replay against labeled historical transactions; tune thresholds.

## Security & Safety (under the Security & Safety Contract)
- Regimes: BSA/AML, PCI-DSS, GLBA, SOX. PII/cardholder data masked in outputs.
- No autonomous account holds, fund freezes, or SAR/CTR filings — human only.
- Full immutable audit trail of every flag and disposition.

## Output
A prioritized case queue + period summary. No customer-facing action.
```

---

## 2. Retail & E-commerce

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Dynamic pricing watch | Monitor → Compare → Act → Notify | Stay competitive within margin rules | Window end / target margin | Price floor/ceiling guardrails; approval for big moves |
| Markdown / clearance optimization | Generate → Evaluate → Refine | Clear inventory at best margin | Sell-through target / season end | Merch approval |
| Out-of-stock & replenishment alerts | Poll-until-state | Avoid lost sales | Stock restored | Buyer approval above spend cap |
| Review / sentiment monitoring | Monitor → Compare → Act → Notify | Catch product issues fast | Window end | No auto-response to customers without review |

**Flagship skill**

```markdown
---
name: dynamic-pricing-loop
description: >-
  Watches competitor prices and demand signals and proposes price changes within
  strict guardrails. Use when the user says "monitor competitor pricing", "adjust
  prices dynamically", "watch the market price", or "optimize pricing".
---

# Dynamic Pricing Loop

## Goal
Keep prices competitive and margin-positive by proposing changes within preset
floors, ceilings, and margin rules.

## Inputs to gather
- SKUs and competitor/demand data sources
- Price floor, ceiling, min margin %, max % change per step
- Cadence (default hourly/daily); approval threshold for large changes

## Loop (five precepts)
1. GOAL — confirm SKUs, guardrails, and the pricing objective.
2. ASSESS — compare current price vs. competitors, demand, and margin rules.
3. ACT — PROPOSE a new price within guardrails; auto-apply only small moves
   inside policy; large moves require approval.
4. CHECKPOINT — merchandiser approves out-of-band changes; notify on every change.
5. STOP — window end or objective met.
6. TEST — dry-run on recent data; verify no guardrail breaches.

## Security & Safety (under the Security & Safety Contract)
- Price-error guardrails (hard floor/ceiling) prevent runaway or $0 pricing.
- Antitrust/anti-collusion: use only public/owned signals; no competitor coordination.
- Customer PII never used for individualized discriminatory pricing.

## Output
Proposed/applied price changes with margin impact; change log.
```

---

## 3. Energy & Utilities

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Grid demand-response / dispatch | Plan → Execute → Check → Re-plan | Balance load vs. generation economically | Settlement period end | Operator approval; never autonomous grid control |
| Renewable generation forecasting | Monitor → Compare → Act → Notify | Optimize dispatch & trading | Day-ahead/intraday close | Trader/operator approval |
| Asset predictive maintenance (turbines, lines) | Monitor → Compare → Act → Notify | Prevent outages | Work order raised | Approval before de-energizing |
| Emissions / regulatory compliance watch | Poll-until-state | Stay within permits | Reporting cycle | Compliance sign-off |

**Flagship skill**

```markdown
---
name: grid-dispatch-advisory-loop
description: >-
  Advises on demand-response and generation dispatch to balance the grid
  economically. Use when the user says "balance the grid", "optimize dispatch",
  "demand response", or "manage load vs generation".
---

# Grid Dispatch Advisory Loop

## Goal
Recommend dispatch / demand-response actions that balance load and generation at
least cost — as decision support for a control-room operator.

## Inputs to gather
- Load forecast, generation availability, market prices, reserve requirements
- Cadence (default 5–15 min); settlement-period window
- Operator/desk for approval

## Loop (five precepts)
1. GOAL — confirm balancing objective and reserve/reliability constraints.
2. ASSESS — each cadence, evaluate forecast vs. available generation and price.
3. ACT — RECOMMEND dispatch/DR actions; stage for operator approval.
4. CHECKPOINT — operator approves; NEVER actuate physical grid assets autonomously.
5. STOP — settlement period end or balance achieved.
6. TEST — backtest against historical intervals.

## Security & Safety (under the Security & Safety Contract)
- Critical infrastructure: NERC CIP; OT/IT segmentation; no direct SCADA control.
- Reliability constraints always override economics.
- Operator-in-the-loop is mandatory for every actionable recommendation.

## Output
Ranked dispatch recommendations with cost/reliability impact; interval log.
```

---

## 4. Transportation

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Dynamic route re-optimization | Plan → Execute → Check → Re-plan | Keep routes optimal amid traffic/incidents | End of day / no disruption | Dispatcher approval for major reroutes |
| Vehicle predictive maintenance | Monitor → Compare → Act → Notify | Prevent breakdowns | Work order raised | Approval before pulling a vehicle |
| Delay monitoring & passenger comms | Monitor → Compare → Act → Notify | Inform riders promptly | Disruption cleared | Review tone before mass notifications |
| Driver hours-of-service compliance | Poll-until-state | Keep drivers legal & safe | Shift end | Hard stop on HOS violations |

**Flagship skill**

```markdown
---
name: fleet-route-reoptimization-loop
description: >-
  Re-optimizes fleet routes when traffic, weather, or incidents disrupt the plan.
  Use when the user says "re-route the fleet", "optimize routes in real time", or
  "reschedule deliveries when traffic changes".
---

# Fleet Route Re-Optimization Loop

## Goal
Keep delivery/transit routes optimal (time, cost, on-time %) as conditions change,
within driver-safety and hours-of-service limits.

## Inputs to gather
- Fleet, stops, time windows, vehicle constraints; traffic/weather feeds
- HOS limits; cadence (default 15 min + on disruption); day window
- Dispatcher for approval

## Loop (five precepts)
1. GOAL — confirm objective and hard constraints (time windows, HOS, vehicle).
2. ASSESS — re-optimize on cadence and on disruption events.
3. ACT — PROPOSE updated routes with before/after ETA & cost; stage for dispatcher.
4. CHECKPOINT — dispatcher approves major reroutes; drivers notified safely.
5. STOP — end of day or no disruption for N hours.
6. TEST — replay yesterday's disruptions.

## Security & Safety (under the Security & Safety Contract)
- Driver PII and location data minimized and access-controlled.
- HOS / safety constraints are hard limits — never violated to save time.
- No route that directs drivers into closed/unsafe conditions.

## Output
Proposed routes + KPIs; dispatcher-approved changes logged.
```

---

## 5. Logistics

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Shipment tracking & exception management | Poll-until-state | Deliver on time; resolve exceptions fast | Delivered / exception closed | Approval before costly expedites |
| Carrier rate shopping | Monitor → Compare → Act → Notify | Best rate + service | Booking window / shipped | Approval before booking |
| Warehouse labor / slotting optimization | Plan → Execute → Check → Re-plan | Maximize throughput | Shift end | Supervisor approval |
| Customs / trade-docs compliance | Poll-until-state | Avoid holds & penalties | Cleared | Compliance review |

**Flagship skill**

```markdown
---
name: shipment-exception-loop
description: >-
  Tracks shipments and manages exceptions (delays, holds, damage) to closure. Use
  when the user says "track my shipments", "watch for delivery exceptions", or
  "escalate stuck shipments".
---

# Shipment Exception Loop

## Goal
Drive every shipment to on-time delivery and resolve exceptions quickly via
recommended actions.

## Inputs to gather
- Shipment IDs / lanes and tracking sources; SLA/ETA targets
- Cadence (default hourly); expedite spend cap
- Ops contact for escalations

## Loop (five precepts)
1. GOAL — confirm shipments in scope and SLA targets.
2. ASSESS — poll status; detect exceptions vs. expected milestones.
3. ACT — on exception, RECOMMEND a remedy (reroute, expedite, notify); auto-act
   only under the spend cap, else escalate.
4. CHECKPOINT — ops approves costly remedies; customer comms reviewed.
5. STOP — delivered, or window end.
6. TEST — dry-run on recent exception cases.

## Security & Safety (under the Security & Safety Contract)
- Trade-compliance and customer data handled per contract; no auto customs filings.
- Expedite spend bounded by cap; larger spend needs approval.

## Output
Exception queue with recommended actions; resolution log.
```

---

## 6. Supply Chain

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Multi-tier supplier risk monitoring | Monitor → Compare → Act → Notify | Spot disruptions early (weather, news, finance) | Review cycle | Verify sources; no auto contract action |
| Demand–supply balancing | Plan → Execute → Check → Re-plan | Match supply to demand | Planning cycle | Planner approval |
| Network inventory positioning | Generate → Evaluate → Refine | Right stock, right place | Cycle end | Planner approval |
| Supplier price / lead-time watch | Monitor → Compare → Act → Notify | Best cost & availability | Sourcing window | Approval before PO |

**Flagship skill**

```markdown
---
name: supplier-risk-monitoring-loop
description: >-
  Monitors multi-tier suppliers for disruption signals and escalates risks. Use
  when the user says "watch supplier risk", "monitor my suppliers for disruptions",
  or "alert me to supply chain risks".
---

# Supplier Risk Monitoring Loop

## Goal
Detect supplier/disruption risks early (financial distress, weather, geopolitical,
quality) and escalate with verified evidence.

## Inputs to gather
- Supplier list (tiers), critical parts, regions; signal sources (news, weather, financial)
- Cadence (default daily); risk thresholds; escalation owner

## Loop (five precepts)
1. GOAL — confirm suppliers/parts in scope and risk categories.
2. ASSESS — scan signals; score risk; correlate to affected parts/lines.
3. ACT — on elevated risk, RECOMMEND mitigations (dual-source, buffer) and notify.
4. CHECKPOINT — human verifies sources; no autonomous PO/contract changes.
5. STOP — review cycle end or risk cleared.
6. TEST — replay against a past disruption.

## Security & Safety (under the Security & Safety Contract)
- Source verification required before escalation (anti-rumor / anti-hallucination).
- Third-party data used per agreements; no autonomous commercial actions.

## Output
Ranked risk register with evidence and mitigation options.
```

---

## 7. Health Care

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Patient deterioration early warning | Monitor → Compare → Act → Notify | Alert clinicians before decline (e.g., sepsis) | Event resolved / shift end | Clinician decides; no autonomous treatment |
| Bed / capacity management | Plan → Execute → Check → Re-plan | Optimize flow | Shift end | Charge nurse approval |
| Claims pre-auth / denial follow-up | Poll-until-state | Reduce denials & delays | Resolved | Staff review before resubmission |
| Care-gap & follow-up reminders | Accumulate-until-target | Close care gaps | Outreach complete | Care-team review |

**Flagship skill**

```markdown
---
name: patient-deterioration-monitoring-loop
description: >-
  Monitors patient vitals/labs for deterioration and alerts clinicians as decision
  support. Use when the user says "monitor patient vitals", "early warning for
  deterioration", "watch for sepsis", or "alert the care team".
---

# Patient Deterioration Monitoring Loop

## Goal
Provide clinical DECISION SUPPORT — alert clinicians to early signs of patient
deterioration. It never diagnoses or treats autonomously.

## Inputs to gather
- Patient cohort/unit and vitals/labs source; early-warning criteria (e.g., MEWS/NEWS)
- Cadence (default continuous/15 min); shift window; clinician to alert

## Loop (five precepts)
1. GOAL — confirm cohort and the early-warning criteria.
2. ASSESS — each cadence, score vitals/labs vs. criteria and trend.
3. ACT — on threshold/trend, ALERT the clinician with the evidence; recommend
   escalation — never order tests, meds, or treatment autonomously.
4. CHECKPOINT — clinician makes all clinical decisions.
5. STOP — event resolved or shift end.
6. TEST — validate against retrospective cases.

## Security & Safety (under the Security & Safety Contract)
- HIPAA/PHI: minimum-necessary, encrypted, access-logged; de-identify in summaries.
- Positioned as Clinical Decision Support; respect FDA CDS boundaries — informational,
  with the clinician able to independently review the basis.
- No autonomous clinical action of any kind.

## Output
Prioritized clinician alerts with the contributing factors; audit log.
```

---

## 8. Pharmacy

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Medication safety / interaction checks | Generate → Evaluate → Refine | Catch interactions & dosing errors | Review complete | Pharmacist verifies; no auto-dispense |
| Controlled-substance inventory monitoring | Poll-until-state | Detect diversion & shortages | Reconciled | DEA compliance; pharmacist sign-off |
| Refill adherence outreach | Accumulate-until-target | Improve adherence | Outreach done | Review before patient contact |
| Cold-chain medication monitoring | Monitor → Compare → Act → Notify | Protect drug efficacy | Excursion handled | QA hold approval |

**Flagship skill**

```markdown
---
name: medication-safety-check-loop
description: >-
  Screens prescriptions for interactions, allergies, and dosing issues for
  pharmacist review. Use when the user says "check for drug interactions", "screen
  prescriptions", or "medication safety check".
---

# Medication Safety Check Loop

## Goal
Flag potential drug–drug/drug–allergy interactions and dosing concerns for a
pharmacist — never to dispense or alter therapy autonomously.

## Inputs to gather
- Prescription queue and patient med/allergy data source; interaction knowledge base
- Cadence (per Rx / batch); pharmacist for verification

## Loop (five precepts)
1. GOAL — confirm screening scope and severity thresholds.
2. ASSESS — screen each Rx vs. interactions, allergies, renal/hepatic dosing.
3. ACT — flag concerns with severity + evidence for pharmacist review.
4. CHECKPOINT — pharmacist verifies and decides; no autonomous dispensing.
5. STOP — queue cleared or shift end.
6. TEST — validate against known interaction cases.

## Security & Safety (under the Security & Safety Contract)
- HIPAA/PHI safeguards; DEA rules for controlled substances; e-signature/audit (21 CFR Part 11).
- Pharmacist sign-off mandatory; never auto-dispense or modify a prescription.

## Output
Per-Rx safety flags ranked by severity; verification log.
```

---

## 9. Life Sciences & Biotech

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Pharmacovigilance signal detection | Monitor → Compare → Act → Notify | Detect adverse-event safety signals | Review cycle | Safety team adjudicates; regulated reporting is human |
| Assay / experiment optimization (DOE) | Generate → Evaluate → Refine | Maximize assay performance | Spec met / budget | Scientist review |
| Literature & patent monitoring | Accumulate-until-target | Stay current on the field | Coverage met | Researcher review |
| Clinical-trial recruitment & monitoring | Poll-until-state | Hit enrollment & data quality | Targets met | Investigator oversight |

**Flagship skill**

```markdown
---
name: pharmacovigilance-signal-loop
description: >-
  Monitors adverse-event reports for safety signals and escalates to the safety
  team. Use when the user says "monitor adverse events", "detect safety signals",
  or "pharmacovigilance monitoring".
---

# Pharmacovigilance Signal Loop

## Goal
Detect potential drug-safety signals in adverse-event data and escalate for human
adjudication — never to make regulatory submissions autonomously.

## Inputs to gather
- AE data sources (internal safety DB, literature, public sources); products in scope
- Signal criteria/disproportionality thresholds; cadence (default daily/weekly)
- Safety/PV team for adjudication

## Loop (five precepts)
1. GOAL — confirm products and signal-detection criteria.
2. ASSESS — scan AE data; compute disproportionality / trend signals.
3. ACT — escalate candidate signals with evidence to the PV team.
4. CHECKPOINT — safety team adjudicates; no autonomous regulatory filing.
5. STOP — review cycle end.
6. TEST — replay against historical, known signals.

## Security & Safety (under the Security & Safety Contract)
- GxP / 21 CFR Part 11: validated, audit-trailed, e-signatures; patient privacy.
- Regulatory submissions (e.g., expedited reports) are strictly human-performed.

## Output
Candidate signal list with evidence and disproportionality metrics.
```

---

## 10. Insurance

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Claims triage & fraud detection | Monitor → Compare → Act → Notify | Route claims correctly; flag fraud | Triage done | Adjuster decides; no auto-deny |
| Underwriting risk monitoring | Poll-until-state | Keep books within appetite | Review cycle | Underwriter approval |
| Renewal / churn watch | Monitor → Compare → Act → Notify | Retain good risks | Renewal cycle | Agent review before outreach |
| Catastrophe exposure monitoring | Monitor → Compare → Act → Notify | Manage accumulation risk | Event passed | Risk-team escalation |

**Flagship skill**

```markdown
---
name: claims-fraud-triage-loop
description: >-
  Triages incoming claims and flags suspicious ones for SIU, fairly and
  transparently. Use when the user says "triage claims", "flag claim fraud", or
  "route suspicious claims".
---

# Claims Fraud Triage Loop

## Goal
Prioritize claims and surface suspicious ones for Special Investigations — never
to deny or approve claims autonomously.

## Inputs to gather
- Claims queue and data sources; fraud indicators; routing rules
- Cadence (per claim/batch); SIU/adjuster contacts

## Loop (five precepts)
1. GOAL — confirm triage rules and fraud indicators.
2. ASSESS — score each claim for severity, complexity, and fraud risk.
3. ACT — route to fast-track / adjuster / SIU with rationale; flag, don't decide.
4. CHECKPOINT — adjuster/SIU makes all coverage decisions.
5. STOP — queue cleared.
6. TEST — replay against adjudicated claims; check fairness.

## Security & Safety (under the Security & Safety Contract)
- PII/PHI safeguards; fair-lending/anti-discrimination — no protected-class signals.
- No autonomous claim denial/approval; decisions are explainable and auditable.

## Output
Triaged claim queue with risk rationale; routing log.
```

---

## 11. Agriculture

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Irrigation optimization | Monitor → Compare → Act → Notify | Right water, right time | Season / target moisture | Water-limit guardrails; approval for large systems |
| Pest / disease early detection | Monitor → Compare → Act → Notify | Catch outbreaks early | Treated / season end | Agronomist review before spraying |
| Harvest timing & yield | Generate → Evaluate → Refine | Maximize yield/quality | Harvest done | Farm-manager decision |
| Commodity price watch (selling) | Monitor → Compare → Act → Notify | Sell at good prices | Sold / window end | Approval before contracts |

**Flagship skill**

```markdown
---
name: irrigation-optimization-loop
description: >-
  Recommends irrigation based on soil moisture, weather, and crop stage. Use when
  the user says "optimize irrigation", "when should I water", or "monitor soil
  moisture".
---

# Irrigation Optimization Loop

## Goal
Recommend irrigation timing/volume to meet crop water needs while respecting water
limits and equipment safety.

## Inputs to gather
- Fields, crop type/stage; soil-moisture sensors + weather forecast
- Water budget/allocation limits; cadence (default hourly/daily)
- Approval owner for large-system actuation

## Loop (five precepts)
1. GOAL — confirm fields, crop needs, and water limits.
2. ASSESS — combine soil moisture, ET, and forecast to compute need.
3. ACT — RECOMMEND irrigation schedule; auto-run small zones within limits, else
   request approval.
4. CHECKPOINT — manager approves large/atypical actions; respect allocations.
5. STOP — season end or moisture target met.
6. TEST — dry-run against last week's sensor/weather data.

## Security & Safety (under the Security & Safety Contract)
- Hard water-allocation caps; never exceed permitted withdrawals.
- Equipment-safety interlocks; no autonomous actuation of large pumps without approval.

## Output
Irrigation recommendations per field with water-use projections.
```

---

## 12. Food (Production, Safety, Service)

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Food-safety / cold-chain (HACCP) monitoring | Monitor → Compare → Act → Notify | Keep CCPs in range | Excursion handled / shift | QA approves any product hold/release |
| Perishable demand & waste reduction | Generate → Evaluate → Refine | Cut spoilage, meet demand | Cycle end | Manager approval on markdowns |
| Quality / contamination inspection | Generate → Evaluate → Refine | Keep defect/contam rate low | In-spec | QA approval before disposition |
| Recall / advisory monitoring | Poll-until-state | React fast to recalls | Resolved | Compliance action is human |

**Flagship skill**

```markdown
---
name: food-safety-coldchain-loop
description: >-
  Monitors HACCP critical control points and cold-chain temperatures, holding
  product on excursions. Use when the user says "monitor cold chain", "watch food
  safety temps", or "HACCP monitoring".
---

# Food Safety / Cold-Chain Loop

## Goal
Keep critical control points (temperature, time) within safe limits and flag
excursions for QA disposition — never auto-release flagged product.

## Inputs to gather
- CCPs and temperature/time data sources; safe limits per product
- Cadence (default 5–15 min); QA contact

## Loop (five precepts)
1. GOAL — confirm CCPs and safe limits.
2. ASSESS — each cadence, compare readings vs. limits; detect excursions/trends.
3. ACT — on an excursion, RECOMMEND a product HOLD and notify QA; log the event.
4. CHECKPOINT — QA decides hold/release; no autonomous release of held product.
5. STOP — shift end or excursion resolved.
6. TEST — replay against logged excursions.

## Security & Safety (under the Security & Safety Contract)
- FDA FSMA / HACCP compliance; tamper-evident audit trail for every CCP reading.
- Product disposition (hold/release) is always a human QA decision.

## Output
CCP status, excursion alerts, and a shift food-safety log.
```

---

## 13. Legal

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Contract obligation & renewal monitoring | Poll-until-state | Never miss an obligation/deadline | Cycle end | Attorney reviews; no autonomous execution |
| Litigation deadline / docket watch | Poll-until-state | Hit every filing deadline | Matter closed | Attorney files; agent only alerts |
| Regulatory change monitoring | Monitor → Compare → Act → Notify | Stay compliant as law changes | Review cycle | Counsel assesses impact |
| E-discovery review assist | Generate → Evaluate → Refine | Find relevant/privileged docs | Review complete | Attorney privilege review |

**Flagship skill**

```markdown
---
name: contract-obligation-monitoring-loop
description: >-
  Tracks contract obligations, renewals, and deadlines and alerts the legal team.
  Use when the user says "track contract obligations", "watch renewal dates", or
  "monitor contract deadlines".
---

# Contract Obligation Monitoring Loop

## Goal
Ensure no contractual obligation, renewal, or deadline is missed — by alerting,
never by executing or signing anything.

## Inputs to gather
- Contract repository and key dates/obligations; alert lead times
- Cadence (default daily); responsible attorney/owner

## Loop (five precepts)
1. GOAL — confirm contracts in scope and obligation types.
2. ASSESS — poll for upcoming obligations/renewals/deadlines vs. lead times.
3. ACT — alert the owner with the obligation, source clause, and due date.
4. CHECKPOINT — attorney decides and acts; no autonomous notices/filings/signatures.
5. STOP — review cycle end.
6. TEST — dry-run on the current contract set.

## Security & Safety (under the Security & Safety Contract)
- Attorney–client privilege & confidentiality strictly preserved; access-controlled.
- No autonomous legal action (notice, filing, execution); grounded in actual clauses,
  with citations — no fabricated legal conclusions.

## Output
Obligation calendar with alerts and source-clause citations.
```

---

## 14. Strategy Consulting

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Competitive / market intelligence monitoring | Monitor → Compare → Act → Notify | Stay ahead of market moves | Engagement cycle | Verify sources; client-confidential |
| Research synthesis (iterate to coverage) | Accumulate-until-target | Comprehensive, cited synthesis | Coverage / gaps closed | Consultant review |
| Benchmarking data collection | Accumulate-until-target | Complete benchmark set | Targets met | Analyst QA |
| Deliverable QA loop | Generate → Evaluate → Refine | Polished, accurate output | Quality bar met | Partner review |

**Flagship skill**

```markdown
---
name: competitive-intel-monitoring-loop
description: >-
  Monitors competitors and markets and synthesizes cited intelligence briefs. Use
  when the user says "monitor competitors", "track market moves", or "competitive
  intelligence".
---

# Competitive Intelligence Monitoring Loop

## Goal
Continuously gather and synthesize competitor/market developments into verified,
cited briefs.

## Inputs to gather
- Competitors/markets/topics; approved public sources
- Cadence (default daily/weekly); engagement/client context

## Loop (five precepts)
1. GOAL — confirm targets and the intelligence questions.
2. ASSESS — scan sources; filter for materiality and recency.
3. ACT — synthesize a brief with citations; flag what's uncertain.
4. CHECKPOINT — consultant reviews before client delivery.
5. STOP — cycle end or coverage achieved.
6. TEST — dry-run on a known week; check source quality.

## Security & Safety (under the Security & Safety Contract)
- Client confidentiality and information-barrier rules; no MNPI/insider misuse.
- Public/licensed sources only; every claim cited; no fabrication.

## Output
A cited intelligence brief with a "watch list" of open questions.
```

---

## 15. AI Labs

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Training-run monitoring | Monitor → Compare → Act → Notify | Catch divergence/waste early | Run end / budget | Approval to kill or scale a run |
| Hyperparameter / eval optimization | Generate → Evaluate → Refine | Best model under budget | Budget / target metric | Researcher review |
| Safety / red-team evaluation | Generate → Evaluate → Refine | Surface failures pre-release | Coverage met | Safety review gate before release |
| Dataset quality monitoring | Poll-until-state | Clean, compliant data | Quality bar met | Data-governance review |

**Flagship skill**

```markdown
---
name: training-run-monitor-loop
description: >-
  Monitors training runs for divergence, stalls, and waste, recommending early
  stops. Use when the user says "watch my training run", "monitor model training",
  or "alert me if training diverges".
---

# Training Run Monitor Loop

## Goal
Watch training metrics and recommend interventions (early stop, checkpoint, scale)
to avoid wasted compute — within a hard compute budget.

## Inputs to gather
- Run/metric source (loss, eval, throughput, hardware); healthy ranges
- Compute budget cap; cadence (default per N steps / minutes); owner

## Loop (five precepts)
1. GOAL — confirm success metrics and the compute budget.
2. ASSESS — track loss/eval/throughput vs. healthy ranges and trends.
3. ACT — on divergence/stall/waste, RECOMMEND early-stop/restart/scale and alert.
4. CHECKPOINT — researcher approves kills or scale-ups (cost-significant actions).
5. STOP — run completes or compute budget reached.
6. TEST — replay against a prior run's logs.

## Security & Safety (under the Security & Safety Contract)
- Compute-spend governance: hard budget cap; no autonomous large compute allocation.
- Model safety: no autonomous deployment; release passes a separate safety-eval gate.

## Output
Run-health timeline with recommended interventions; decision log.
```

---

## 16. Software Industry

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Deploy health & auto-rollback | Plan → Execute → Check → Re-plan | Safe releases | Stable / rolled back | Rollback within policy; approval for fixes-forward |
| Incident detection & triage (SRE) | Monitor → Compare → Act → Notify | Detect & route incidents fast | Resolved | On-call approves remediations |
| Dependency vulnerability monitoring | Poll-until-state | Patch known CVEs | Patched | Approval before prod changes |
| Cloud cost / performance optimization | Generate → Evaluate → Refine | Cut cost, keep SLOs | Target met | Owner approves resource changes |

**Flagship skill**

```markdown
---
name: deploy-health-monitor-loop
description: >-
  Watches post-deploy metrics and recommends or triggers a policy-bounded rollback.
  Use when the user says "monitor the deploy", "watch release health", or "roll back
  if the deploy is bad".
---

# Deploy Health Monitor Loop

## Goal
Confirm a release is healthy and, if not, recover quickly — favoring a safe
rollback over autonomous forward changes.

## Inputs to gather
- Service + key SLIs (error rate, latency, saturation); healthy thresholds
- Bake time / cadence; rollback policy; on-call owner

## Loop (five precepts)
1. GOAL — confirm the SLIs and pass/fail thresholds for this release.
2. ASSESS — during the bake window, compare metrics vs. baseline.
3. ACT — if unhealthy, trigger a rollback IF within policy; otherwise alert on-call.
   Never apply ad-hoc forward fixes to prod autonomously.
4. CHECKPOINT — on-call approves anything beyond a policy rollback.
5. STOP — healthy past bake time, or rolled back.
6. TEST — dry-run against a prior bad-deploy signature.

## Security & Safety (under the Security & Safety Contract)
- Least-privilege CI/CD credentials; secrets never logged.
- Change-management compliance; production writes limited to policy-approved rollback.
- Full audit of detections and actions.

## Output
Release-health verdict and any rollback action, with a timeline.
```

---

## 17. Construction

**Use cases**

| Use case | Loop pattern | Goal | Stop condition | Human checkpoint / safety |
|---|---|---|---|---|
| Schedule / critical-path monitoring | Plan → Execute → Check → Re-plan | Keep the project on track | Phase/project end | PM approves schedule changes |
| Site safety monitoring (vision/IoT) | Monitor → Compare → Act → Notify | Prevent incidents | Shift end | Safety officer escalation; OSHA |
| Material delivery & inventory | Poll-until-state | Avoid delays/shortages | Delivered | Approval above spend cap |
| Budget / cost-overrun monitoring | Monitor → Compare → Act → Notify | Catch overruns early | Cycle end | PM/finance review |

**Flagship skill**

```markdown
---
name: project-schedule-monitoring-loop
description: >-
  Tracks construction progress vs. plan and flags slippage on the critical path.
  Use when the user says "monitor the project schedule", "track construction
  progress", or "alert me to schedule slippage".
---

# Project Schedule Monitoring Loop

## Goal
Keep the project on schedule by detecting slippage early and proposing recovery
options for the PM.

## Inputs to gather
- Project schedule (tasks, dependencies, critical path); progress source
- Cadence (default daily); slippage thresholds; PM owner

## Loop (five precepts)
1. GOAL — confirm the baseline schedule and critical path.
2. ASSESS — compare actual progress vs. plan; recompute the critical path.
3. ACT — on slippage, PROPOSE recovery (re-sequence, crews, expedite) with impact.
4. CHECKPOINT — PM approves schedule/commitment changes.
5. STOP — phase/project complete.
6. TEST — replay against last month's progress data.

## Security & Safety (under the Security & Safety Contract)
- Site-safety findings escalate immediately to the safety officer (OSHA alignment).
- Worker PII minimized; no autonomous financial/contractual commitments.

## Output
Schedule-health report with slippage alerts and recovery options.
```

---

## 18+. Additional Industries (others worth covering)

Full use-case tables; each names a starter skill you can expand on request (same `SKILL.md` pattern and Security & Safety Contract).

| Industry | High-value loop use case | Starter skill | Key safety/security note |
|---|---|---|---|
| **Telecom** | Network anomaly/outage detection & capacity watch | `network-anomaly-monitoring-loop` | No autonomous network config changes; NOC approval |
| **Media & Entertainment** | Content moderation & trend monitoring | `content-moderation-loop` | Human review for takedowns; appeals; bias checks |
| **Real Estate** | Listing/price & lease-portfolio monitoring | `property-market-watch-loop` | No autonomous offers; fair-housing compliance |
| **Education** | Student early-warning & intervention | `student-early-warning-loop` | FERPA; counselor-in-the-loop; no labeling students |
| **Government / Public Sector** | Benefits-fraud & permit-processing monitoring | `benefits-integrity-loop` | Due process; transparency; no autonomous denials |
| **Hospitality & Travel** | Dynamic rate & rebooking watch (the classic hotel loop) | `travel-rebooking-loop` | Reversible bookings; approval before charges |
| **Automotive** | Connected-vehicle telemetry & recall monitoring | `vehicle-telemetry-loop` | Safety-critical; OEM approval; driver privacy |
| **Aerospace & Defense** | Fleet predictive maintenance & parts (ITAR) | `aero-maintenance-loop` | ITAR/export controls; airworthiness sign-off |
| **HR / Talent** | Candidate sourcing & attrition-risk watch | `talent-pipeline-loop` | EEOC fairness; no protected-class signals; human hiring decisions |
| **Marketing / Advertising** | Campaign optimization & brand-safety watch | `campaign-optimization-loop` | Hard budget caps; brand-safety guardrails; privacy |
| **Cybersecurity** | Threat detection & response (SOAR) | `threat-response-loop` | Auto-contain only within policy; analyst approval to isolate/block |
| **Mining & Metals** | Equipment predictive maintenance & grade control | `mining-asset-loop` | OT safety; operator approval for equipment actions |
| **Water Utilities** | Leak detection & demand monitoring | `water-network-loop` | Critical infrastructure; operator approval for valve/pump actions |
| **Nonprofit / NGO** | Donor engagement & grant-compliance monitoring | `grant-compliance-loop` | Donor PII; no autonomous outreach without review |

> **Cybersecurity note (SOAR):** of all the "additional" loops, automated threat *response* is the most action-capable. Keep auto-containment strictly within a pre-approved playbook (e.g., quarantine a non-critical endpoint), require analyst approval to block accounts or isolate production systems, and log everything for the SOC.

---

## Cross-Cutting Guidance

**To stand up any of these safely:**

1. **Start read-only.** Run the loop in observe/recommend mode first; grant action scopes only after it's proven on historical replays.
2. **Keep the human disposing.** For anything regulated, financial, clinical, legal, or safety-critical, the loop *proposes* and a qualified human *approves*.
3. **Bound everything.** Time window, iteration cap, and spend/compute budget on every loop, plus a one-command kill switch.
4. **Audit by default.** Immutable logs of inputs, decisions, and actions — your evidence for SOX, HIPAA, 21 CFR Part 11, FINRA, GxP, and internal review.
5. **Ground and verify.** Cite sources; add a verification step before acting; never fabricate.
6. **Check fairness.** Wherever a loop affects people, screen for bias and exclude protected characteristics.

**Common compliance regimes referenced:** HIPAA, PCI-DSS, GLBA, SOX, GDPR/CCPA, FINRA/SEC, FDA 21 CFR Part 11 & GxP, FDA FSMA/HACCP, NERC CIP, OSHA, FERPA, EEOC/fair-lending, ITAR/EAR.

---

## Want These as Live Skills?

Every skill above is copy-paste-ready as a `SKILL.md`. I can also **materialize any or all of them as installed Cowork skills** (in your OneDrive `Documents/Cowork` skills folder) so they're invocable with `/<skill-name>` — and I can expand any of the "Additional industries" starter skills into a full definition. Just tell me which ones.

*Companion to "Loop Engineering: A Practical Guide" and its manufacturing section. Compiled June 24, 2026.*
