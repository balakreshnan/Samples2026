# Loop Engineering: A Practical Guide

*A detailed look at one of the newest trends in generative and agentic AI — what it is, why it matters, where to use it, and how to do it well.*

---

## 1. What Is Loop Engineering?

**Loop engineering** is the deliberate, careful design of *iterative cycles* that an AI performs on a human's behalf — instructing the AI to keep working on a task, repeating a set of steps, until a specified condition or final state is reached.

It contrasts with the way most people use AI today: a **turn-by-turn**, one-question-one-answer exchange (sometimes called "one-and-done" or "one-shot"). In the traditional model, you and the AI take turns; you ask, it answers, you ask again. Loop engineering instead instructs the AI to **run a continuous loop** and lets you step partly *out* of the loop while it works behind the scenes.

A useful working definition (adapted from Dr. Lance Eliot's strawman definition in *Forbes*, June 2026):

> *"Loop engineering is the mindful design of iterative cycles that AI is to perform on behalf of a human so that a stipulated task is undertaken and will produce results aligned with a suitably stated goal. This contrasts with a typical one-and-done or one-shot approach of using AI. Loops can be established for AI agents and can also be used with conventional turn-by-turn AI."*

> **Why is it called "engineering"?**
> Because you are not supposed to just set up a loop willy-nilly. A loop is powerful — like starting a flywheel. If your specification has loopholes or gaps, the AI will faithfully follow flawed instructions and the result can be a mess (runaway costs, unintended actions, or a loop that never stops). The "engineering" emphasizes deliberate design: cover all the bases, anticipate edge cases, and make sure the loop *will eventually stop*.

It is closely related to **prompt engineering** — it's essentially prompt engineering applied to a *workflow* rather than a single response.

---

## 2. Loop-Centric vs. Prompt-Centric Thinking

The core mental shift in loop engineering is moving from a **prompt-centric** mindset to a **workflow-centric** one.

| Dimension | Prompt-Centric (traditional) | Loop-Centric (loop engineering) |
|---|---|---|
| **Interaction** | Turn-by-turn; you ask, AI answers | AI runs a continuous loop on your behalf |
| **Your role** | In the loop for every step | Optionally out of the loop ("fire and forget") |
| **Unit of design** | A single prompt | A series of tasks (a workflow) |
| **Failure blast radius** | Small — one bad answer | Large — a runaway cycle can spin out |
| **Key question** | "How do I word this prompt?" | "What is the sequence of tasks, and when does it stop?" |
| **Cost profile** | Predictable, per-turn | Can balloon if iterations aren't bounded |

The trade-off: one-shot prompts are *safe* (one bad prompt rarely causes real harm), but limited. Loops are *powerful* but riskier — a poorly bounded loop is like a flywheel that keeps cycling and can spin out of orbit.

---

## 3. Why Use Loop Engineering?

Loop engineering exists because many valuable tasks are **not** "ask once and done." They require ongoing monitoring, repeated attempts, or continuous reassessment against changing conditions. Loop engineering lets the AI do that diligent, repetitive watching so you don't have to.

**Key benefits:**

- **Continuous, autonomous work** — the AI keeps working "tirelessly" on your goal without you re-prompting it each time.
- **You stay out of the busywork** — set it up once ("fire and forget") instead of logging back in repeatedly to nudge the AI forward.
- **Captures opportunities you'd otherwise miss** — e.g., a better deal that appears at 2 a.m. while you're asleep.
- **Handles changing conditions** — the loop re-checks the world on each iteration and reacts when something changes.
- **Scales effort** — repetitive monitoring that would exhaust a human is trivial for an AI loop.

**The catch (why "engineering" is essential):**

- A loop with no frequency limit might iterate every nanosecond for days, running up compute charges that dwarf any benefit.
- A loop with no human checkpoint might take irreversible actions you'd never approve (see the seedy-hotel example below).
- A vague loop can be interpreted differently than you intended, or the AI can hallucinate mid-loop and drift off course.

---

## 4. Where to Use Loop Engineering

Loop engineering is **primarily focused on AI agents (agentic AI)** — agents can take real-world actions (book, cancel, buy, send), so looping over those actions is where it shines. It can *also* be applied to conventional generative AI, though that is harder today and will get easier as models advance.

### Good fits for a loop

- **Monitor-and-act tasks** — watch a price, an inventory level, a job board, or a calendar slot, and act when a condition is met (e.g., keep hunting for a better hotel deal and rebook automatically).
- **Iterate-until-quality tasks** — keep drafting/refining a document, image, or piece of code until it meets a defined quality bar.
- **Polling / status-watching** — repeatedly check whether a long-running process (a deploy, a shipment, an approval) has reached a target state, then notify or proceed.
- **Search-and-optimize** — repeatedly search a space of options and converge on the best one under your constraints.
- **Continuous data gathering** — periodically collect and summarize new information (news, mentions, filings) over a defined window.

### Poor fits (keep it turn-by-turn)

- One-off questions or single answers ("What's the capital of France?").
- Tasks where every step needs your judgment anyway (no autonomy to gain).
- High-stakes irreversible actions with no safe way to insert a checkpoint.
- Anything where you can't define a clear stopping condition.

---

## 5. The Five Precepts of Loop Engineering

These five precepts, drawn from Eliot's framing, are the checklist for designing a sound loop:

1. **Make a goal for the loop.** Establish a clear, unambiguous goal and verify that the AI correctly echoes back its understanding of that goal before it starts.

2. **Provide a loop assessment mechanism.** Give the AI a way to assess each iteration — to decide whether to keep looping or stop. (How does it *measure* progress toward the goal?)

3. **Include a human feedback checkpoint.** Build in human-AI checkpoints so that, while looping, the AI keeps a human informed and gives an opportunity for course-correction or shutdown. This is the safest practice — especially before irreversible actions.

4. **Establish the loop stoppage.** Define explicit rules for when the loop ends: goal attained, time window elapsed, resource/budget cap hit, or a max number of iterations reached. *Every loop must be able to terminate.*

5. **Test the loop and adjust.** No matter how clever the loop seems, test it on a few quick iterations first, inspect what happens, and refine. Don't "let it loose" until you're confident.

> **Mnemonic:** **G-A-C-S-T** — **G**oal, **A**ssessment, **C**heckpoint, **S**toppage, **T**est.

---

## 6. Worked Examples

### Example A — A *non-loop* (traditional turn-by-turn) for contrast

> **User:** "I am going to be in Boston this coming weekend and need a suitable hotel. What are my options?"
> **AI:** "There are dozens of hotels in Boston. Do you have a price range? Any preferred neighborhood?"
> **User:** "$150–$250 per night. Near the aquarium preferred."
> **AI:** "I found a hotel four blocks from the aquarium, $200/night, available this weekend. Should I book it?"

Here you must keep coming back. The next day a friend mentions a better deal, so you log back in and ask the AI to cancel and rebook. It works — but *you* did all the watching and re-prompting. That's the pain loop engineering removes.

### Example B — The same task *as a loop*

> **User:** "I want you to book a hotel in Boston for my trip this weekend. My price range is $150–$250. I prefer a hotel near the aquarium. If you find such a hotel, make a reservation that **can be canceled**. Then **loop over the next 48 hours, hourly**, looking for a better deal. If you find one, book it and cancel the prior reservation. **Send me a message whenever you book a better deal.** The loop is to **end after 48 hours**. First do a **quick test** with me to confirm the loop is sensible and complete."

Notice how this single prompt encodes all five precepts:

| Precept | Where it appears in the prompt |
|---|---|
| **Goal** | Book a Boston hotel near the aquarium, $150–$250, best available deal |
| **Assessment** | "looking for a *better* deal" (the comparison test each hour) |
| **Checkpoint** | "Send me a message whenever you book a better deal" |
| **Stoppage** | "The loop is to end after 48 hours" + hourly cadence |
| **Test** | "First do a quick test with me to confirm the loop is sensible" |

The author also notes a deliberate design choice: he took *himself out of the decision portion* of the loop (the AI may book and cancel on its own, only notifying him). If that felt too risky, the safer variant is to require **permission before acting**: *"…if you find a better deal, ask me before canceling and rebooking."*

### Example C — How the same loop goes wrong ("loopy" loops)

- **No frequency limit:** "find me a better deal" with no cadence → the AI loops every nanosecond for days; compute charges far exceed any discount. *Fix: specify cadence ("hourly") and a window ("48 hours").*
- **No human checkpoint before irreversible action:** the AI finds a cheaper but seedy-reputation hotel, cancels your good reservation, and books it. Now the original is gone. *Fix: add a permission checkpoint before cancel/rebook, or add a quality constraint (rating ≥ 4 stars).*

---

## 7. Loop Engineering in Manufacturing

Manufacturing is one of the strongest fits for loop engineering. Factories already think in terms of **continuous monitoring, closed-loop control, and act-on-condition** — exactly the workflow-centric mindset loops require. The difference is that an AI loop can watch sensors, vision feeds, supplier prices, and schedules *simultaneously and tirelessly*, then act or escalate when a condition is met.

> **Safety first in industrial settings.** On a factory floor, "irreversible action" can mean scrapping a batch, halting a line, or physically changing machine parameters. In Operational Technology (OT) environments, default to **human checkpoints and permission gates** before any physical actuation, and keep a human "in the loop" rather than "out of the loop" until the loop is proven. Loops should *recommend or stage* changes that an operator approves, especially early on.

### Manufacturing use cases at a glance

| # | Use case | Loop pattern | Goal | Typical stop condition | Checkpoint before acting? |
|---|---|---|---|---|---|
| 1 | **Predictive maintenance** | Monitor → Compare → Act → Notify | Catch equipment failure before it happens | Anomaly resolved, or work order raised | Notify; auto-create work order, human approves shutdown |
| 2 | **Visual quality inspection + process correction** | Generate → Evaluate → Refine | Keep defect rate below target | Defect rate < threshold for N parts | Operator approves parameter change |
| 3 | **Closed-loop process tuning (APC)** | Plan → Execute → Check → Re-plan | Hold output in spec / maximize yield | Output in spec & stable, or iteration cap | Yes — engineer signs off on setpoints |
| 4 | **Raw-material / component procurement** | Monitor → Compare → Act → Notify | Secure best price + lead time | PO placed, or sourcing window ends | Approval gate before issuing PO |
| 5 | **Inventory & replenishment** | Poll-until-state | Avoid line stoppage from stockouts | Reorder placed / stock above min | Auto-reorder under cap; human above cap |
| 6 | **Production schedule re-optimization** | Plan → Execute → Check → Re-plan | Keep schedule optimal as conditions change | End of shift / no better schedule found | Planner approves the re-sequenced plan |
| 7 | **Energy / utilities optimization** | Monitor → Compare → Act → Notify | Lower energy cost per unit | Target met, or shift ends | Notify; stage changes for facility approval |
| 8 | **Digital-twin / simulation design** | Accumulate-until-target | Find a design that meets all specs | Spec met, or simulation budget spent | Engineer reviews candidate designs |

### Example D — Predictive maintenance loop (worked)

A CNC machine's spindle is monitored for vibration and temperature. You want the AI to watch continuously and act before a failure causes unplanned downtime.

> **User:** "Monitor the vibration and temperature telemetry for spindle **#CNC-07**. **Every 5 minutes**, compare the readings against the baseline and our alarm thresholds. **If vibration trends above 4.5 mm/s or temperature exceeds 75 °C, OR you detect a rising trend projected to breach those limits within 2 hours**, do the following: (1) raise a maintenance work order in the CMMS tagged 'predictive', (2) message the shift supervisor with the readings and the projected failure window, and (3) recommend — but **do not execute** — a spindle slowdown or stop; wait for supervisor approval before any line action. **Continue looping for the duration of the shift (8 hours)**, then send an end-of-shift summary. Run a **short 3-cycle test** against last week's logged data first so I can confirm the thresholds and alerts behave correctly."

Mapped to the five precepts:

| Precept | Where it appears |
|---|---|
| **Goal** | Prevent unplanned spindle failure on #CNC-07 |
| **Assessment** | 5-minute comparison vs. baseline + trend projection |
| **Checkpoint** | Supervisor notified; **approval required** before any stop/slowdown |
| **Stoppage** | End of 8-hour shift; or condition handled via work order |
| **Test** | 3-cycle dry run against last week's logged data |

### Example E — Component procurement loop (the "hotel example," industrialized)

The hotel-rebooking loop maps almost directly onto **strategic sourcing**: keep hunting for a better supplier deal on a critical component, but never commit money without a gate.

> **User:** "We need **5,000 units of bearing part #BR-2245** delivered to Plant 3 by the 15th. Place a provisional order with our approved vendor that **can be canceled** within 24 hours. Then **loop twice daily for the next 5 days**, checking approved suppliers for a better total-landed-cost offer that still **meets our quality spec and on-time delivery**. If you find one, **ask me to approve** before switching vendors and canceling the prior PO. Message me on every candidate found, and **stop after 5 days or once the part ships**. Test the loop on a dry run first and show me what a switch decision would look like."

Note the manufacturing-specific guardrails baked in: a **cancelable provisional PO** (reversibility), **quality-spec and delivery constraints** (so "cheaper but worse" never wins — the industrial analog of the seedy-hotel trap), and an **explicit approval gate** before moving money.

### Example F — Closed-loop quality + parameter correction

> **User:** "On the injection-molding line, **inspect every molded part** via the vision system. Track the **defect rate over a rolling 50-part window**. **Goal: keep defect rate below 2%.** If it rises above 2%, diagnose the likely cause (flash, short shot, warping) and **propose** a corrective parameter adjustment (melt temperature, hold pressure, cooling time). **Stage** the adjustment for operator approval — do **not** change machine setpoints automatically. After an approved change, keep looping and confirm the defect rate falls back below target within 50 parts; if it doesn't after **3 adjustment cycles**, **stop and escalate** to the process engineer. Send a summary at end of run."

This is a **Generate → Evaluate → Refine** loop with a hard iteration cap (3 adjustment cycles) and an escalation path — so the loop can't endlessly chase a problem it can't fix.

### Example G — Production schedule re-optimization (worked, in depth)

A plant rarely runs the schedule it started the day with. Machines go down, materials arrive late, rush orders jump the queue, and an operator calls in sick. Each disruption ripples through every downstream work center. Re-planning by hand is slow and is usually obsolete by the time it's published — which is exactly why this is a high-value loop: the AI re-optimizes the *remaining* schedule the moment conditions change and proposes an updated plan for the planner to approve.

> **User:** "Monitor the live production schedule for **Plant 3** across all work centers. **Every 15 minutes — and immediately on any disruption event** (machine down, material shortage, rush order, labor change), pull machine status, OEE, open work orders, due dates, and material availability from the MES/ERP. When a disruption is detected, **re-optimize the remaining schedule to minimize total weighted tardiness and changeover time**, while respecting **machine capacity, sequence-dependent setups, material availability, and committed ship dates**. Present the proposed re-sequenced plan with a **before/after comparison** (on-time %, makespan, number of changeovers) and **wait for the planner's approval before committing it to the MES**. **Loop until end of the production day; stop early if there are no disruptions for 4 hours.** Run a **dry run on yesterday's logged disruptions** first so I can confirm the objective and constraints are right."

Mapped to the five precepts:

| Precept | Where it appears |
|---|---|
| **Goal** | Keep the Plant 3 schedule optimal (min weighted tardiness + changeovers) as conditions change |
| **Assessment** | Re-optimize on a 15-min poll *and* on disruption events; compare new plan vs. current |
| **Checkpoint** | Planner reviews before/after and **approves before the MES is updated** |
| **Stoppage** | End of production day; or 4 hours with no disruption |
| **Test** | Dry run replayed against yesterday's logged disruptions |

Why it's tricky (and why "engineering" matters here): scheduling is a constrained optimization problem, so the loop must optimize against an **explicit objective function** and a **complete constraint set** — otherwise it will happily produce a "better" schedule that violates a setup rule or breaks a ship-date commitment. Keep the planner in the loop until the objective and constraints are proven on historical replays.

#### Skills to use — Production schedule re-optimization

- **People & roles:** Production planner/scheduler (owns approval), operations research (OR) specialist, MES/automation engineer, supply/materials planner.
- **Domain knowledge:** Finite-capacity scheduling, job-shop / flow-shop sequencing, Theory of Constraints (bottleneck management), sequence-dependent setup/changeover modeling, due-date and priority rules.
- **Systems & data:** MES (e.g., Opcenter, Rockwell, DELMIA Apriso), APS/planning (SAP PP/DS, Kinaxis, Blue Yonder, o9), ERP (SAP/Oracle) for orders & due dates, real-time machine status / OEE feeds, materials/inventory availability.
- **Technical & AI skills:** Mathematical optimization — mixed-integer linear programming (MILP) with solvers like Gurobi, CPLEX, or open-source OR-Tools; constraint programming; metaheuristics (genetic algorithms, simulated annealing, tabu search) for large instances; discrete-event simulation to validate plans; optionally reinforcement learning for dynamic dispatching.
- **Loop-engineering craft:** Event-driven triggering (re-optimize on disruption, not just on a timer), a precise objective function, a hard cap on solve time per iteration, and a **mandatory approval gate** before writing back to the MES.

---

### Skills to use — for every manufacturing use case

These are the competencies, tools, systems, and roles a team needs to *engineer and operate* each loop. Think of them in five layers: **people**, **domain knowledge**, **systems & data** (the manufacturing/OT systems to connect), **technical & AI skills**, and **loop-engineering craft** (the precept-level design choices unique to that loop).

#### 1. Predictive maintenance
- **People & roles:** Reliability/maintenance engineer, data scientist, OT/controls engineer.
- **Domain knowledge:** Reliability-Centered Maintenance (RCM), FMEA / failure modes, vibration & condition-monitoring analysis, remaining-useful-life (RUL) thinking.
- **Systems & data:** IoT/sensor telemetry (vibration, temperature, acoustic, motor current), data historian (AVEVA/OSIsoft PI), CMMS (IBM Maximo, SAP PM) for work orders, edge gateways.
- **Technical & AI skills:** Time-series anomaly detection, trend/RUL modeling, signal processing (FFT), threshold + predictive alerting.
- **Loop-engineering craft:** Threshold *and* trend-projection logic, automatic work-order creation, and an approval gate before any line slowdown/stop.

#### 2. Visual quality inspection + process correction
- **People & roles:** Quality engineer, machine-vision engineer, process engineer.
- **Domain knowledge:** Statistical Process Control (SPC), defect taxonomy, root-cause analysis (fishbone / 5-Whys), Cp/Cpk capability.
- **Systems & data:** Industrial vision cameras & lighting, MES, quality/LIMS database, PLC for parameter setpoints.
- **Technical & AI skills:** Computer vision / deep-learning defect classification, rolling-window SPC, closed-loop control logic.
- **Loop-engineering craft:** Rolling defect-rate window, **staged** parameter changes with operator approval, hard cap on adjustment cycles before escalation.

#### 3. Closed-loop process tuning (Advanced Process Control)
- **People & roles:** Process engineer, control-systems engineer, OR/optimization specialist.
- **Domain knowledge:** Advanced Process Control (APC), Model Predictive Control (MPC), Design of Experiments (DOE), process-stability and spec-limit concepts.
- **Systems & data:** SCADA/DCS, PLCs, historian, soft sensors.
- **Technical & AI skills:** MPC, Bayesian optimization, reinforcement learning for control, surrogate/soft-sensor modeling.
- **Loop-engineering craft:** Explicit spec limits + stability checks, iteration caps, and engineer sign-off on new setpoints before actuation.

#### 4. Raw-material / component procurement
- **People & roles:** Sourcing/procurement manager, supply-chain analyst, commodity buyer.
- **Domain knowledge:** Strategic sourcing, total-landed-cost analysis, supplier qualification & risk, incoterms/lead-time management.
- **Systems & data:** ERP (SAP/Oracle), e-procurement & supplier portals, EDI feeds, market/commodity price data.
- **Technical & AI skills:** Price/lead-time monitoring, multi-criteria decision analysis (cost vs. quality vs. delivery), agentic web/portal automation.
- **Loop-engineering craft:** Cancelable provisional PO (reversibility), quality + on-time-delivery constraints so "cheaper-but-worse" can't win, approval gate before issuing/switching a PO.

#### 5. Inventory & replenishment
- **People & roles:** Inventory/materials planner, supply-chain analyst.
- **Domain knowledge:** Reorder point & EOQ, safety-stock and service-level math, JIT / Kanban, demand variability.
- **Systems & data:** ERP/WMS, inventory database, demand forecasts, supplier lead-time data.
- **Technical & AI skills:** Demand forecasting, reorder-quantity optimization, exception detection.
- **Loop-engineering craft:** Min/max thresholds, auto-reorder under a spend cap with human approval above it, stockout-risk alerting.

#### 6. Production schedule re-optimization
- *See the full skills breakdown under **Example G** above.* In short: OR/optimization skills (MILP, constraint programming, metaheuristics), MES/APS/ERP integration, scheduling-theory domain knowledge, and an event-driven loop with a planner approval gate.

#### 7. Energy / utilities optimization
- **People & roles:** Facility/energy manager, sustainability engineer, controls engineer.
- **Domain knowledge:** Energy management (ISO 50001), demand response, peak-shaving & load-shifting, tariff structures.
- **Systems & data:** Energy meters/submeters, Building Management System (BMS), SCADA, utility tariff & real-time price feeds.
- **Technical & AI skills:** Load forecasting, optimization under tariff/peak constraints, scheduling of flexible loads.
- **Loop-engineering craft:** Cost-per-unit objective with throughput/comfort constraints, staged changes for facility approval, time-bounded by shift/tariff period.

#### 8. Digital-twin / simulation-driven design
- **People & roles:** Simulation/design engineer, R&D engineer, data scientist.
- **Domain knowledge:** Physics-based modeling, FEA/CFD, generative & parametric design, DOE.
- **Systems & data:** Digital-twin platforms (Azure Digital Twins, Ansys Twin Builder, Siemens), CAD/CAE tools, simulation engines.
- **Technical & AI skills:** Surrogate modeling, Bayesian/evolutionary optimization, parametric sweep automation.
- **Loop-engineering craft:** Clear spec acceptance criteria, a simulation compute/iteration budget as the stop condition, and engineer review of candidate designs before adoption.

> **Cross-cutting skills (needed for *every* loop above):** loop/prompt engineering and agent orchestration; data engineering and OT/IT integration (OPC UA, MQTT, REST APIs into MES/ERP/historian); MLOps/monitoring to watch the loop itself; and — most important in a plant — **industrial safety and change-management governance** so that physical actions always pass through a human checkpoint until the loop has earned trust.

---

### Why manufacturing loops pay off

- **Downtime avoided** — predictive loops catch failures before they stop the line.
- **Quality held automatically** — inspection-and-correct loops keep defect rates in spec without constant operator attention.
- **Cost captured continuously** — procurement and energy loops grab savings opportunities the moment they appear.
- **Resilience to change** — scheduling and replenishment loops re-react as machines, orders, and inventory shift.

The same cautions apply, amplified: bound every loop by time, iterations, and budget; constrain on **quality and safety**, not just cost; and keep an operator approving physical actions until the loop has earned trust.

---

## 8. Ready-to-Use Claude Skills for the Manufacturing Loops

Below are eight **Claude/Cowork skill definitions** — one per manufacturing use case — that you can drop in and reuse. Each is written in the standard `SKILL.md` format: YAML frontmatter (a `name` and a `description` with trigger phrases that tell Claude *when* to activate the skill), followed by a body that encodes the loop's **five precepts** (Goal, Assessment, Checkpoint, Stoppage, Test) plus the inputs to gather and the safety gates.

**How to install one:** save the block as `SKILL.md` inside a matching folder — e.g. `…/.claude/skills/predictive-maintenance-loop/SKILL.md` — in your personal Cowork skills location (your OneDrive `Documents/Cowork`). Once installed, you can invoke it with `/<skill-name>` or just describe the task and Claude will trigger it. *(Want these as live files instead of text? I can create them for you — just ask.)*

> **Universal safety rule (baked into every skill):** physical or irreversible actions — stopping a line, changing a machine setpoint, issuing a PO, committing a schedule — are **recommended/staged for human approval**, never executed autonomously, until you explicitly relax the gate.

### Skill 1 — Predictive Maintenance Loop

```markdown
---
name: predictive-maintenance-loop
description: >-
  Runs a predictive-maintenance monitoring loop over equipment telemetry and
  raises a work order before failure. Use when the user says "monitor [machine]
  for failures", "watch the spindle vibration/temperature", "set up predictive
  maintenance", or "alert me before [asset] fails".
---

# Predictive Maintenance Loop

## Goal
Detect impending equipment failure early and raise a maintenance work order
before unplanned downtime occurs.

## Inputs to gather (ask only if not provided or discoverable)
- Asset/equipment ID and telemetry source (historian or IoT tags)
- Alarm thresholds + baseline values; trend-projection horizon (default 2h)
- Polling cadence (default: every 5 minutes)
- Monitoring window (default: one 8h shift)
- Who to notify (supervisor) and the CMMS for work orders

## Loop (five precepts)
1. GOAL — confirm the asset and the failure condition; echo it back.
2. ASSESS — each cadence, compare readings vs. baseline + thresholds; project trend.
3. ACT — if a threshold is breached OR a trend projects a breach within the horizon:
   create a "predictive" CMMS work order; notify the supervisor with readings +
   projected failure window; RECOMMEND (do not execute) a slowdown/stop.
4. CHECKPOINT — notify a human on every alert; never actuate equipment autonomously.
5. STOP — at end of window, or once the condition is handled.
6. TEST — dry-run 3 cycles on last week's logged data; confirm thresholds first.

## Output
Per-alert notifications + an end-of-shift summary.

## Safety / stoppage
Never stop or slow a line without operator approval. Bound by time window and a
max alert count.
```

### Skill 2 — Quality Inspection & Process-Correction Loop

```markdown
---
name: quality-inspection-loop
description: >-
  Runs a vision-based quality-inspection loop that tracks defect rate and stages
  corrective parameter changes. Use when the user says "inspect every part",
  "keep defects under [X]%", "watch the defect rate", or "correct the process
  when quality drifts".
---

# Quality Inspection & Process-Correction Loop

## Goal
Keep the defect rate below a target by inspecting parts and proposing corrective
parameter adjustments.

## Inputs to gather
- Line/station and vision data source; defect taxonomy
- Target defect rate (default 2%) and rolling window size (default 50 parts)
- Adjustable parameters (e.g., melt temp, hold pressure, cooling time)
- Max adjustment cycles before escalation (default 3); escalation contact

## Loop (five precepts)
1. GOAL — confirm the target defect rate and the rolling window.
2. ASSESS — track defect rate over the rolling window after each inspected part.
3. ACT — if rate exceeds target: diagnose likely cause and PROPOSE a parameter
   change; STAGE it for operator approval (do not change setpoints automatically).
4. CHECKPOINT — operator approves each change; confirm rate falls back below target.
5. STOP — if not resolved after the max adjustment cycles, STOP and escalate to
   the process engineer. End at end of run.
6. TEST — replay against a recent production log first.

## Output
Defect-rate trend, each proposed correction, and an end-of-run summary.

## Safety / stoppage
No autonomous setpoint changes. Hard cap on adjustment cycles, then escalate.
```

### Skill 3 — Closed-Loop Process Tuning (APC)

```markdown
---
name: process-control-loop
description: >-
  Iteratively tunes process setpoints to hold output in spec / maximize yield,
  staging recommendations for engineer sign-off. Use when the user says "tune the
  process", "hold output in spec", "optimize yield", or "advanced process control".
---

# Closed-Loop Process Tuning (APC)

## Goal
Keep process output within spec and/or maximize yield by recommending setpoint
adjustments.

## Inputs to gather
- Process unit and controlled/manipulated variables; spec limits
- Stability criteria; sampling cadence; iteration cap
- Engineer who signs off on setpoints

## Loop (five precepts)
1. GOAL — confirm spec limits and the optimization objective (in-spec / yield).
2. ASSESS — each cadence, evaluate output vs. spec and stability.
3. ACT — RECOMMEND adjusted setpoints (MPC/optimization); STAGE for sign-off.
4. CHECKPOINT — engineer approves before any actuation.
5. STOP — when output is in spec and stable, or the iteration cap is reached.
6. TEST — validate on historical data / a soft-sensor sandbox first.

## Output
Recommended setpoints with predicted effect; convergence summary.

## Safety / stoppage
Never write setpoints to the DCS/PLC without sign-off. Iteration cap enforced.
```

### Skill 4 — Procurement Watch Loop

```markdown
---
name: procurement-watch-loop
description: >-
  Watches approved suppliers for a better total-landed-cost offer on a part and
  proposes a switch behind an approval gate. Use when the user says "find the best
  price for [part]", "watch suppliers for a better deal", or "keep sourcing [item]".
---

# Procurement Watch Loop

## Goal
Secure the best total-landed-cost offer for a component that still meets quality
and on-time-delivery requirements.

## Inputs to gather
- Part number, quantity, required delivery date, destination plant
- Quality spec + on-time-delivery constraint; approved supplier list
- Watch cadence (default twice daily) and sourcing window (default 5 days)
- Spend approval threshold

## Loop (five precepts)
1. GOAL — confirm part, quantity, due date, and quality/delivery constraints.
2. ASSESS — each cadence, check approved suppliers for a better total-landed-cost
   offer that still meets spec + delivery.
3. ACT — place a CANCELABLE provisional PO up front; on a better qualifying offer,
   ASK for approval before switching vendors and canceling the prior PO.
4. CHECKPOINT — human approves any money-moving action; notify on every candidate.
5. STOP — after the sourcing window, or once the part ships.
6. TEST — dry-run and show what a switch decision would look like.

## Output
Candidate offers, switch recommendations, and a final sourcing summary.

## Safety / stoppage
Reversible provisional PO only; cost can never outweigh quality/delivery constraints.
```

### Skill 5 — Inventory Replenishment Loop

```markdown
---
name: inventory-replenishment-loop
description: >-
  Polls inventory levels and reorders to prevent line stoppage. Use when the user
  says "watch stock for [item]", "prevent stockouts", "auto-reorder when low", or
  "monitor inventory levels".
---

# Inventory Replenishment Loop

## Goal
Avoid line stoppages from stockouts by reordering before stock falls below the
minimum.

## Inputs to gather
- SKU/material and location; min (reorder point) and max/target levels
- Supplier + lead time; reorder spend cap for auto-action
- Poll cadence (default hourly)

## Loop (five precepts)
1. GOAL — confirm the SKU, reorder point, and target level.
2. ASSESS — each cadence, read on-hand + in-transit vs. the reorder point.
3. ACT — if below reorder point: place a replenishment order to reach target IF
   under the spend cap; otherwise request human approval.
4. CHECKPOINT — notify on every reorder; approval required above the cap.
5. STOP — stock restored above min, or end of monitoring window.
6. TEST — replay against recent consumption data first.

## Output
Reorder events and a stock-position summary.

## Safety / stoppage
Auto-reorder only under the spend cap; human approval above it.
```

### Skill 6 — Schedule Re-Optimization Loop

```markdown
---
name: schedule-reoptimization-loop
description: >-
  Re-optimizes the production schedule on disruptions and proposes an updated plan
  for planner approval. Use when the user says "re-optimize the schedule",
  "reschedule when a machine goes down", or "keep the production plan optimal".
---

# Schedule Re-Optimization Loop

## Goal
Keep the production schedule optimal (minimize weighted tardiness + changeovers)
as conditions change.

## Inputs to gather
- Plant/line and work centers; MES/ERP/APS data sources
- Objective (default: min weighted tardiness + changeover time)
- Constraints: capacity, sequence-dependent setups, material availability, ship dates
- Poll cadence (default 15 min) + disruption event triggers; solve-time cap

## Loop (five precepts)
1. GOAL — confirm the objective function and the full constraint set.
2. ASSESS — each cadence AND on any disruption (machine down, material short,
   rush order, labor change), pull live status and re-optimize the remaining schedule.
3. ACT — present the re-sequenced plan with a before/after comparison (on-time %,
   makespan, changeovers); WAIT for planner approval before committing to the MES.
4. CHECKPOINT — planner approves before any MES write-back.
5. STOP — end of production day, or 4h with no disruption.
6. TEST — dry-run against yesterday's logged disruptions; confirm objective/constraints.

## Output
Proposed schedules with before/after KPIs; end-of-day summary.

## Safety / stoppage
Never write to the MES without planner approval. Cap solve time per iteration.
```

### Skill 7 — Energy Optimization Loop

```markdown
---
name: energy-optimization-loop
description: >-
  Monitors energy use against tariffs and stages load-shifting actions to cut cost
  per unit. Use when the user says "optimize energy use", "reduce energy cost",
  "shift load off peak", or "watch our utility spend".
---

# Energy Optimization Loop

## Goal
Lower energy cost per unit produced while meeting throughput targets.

## Inputs to gather
- Metered loads/equipment; BMS/SCADA data; utility tariff + real-time price feed
- Throughput and comfort/safety constraints
- Cadence (default 15 min); shift/tariff-period window

## Loop (five precepts)
1. GOAL — confirm the cost-per-unit objective and throughput constraints.
2. ASSESS — each cadence, evaluate load vs. tariff/peak signals and forecast.
3. ACT — RECOMMEND load-shift / peak-shave actions; STAGE for facility approval.
4. CHECKPOINT — facility approves before any equipment change.
5. STOP — end of shift/tariff period, or target met.
6. TEST — simulate against recent load + tariff data first.

## Output
Recommended actions with projected savings; period summary.

## Safety / stoppage
Throughput and comfort/safety constraints always override cost. No autonomous changes.
```

### Skill 8 — Digital-Twin Design Loop

```markdown
---
name: digital-twin-design-loop
description: >-
  Iterates simulated designs until one meets all specs, within a compute budget.
  Use when the user says "optimize this design", "run simulations until it meets
  spec", "explore design parameters", or "use the digital twin to find a design".
---

# Digital-Twin Design Loop

## Goal
Find a design that meets all specifications by iterating simulations.

## Inputs to gather
- Design parameters + ranges; the digital-twin / simulation model
- Spec acceptance criteria; objective (e.g., min weight at required strength)
- Simulation compute/iteration budget; reviewing engineer

## Loop (five precepts)
1. GOAL — confirm the spec acceptance criteria and the objective.
2. ASSESS — run a simulation, score the candidate vs. specs + objective.
3. ACT — propose the next parameter set (Bayesian/evolutionary search); iterate.
4. CHECKPOINT — engineer reviews candidate designs before adoption.
5. STOP — a design meets all specs, or the compute/iteration budget is spent.
6. TEST — validate the model on a known/benchmark design first.

## Output
Best candidate design(s) with spec compliance; search summary.

## Safety / stoppage
Hard compute/iteration budget. No design is "adopted" without engineer review.
```

---

## 9. Common Loop Patterns (a workflow vocabulary)

Loop engineering is workflow-centric, so it helps to recognize recurring loop shapes:

- **Monitor → Compare → Act → Notify:** poll a source, compare against the current best, take action if better, notify the human. *(The hotel example.)*
- **Generate → Evaluate → Refine:** produce a draft, score it against a rubric, improve it, repeat until the quality bar is met. *(Writing, code, design.)*
- **Plan → Execute → Check → Re-plan:** the AI plans steps, executes one, checks the result, and adjusts the plan on the next pass.
- **Poll-until-state:** repeatedly check a long-running process until it reaches a target state, then proceed or alert.
- **Accumulate-until-target:** keep gathering items (findings, candidates, data points) until you have enough or a deadline hits.

Each of these still needs the five precepts wrapped around it — especially a hard stopping rule.

---

## 10. Risks and Pitfalls

| Risk | What happens | Mitigation |
|---|---|---|
| **Runaway cost** | Loop iterates far too often, consuming expensive compute | Set cadence, max iterations, time window, and a budget cap |
| **Endless cycle** | Loop never reaches a stop condition (the "dog chasing its tail") | Define explicit, reachable stoppage rules; add a hard max-iteration backstop |
| **Irreversible mistakes** | AI takes an action you'd never approve and can't undo | Human checkpoint / require permission before destructive actions |
| **Hallucination mid-loop** | AI invents facts or actions during an iteration | Checkpoints, verification steps, and logging each iteration |
| **Instruction drift / vagueness** | AI interprets the loop differently than intended | Have the AI echo the goal; test before launch; be specific |
| **Silent failure** | Loop fails quietly and you assume it's working | Require status notifications; check in periodically |

A guiding caution: *"With a loop, you are potentially starting a flywheel that might cycle endlessly and spin out of orbit."* Good loops require attention to detail.

---

## 11. Best-Practice Checklist

Before you launch a loop, confirm you have:

- [ ] **A clear, written goal** — and the AI has echoed its understanding back to you.
- [ ] **A cadence/frequency** — how often the loop runs (e.g., hourly), not "as fast as possible."
- [ ] **An assessment rule** — how the AI decides to continue vs. stop each iteration.
- [ ] **At least one stop condition** — goal met, time window, max iterations, or budget cap (ideally several, OR'd together).
- [ ] **A human checkpoint** — notifications, and *permission gates* before any irreversible/high-stakes action.
- [ ] **Reversibility where possible** — e.g., "make a reservation that *can be canceled*."
- [ ] **Quality constraints** — guardrails that prevent "technically better" but unacceptable outcomes (e.g., minimum rating).
- [ ] **A test run** — a few iterations inspected and refined before full launch.
- [ ] **Logging/visibility** — a record of what the loop did each cycle.
- [ ] **A decision about your position** — in-the-loop (approve each step) vs. out-of-the-loop (notify only)?

---

## 12. A Reusable Loop-Prompt Template

```text
GOAL:
  <one clear sentence describing the desired end state>

TASKS PER ITERATION (the workflow):
  1. <step>
  2. <step>
  3. <step>

CADENCE:
  Run every <interval> (e.g., hourly).

ASSESSMENT:
  On each iteration, continue only if <condition>; otherwise <action>.

ACTIONS & PERMISSIONS:
  You MAY <reversible actions> on your own.
  You MUST ask me before <irreversible / high-stakes actions>.

CONSTRAINTS / QUALITY GUARDRAILS:
  <e.g., rating >= 4 stars; price <= $X; reputable sources only>

STOPPAGE RULES (any of these ends the loop):
  - Goal attained
  - <time window> elapsed (e.g., 48 hours)
  - <max iterations> reached
  - <budget / resource cap> exceeded

NOTIFICATIONS:
  Message me whenever <event>, and send a final summary when the loop ends.

TEST FIRST:
  Before running for real, do a short dry run of <N> iterations,
  show me what would happen, and wait for my confirmation.
```

---

## 13. Key Takeaways

- **Loop engineering = workflow-centric AI.** You design an iterative cycle, not a single prompt.
- **It's primarily for agentic AI** (agents that can act), but applies to conventional generative AI too.
- **The five precepts — Goal, Assessment, Checkpoint, Stoppage, Test — are the backbone.** Skip one and the loop can misbehave.
- **Always make the loop stoppable.** Bound it by time, iterations, budget, and goal.
- **Insert human checkpoints before irreversible actions.** Reversibility and permission gates prevent the worst outcomes.
- **Test before you let it loose.** A cheap dry run beats an expensive runaway.

> *"A circular loop can be extremely useful… but it won't get very far without the right kind of setup. Make sure the AI is fruitfully running in a circle, and you'll be okay."*

---

## Source

- Lance Eliot, *"Loop Engineering Is Fully Making The Rounds For Boosting Generative AI And Agentic AI,"* **Forbes**, June 17, 2026. The definition, five precepts, and hotel-booking examples in this guide are drawn from that article; the loop-pattern vocabulary, comparison tables, and reusable template are added context for practical application.

*Guide compiled June 24, 2026.*
