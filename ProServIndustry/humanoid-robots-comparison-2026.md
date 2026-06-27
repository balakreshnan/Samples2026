# Humanoid Robot Makers & Products — Comparative Analysis (2025–2026)

*Prepared for: Balamurugan Balakreshnan · Date: 2026-06-27*

A consolidated, honestly-scored survey of the major humanoid robot makers — their products, designs, cost, technical architecture, AI stack, best-fit use cases, and a 1–100 real-world functionality score.

> **How to read this report.** The scoring axis is **demonstrated, autonomous, real-world utility *today*** — not marketing claims, not teleoperated demos, not mobility for its own sake. A robot that quietly does one narrow job autonomously at production cadence scores higher than one with spectacular demos that turn out to be remote-controlled. Confidence levels are flagged because a large fraction of 2026 "spec sheet" content on the web is AI-generated with fabricated numbers — primary/company sources were prioritized.

---

## 1. Executive Summary

- **The real divide is "demoed" vs. "autonomously deployed."** Only a handful of robots have independently documented, revenue-generating, largely-autonomous real-world work: **Agility Digit**, **Figure 02/03**, **Boston Dynamics Atlas** (pilot), and **UBTECH Walker S2**.
- **Commercial deployment leader: Agility Digit** — 100,000+ totes moved at GXO, ~98% task success over 18 months, plus a paid Toyota Canada contract. Narrow scope, but genuinely autonomous.
- **Dexterous-work leader: Figure 02 → 03** — 11-month BMW Spartanburg deployment, 90,000+ parts loaded, running its in-house **Helix VLA model fully onboard** (not teleoperated).
- **Athleticism/mobility leader: Boston Dynamics Atlas (electric)** — best-in-class movement and recovery; autonomous parts-sequencing pilots at Hyundai; still pre-commercial.
- **Volume leader: Chinese makers** — AgiBot (~5,100 units in 2025) and Unitree (~4,200) ship in the *thousands*; UBTECH Walker S2 has 500+ industrial units deployed. US makers (Tesla, Figure, Agility) still ship in the *hundreds*.
- **Biggest hype-vs-reality gaps:** Tesla Optimus (units deployed mainly for *data collection*, historic demos teleoperated), 1X NEO (a real consumer product but **teleoperation-dependent** at launch), and XPENG IRON (pre-production demo platform).
- **Market size:** Goldman Sachs projects a **$38B** total addressable market by 2035 (~1.4M units); other analysts range wildly to multi-trillion-dollar 2050 scenarios — treat long-range numbers as speculative.

---

## 2. Maker & Product Roster (at a glance)

| Maker | Flagship | Origin | Primary target | Stage |
|---|---|---|---|---|
| Tesla | Optimus (Gen 2 / Gen 3) | 🇺🇸 | Manufacturing → consumer | Data-collection pilots |
| Figure AI | Figure 02 / 03 (Helix) | 🇺🇸 | Manufacturing, logistics | Production-validated pilot |
| Boston Dynamics | Atlas (all-electric) | 🇺🇸 (Hyundai-owned) | Manufacturing | Pre-commercial pilot |
| Agility Robotics | Digit | 🇺🇸 | Warehouse / logistics | **Commercially deployed** |
| 1X Technologies | NEO / EVE | 🇳🇴/🇺🇸 (OpenAI-backed) | **Home / consumer** | Shipping (teleop-assisted) |
| Unitree | G1 / H1 / R1 | 🇨🇳 | Research, low-cost | Shipping (research/demo) |
| Apptronik | Apollo | 🇺🇸 | Logistics, manufacturing | Named pilots |
| Sanctuary AI | Phoenix | 🇨🇦 | General-purpose work | Pilot (teleop-centric) |
| Fourier Intelligence | GR-1 / GR-2 / GR-3 | 🇨🇳 | Healthcare / research | Research/healthcare pilot |
| UBTECH | Walker S2 | 🇨🇳 | Industrial / auto | **Volume industrial deploy** |
| XPENG | IRON (2nd Gen) | 🇨🇳 | Consumer/showcase | Pre-mass-production |
| AgiBot (Zhiyuan) | Genie G2 | 🇨🇳 | Manufacturing | High-volume shipping |
| Honda | ASIMO (retired 2018) | 🇯🇵 | — | Legacy / discontinued |

---

## 3. Specs & Design Comparison

> ⚠️ **Confidence note.** Figures marked *(est.)* are third-party estimates, not company-confirmed. Boston Dynamics has published **no official spec sheet** for the electric Atlas; Tesla has not officially released full Gen 3 specs.

| Robot | Height | Weight | DoF (total) | Hand DoF | Payload | Battery / runtime | Actuation | Sensors |
|---|---|---|---|---|---|---|---|---|
| **Tesla Optimus Gen 2** | 173 cm | ~57–73 kg *(disputed)* | ~40 | 11/hand | ~20 kg | 2.3 kWh | Electric | Vision-only, 8 cameras, **no LiDAR** |
| **Figure 02** | 168 cm | 70 kg | 41 | 16/hand | ~20–25 kg | 2.25 kWh / ~5 hr | Electric | 6× RGB cameras |
| **Figure 03** | ~168 cm | ~70 kg | 41+ | 16/hand | ~20 kg | onboard | Electric | + palm cameras, fingertip tactile (~3 g) |
| **BD Atlas (electric)** | ~150 cm | ~89 kg *(est.)* | ~28 *(est.)* | — | ~25 kg *(est.)* | onboard | Electric (360° joints) | ToF, stereo/RGB-D, IMU |
| **Agility Digit** | 175 cm | ~63.5 kg | ~28 | low (clamp) | 16 kg (→~22.6) | ~4 hr | Electric | **LiDAR** + 4× RealSense, IMU, force |
| **1X NEO** | ~165 cm | ~30 kg | 44 | 22/hand | ~25 kg lift | 842 Wh / ~4 hr | **Tendon-driven, soft body** | Cameras (no LiDAR) |
| **Unitree G1** | 132 cm (folds 69) | ~35 kg | 23 (EDU 23–43) | 7 (optional) | — | swappable | Electric | Depth + LiDAR options |
| **Apptronik Apollo** | 173 cm | ~72 kg | ~71 *(est., low conf.)* | PSYONIC Ability Hand | ~25 kg | 4 hr hot-swap | Electric (modular legs/wheels) | Vision + force |
| **Sanctuary Phoenix** | 170 cm | ~70 kg | 20–44 *(disputed)* | ~20 | ~25 kg | onboard | **Hydraulic/electric (disputed)** | Vision |
| **Fourier GR-2** | 175 cm | — | 53 *(low conf.)* | 12/hand | — | swappable | Electric (380 N·m peak) | Vision |
| **UBTECH Walker S2** | 176 cm | 73 kg | 52 | 11/hand (7 active) | ~15 kg | **autonomous swap, ~24h** | Electric | Binocular stereo |
| **XPENG IRON** | ~178 cm | — | 82 (+22 hands) | 22 | — | solid-state | Electric "bionic muscle" | Vision, 3× Turing chips |

### Design philosophies — what actually makes each different
- **Tesla Optimus** → *Mass-manufacturing first.* Designed from the ground up for automotive-scale production (lines targeting up to 1M units/yr), full vertical integration, vision-only sensing (no LiDAR) to mirror the FSD approach.
- **Figure 02/03** → *Vertically-integrated dexterity.* In-house Helix VLA brain, added fingertip tactile + palm cameras on 03; optimized for a single, well-defined factory task done reliably.
- **Boston Dynamics Atlas** → *Super-human efficiency over biomimicry.* Electric actuators give 360° rotation at hip/waist/neck (beyond human range); 3D-printed titanium; best-in-class dynamic recovery.
- **Agility Digit** → *Purpose-built warehouse reliability.* No head; reverse-jointed "bird-leg" (digitigrade) legs store/return energy per step, lower the center of gravity, and enable autonomous fall recovery. Repeatability over dexterity.
- **1X NEO** → *Safe-in-the-home.* Tendon-driven actuation + soft 3D-lattice polymer body, only ~22 dB operating noise — explicitly engineered to be safe around people in a house.
- **Unitree** → *Low-cost democratization.* Strong dynamic mobility (running, acrobatics) at a fraction of competitors' price; research/education positioning.
- **Apptronik Apollo** → *Approachable, modular, safety-rated.* NASA Valkyrie lineage; runs on legs *or* a wheeled base; uses PSYONIC prosthetic-derived hands.
- **UBTECH Walker S2** → *Uptime.* World's first industrial humanoid with **autonomous battery swap** — walks to a power station and swaps its own depleted pack in ~3 min for ~24h operation.
- **XPENG IRON** → *Extreme anthropomorphism.* Humanoid spine, bionic muscles, full flexible skin covering — a showcase of human-like form.

---

## 4. Cost / Price Comparison

| Robot | Price (unit) | Notes |
|---|---|---|
| **Unitree R1** | **~$5,900** | Cheapest full humanoid on the market |
| **Unitree G1** | from **$13,500** | EDU variants higher |
| **1X NEO** | **$20,000** or $499/mo | Consumer; OpenAI-backed |
| **Tesla Optimus** | target **<$20K** (consumer), ~$30K initial | Aspirational, not an actual sale price |
| **Apptronik Apollo** | **<$50,000** at scale *(est.)* | $520M raised @ $5B valuation |
| **UBTECH Walker S2** | **$68,000–120,000** *(est.)* | No official public pricing |
| **Figure 02** | **~$100,000–130,000** *(est.)* | Enterprise/B2B only |
| **Fourier GR-2** | **$150,000+** *(est.)* | Enterprise-only, no NA channel |
| **Agility Digit** | **~$250,000** or RaaS ~$30/hr | Robots-as-a-Service available |
| **BD Atlas** | **No public price** | Not purchasable; 2026 output allocated to Hyundai |
| **Sanctuary Phoenix** | Not public | Pilot only |
| **XPENG IRON** | Not public | Mass production targeted end-2026 |

> **Cost trend:** Per-unit manufacturing cost has dropped from ~$50K–$250K to ~$30K–$150K — roughly a **40% decline**, far faster than the 15–20% analysts expected, driven largely by Chinese supply chains and Unitree's price war.

---

## 5. Technical Architecture & AI Enablement

The industry has converged on **Vision-Language-Action (VLA) models + sim-to-real transfer** — learn behavior in a physics simulator, then transfer to hardware. NVIDIA's **Jetson Thor** compute and **Isaac Sim/Lab + GR00T** stack are the shared infrastructure under many of these robots.

| Robot | AI-enabled? | AI model / "brain" | Compute hardware | Training approach |
|---|---|---|---|---|
| **Figure 02/03** | ✅ Yes (leading) | **Helix** VLA in-house, tri-system (S0 1 kHz balance / S1 200 Hz motor / S2 7B-param VLM planner). Dropped OpenAI for vertical integration. | NVIDIA RTX-class GPUs onboard | Onboard, no cloud; imitation + sim-to-real |
| **Tesla Optimus** | ✅ Yes | FSD-derived neural nets + (Gen 3) VLA + **Grok** voice | Custom **AI5** chip | End-to-end vision learning |
| **BD Atlas** | ✅ Yes | Toyota Research Institute **Large Behavior Models**; NVIDIA GR00T; Google DeepMind RL pipeline | **NVIDIA Jetson Thor** (~800 TFLOPS) | Isaac Lab sim-to-real |
| **Agility Digit** | ✅ Yes | LSTM whole-body control (**<1M params**); Agility Arc cloud fleet mgmt | Onboard | Isaac Sim, zero-shot transfer |
| **1X NEO** | ✅ Yes | **Redwood AI** | **NVIDIA Jetson Thor** (up to 2,070 FP4 TFLOPS) | Imitation; **+ human teleoperation** |
| **Apptronik Apollo** | ✅ Yes | **Google DeepMind Gemini Robotics** (natural-language, multi-step planning) | Onboard | MANUS data-glove teleop → imitation learning |
| **Sanctuary Phoenix** | ✅ Yes | **Carbon** — symbolic reasoning + LLMs + RL ("cognition-first") | Onboard | Teleoperation demonstrations |
| **UBTECH Walker S2** | ✅ Yes | **BrainNet** reasoning + Co-Agent + swarm intelligence | Onboard | Binocular stereo vision |
| **Unitree G1/H1** | ◑ Emerging | **UnifoLM** world model (early) | Onboard | Primarily mobility/RL |
| **XPENG IRON** | ✅ Yes | Onboard Turing AI (3,000 TOPS) | **3× Turing chips** | Pre-production |
| **Fourier GR** | ◑ Partial | Research-grade | Onboard | Research/healthcare |

**Model/partner map (the AI alliances):**
- Figure → **in-house Helix** (formerly OpenAI)
- Apptronik & Atlas → **Google DeepMind** (Gemini Robotics / RT-2 lineage)
- Tesla → **own FSD nets + AI5 + Grok**
- 1X → **Redwood AI** on Jetson Thor
- Sanctuary → **Carbon**
- Many → **NVIDIA GR00T + Jetson Thor + Isaac Sim**

---

## 6. Best-Fit Use Cases (Comparative)

| Use case | Best-suited robots | Why |
|---|---|---|
| **Warehouse / logistics** | **Agility Digit** (leader), Apptronik Apollo, UBTECH Walker S2 | Digit is purpose-built, OSHA-validated, autonomous, proven at GXO / Amazon / Toyota |
| **Auto / manufacturing assembly** | **Figure 03**, **Atlas**, **UBTECH Walker S2**, Apollo | Dexterous pick-and-place (Figure @ BMW); Atlas mobility; Walker S2 volume in China |
| **Home / consumer** | **1X NEO** (only true consumer product), Tesla Optimus (future) | NEO is purpose-designed soft/safe for homes — but autonomy still immature (teleop-assisted) |
| **Research / education / low-cost** | **Unitree G1 / H1 / R1** | $5.9K–$13.5K pricing, excellent mobility, open dev access |
| **Healthcare / elder care** | **Fourier GR-3**, (future) NEO, Apollo | Fourier's rehab heritage (2,000+ medical institutions) |
| **High-uptime / 24h lines** | **UBTECH Walker S2** | Autonomous battery swap → ~24h operation |
| **Athletic / unstructured mobility** | **Boston Dynamics Atlas** | Best dynamic balance, recovery, 360° joints |

**Key differentiators at a glance:**
- **Cheapest:** Unitree R1 (~$5,900) / G1 (~$13.5K)
- **Most autonomous in production:** Agility Digit; Figure 02/03
- **Best mobility/athleticism:** Boston Dynamics Atlas
- **Best AI integration:** Figure (Helix), Apptronik (Gemini Robotics), NVIDIA GR00T partners
- **Highest uptime:** UBTECH Walker S2 (autonomous battery swap)
- **Most shipped (volume):** AgiBot (~5,100/2025), Unitree (~4,200), UBTECH (~1,000+)

---

## 7. Deployment Status — Demoed vs. Actually Deployed

| Robot | Documented real-world deployment | Autonomous or teleoperated? |
|---|---|---|
| **Agility Digit** | 100,000+ totes @ GXO; Toyota Canada (7 robots, RAV4 plant); Amazon pilots | ✅ **Autonomous** (narrow scope) |
| **Figure 02** | 11-mo @ BMW Spartanburg; 90,000+ parts; 1,250+ hrs; contributed to 30,000+ X3 vehicles | ✅ **Autonomous** (onboard Helix) |
| **UBTECH Walker S2** | 500+ units @ BYD, Geely, Audi FAW, Foxconn, SF Express, Airbus; ~$112M order book | ◑ Autonomous (per-task unaudited) |
| **BD Atlas** | Autonomous parts-sequencing pilots @ Hyundai (CES 2026) | ✅ Autonomous (pilot, pre-production) |
| **Apptronik Apollo** | Mercedes-Benz (Berlin, Kecskemét), GXO, Jabil — intra-logistics | ◑ Teleop-assisted learning |
| **AgiBot** | 5,100+ units shipped 2025; Genie G2 on assembly lines | ◑ Vendor-reported autonomy |
| **1X NEO** | Shipping to homes | ⚠️ **Teleoperation-dependent** at launch |
| **Sanctuary Phoenix** | Magna International pilots | ⚠️ Teleop-centric |
| **Tesla Optimus** | ~100s units @ Fremont / Giga Texas | ⚠️ **Data collection, not productive work**; historic demos teleoperated |
| **Unitree / Fourier / XPENG** | Research, performance, demos | ◑ Mostly demo/research |

---

## 8. Honest Real-World Functionality Score (1–100)

**Scoring axis:** demonstrated, *autonomous*, real-world utility *today*. Documented autonomous production hours and independent field data are rewarded; demos, teleoperation, and forward-looking spec sheets are discounted.

| Rank | Robot | Score | Core evidence / honest caveat |
|---|---|---:|---|
| 1 | **Agility Digit** | **73** | 100k+ totes @ GXO, ~98% success/18 mo, OSHA-validated, paid Toyota contract — but deliberately narrow logistics only |
| 2 | **Figure 02** | **70** | 11-mo BMW, 90k parts, onboard Helix VLA, genuinely autonomous |
| 3 | **Figure 03** | **63** | Leading-edge Helix 02 + tactile, but limited field hours so far |
| 4 | **Boston Dynamics Atlas** | **58** | Best mobility + autonomous Hyundai pilots; still pre-commercial |
| 5 | **UBTECH Walker S2** | **54** | 500+ shipped, real factory QC/handling, battery-swap; China-centric, autonomy unaudited |
| 6 | **Apptronik Apollo** | **51** | Real Mercedes/GXO/Jabil pilots; teleop-assisted, DeepMind mostly lab |
| 7 | **AgiBot** | **48** | Huge volume + vendor-claimed 99.9% line success (**unverified**) |
| 8 | **Tesla Optimus (Gen 2)** | **40** | ~100s units but **data-collection, not productive**; historic demos teleoperated; Gen 3 unproven |
| 9 | **Galbot** | **40** | Wheeled retail pilots, China; limited independent data |
| 10 | **Unitree G1/H1** | **38** | Cheap + agile, but research/demo; limited useful-work autonomy |
| 11 | **Sanctuary Phoenix** | **37** | Strong cognition thesis; teleop-centric, marketing claims unverified |
| 12 | **Fourier GR-1/2/3** | **34** | Healthcare/research; weak spec sourcing |
| 13 | **XPENG IRON** | **32** | Anthropomorphic demo platform; pre-mass-production |
| 14 | **1X NEO** | **31** | Real consumer product **but teleoperation-dependent** at launch (autonomous-only basis) |

> **Why the "headline" robots score lower:** Optimus, NEO, and XPENG generate the most press but score lower on *demonstrated autonomous utility* precisely because consumer breadth currently forces reliance on teleoperation or data-collection framing. The robots that score highest **narrowed scope and instrumented autonomy** (Digit's structured logistics; Figure's single well-defined BMW task) rather than chasing general-purpose breadth.

---

## 9. Market Outlook

| Analyst | Projection |
|---|---|
| **Goldman Sachs** | $38B TAM by 2035, ~1.4M units (blue-sky up to $154B) |
| **UBS** | 2M workplace units by 2035 ($30–50B); → 300M units / $1.4–1.7T by 2050 |
| **Morgan Stanley** | ~$5T market by 2050 (>1B units) |
| **Citigroup** | 648M units / $7T by 2050 |
| **MarketsandMarkets** | $2.92B (2025) → $15.26B (2030), ~39% CAGR |

The near-term (2025–2030) numbers are reasonably grounded; 2050 projections vary by two orders of magnitude and should be treated as scenario-planning, not forecasts.

---

## 10. Caveats & Confidence

1. **Source contamination is the #1 risk.** Much 2026 "spec" content is AI-generated with fabricated numbers (one site falsely claimed 50,000 active Optimus units; another priced Atlas at "$320K–500K" with "Pet Friendly: Yes"). Figures here were anchored to company/primary sources where possible.
2. **Disputed specs to re-verify before quoting exact figures:** Optimus Gen 2 weight (57 vs 73 kg); Atlas DoF and pricing (BD has **no official spec sheet**); Sanctuary actuation (hydraulic vs electric) and DoF; Apollo total DoF; all Fourier and XPENG figures.
3. **"Shipped/announced ≠ autonomous."** Vendor success rates (UBTECH, AgiBot's 99.9%, Sanctuary's 98%) are **not independently audited.**
4. **Next-gen = forward-looking.** Tesla Optimus Gen 3, XPENG IRON mass production, and Figure 03 at-scale are promises, not shipped capability — they are not scored on those promises.
5. **Honda ASIMO** was retired in 2018 with no competitive general-purpose successor in the current race.

---

*Sources (high-trust first): figure.ai (BMW deployment), 1x.tech (NEO teleoperation disclosure), agilityrobotics.com, Boston Dynamics, goldmansachs.com (market), Interesting Engineering (Apptronik/Figure 03), CGTN (XPENG), chosun.com & PRNewswire-cited data (UBTECH), Tech Times (cross-vendor milestones, NVIDIA GR00T), unitree.com. Aggregator sites used only for cross-checking and flagged where uncorroborated. Confidence: MODERATE overall — strong for market leaders, weaker for Tesla Gen 3, Sanctuary, Fourier, and XPENG.*
