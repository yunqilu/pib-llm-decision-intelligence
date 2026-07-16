# Day 3 Clarification Memo — What I Understand + What I Need to Clarify

**To:** Leo
**From:** Yunqi Lu — Decision Intelligence & Optimization Research Associate (Role 16, Applied/Optimization)
**Date:** 2026-07-13
**Tracker item:** Week 1 — "Day 3 clarification memo — submit to Leo" (DELIVERS)
**Sources read:** Glossary (tracker 📚 tab) · onboarding `00-overview.md` (DFI) · `pib-llm-backend` README + `docs/ARCHITECTURE.md` (NOI) · Role 16 brief · NOI mock interfaces (`services/blackbox_mock.py`, `services/telemetry_chunker.py`, `scripts/seed_pgvector.py`, schema-diff + golden-questions specs) · Week 1 joint-task doc with James
**Scope note:** All of the above stayed within the Role 16 boundary — open NOI repos and standard mock interfaces only; no access to or reasoning about Blackbox / BioCore internals.

---

## Part 1 — What I Understand

### 1.1 The platform and where I sit

PhysaFlow is split into two ecosystems with a strictly one-way data flow, **DFI → NOI**:

| | DFI (`leomutto-org`) | NOI (`leomutto-org2`) — my workspace |
|---|---|---|
| Role | Data-feed side: telemetry producers (simulator, world_wide_ingestor, bip-project), ingest/validation (MASS Enterprise v1.1 contracts), Blackbox invocation, storage in `pilotdb` | Conversational/decision-intelligence layer: multi-tenant RAG (FastAPI + pgvector + SBERT + Bedrock), reads `pilotdb` read-only, embeds chunks into its own `llmdb` |
| My access | Background context only (onboarding docs) | Full — this is where Role 16 operates |

Everything runs in **Shadow Mode** (read-only; recommendations, never control actions) across all 19 active projects. The **Blackbox / BioCore** engine is strictly off-limits; the mock (`blackbox_mock.py`) is the sanctioned stand-in, and since card LLM-132 it matches the real EDD-Q1 response schema — so anything I prototype against the mock should transfer.

### 1.2 My mission (Role 16)

Formalize recovery of **stranded capacity** (the 30–40% of installed capacity that is powered but unproductive due to siloed power/cooling/thermal management) as a **constrained multi-objective optimization problem** (greedy → LP/MILP baselines), while James formalizes the same problem as an MDP solved with RL. One shared problem, one shared environment and benchmark harness, two solution methods benchmarked head-to-head — that's the Week 4 payoff. Deliverables are research memos, baselines, benchmarks, and honestly-measured recovery percentages.

### 1.3 What the mock interfaces actually expose (inventory result)

From the full inventory (details in `glossary-and-mock-telemetry-inventory.md`):

**Numeric telemetry that flows through the pipeline** (`telemetry_chunker.py`, 5min/1h/4h/24h windows): `cpu_avg`, `it_load_raw/optimized`, `pue_raw/optimized` (≥ 1.0 enforced; 0.0 = "not reported" sentinel, per NOI-213), `impact_kwh_24h`, `co2_kg`, `cost_usd`, `confidence`, `risk`.

**Decision variables observed as mock Blackbox action targets** (4 tenants: `alcf`, `acme`, `tenant-001`, `tenant-002`): `it_load_kw` (reduce), `chiller_setpoint_c` (raise), `pump_b12_duty_pct` (reduce), `setpoint_temperature_c` (raise), `production_load_pct` (shift). Each action carries `current / suggested / delta / units`, `expected_impact.pue_delta`, reason codes, and a `constraints` list.

**Constraint coverage by physical category:**

| Category    | Numeric state in mock?                                        | Decision variable in mock? | Machine-readable constraint?         |
| ----------- | ------------------------------------------------------------- | -------------------------- | ------------------------------------ |
| ⚡ Power     | ✅ yes (best covered)                                          | ✅ `it_load_kw`             | ⚠️ free text only                    |
| ❄️ Cooling  | ❌ no cooling-load field                                       | ✅ setpoints, pump duty     | ⚠️ free text (e.g. "N+1 redundancy") |
| 🌡️ Thermal | ❌ temps only inside constraint strings ("28°C inlet redline") | ✅ `setpoint_temperature_c` | ⚠️ free text only                    |
| 💧 Water    | ❌ entirely absent (verified by exhaustive grep)               | ❌ none                     | ❌ n/a                                |

The DFI side does have richer physics (SPECpower power curve, ASHRAE thermal model, `water_liters`, inlet/return temps) per the onboarding docs — but none of it is exposed through NOI's mock interfaces, so it is background knowledge, not something I can prototype against.

### 1.4 Joint task with James — status

The Week 1 joint task produced a shared-scope proposal (one problem, two tracks) and a repo layout, now scaffolded as `pib-llm-decision-intelligence`: shared `env/` (simulator + mock-telemetry adapter + scenarios), shared `benchmark_harness.py`, separate `optimization/` (me) and `rl/` (James) tracks, ADR-style decision records, and weekly memos. `problem_spec.md` and `data_schema.md` exist but are **DRAFT pending joint sign-off with James** — the draft state/action fields come directly from the inventory above. Ownership rule: `env/`, `shared/`, and `problem_spec.md` change only by mutual PR review.

---

## Part 2 — What I Need to Clarify

Each question comes with my proposed default so a one-word answer unblocks me.

**Q1 — Naming: which expansions are canonical?** The Glossary defines DFI = "Distributed Flow Intelligence" and NOI = "Network Organism Intelligence", but the onboarding overview says DFI = "Data Feed Infrastructure" and the NOI backend README says NOI = "Net Operating Income optimization". I assume Glossary = external/product naming, code docs = internal/legacy framing, and I'll use the Glossary versions in anything Leo-facing. *Confirm?*

**Q2 — Is water in scope for the summer formulation?** Water exists on the DFI side (`water_liters`, `water_use_proxy`) but has zero representation in NOI's mock interfaces. My default: **exclude water from the v0 formulation**, note it as a documented limitation, and design the schema so a water term can be added later without rework. *OK, or should we request water be added to the mock?*

**Q3 — Constraints are free-text strings, not structured bounds.** E.g. `"Do not exceed thermal redline (28°C inlet)"`. For the Week 2 formulation I plan to **hand-encode the parseable numeric bounds into `problem_spec.md`**, explicitly tagging each as *sourced* (parsed from mock text) vs *approximated* (assumed). Is a structured constraints schema coming from the real engine at some point, or is hand-encoding the expected path?

**Q4 — Thermal state: proxy acceptable?** Inlet/return temperatures exist in DFI's data but not in NOI's mock; thermal appears only narratively. Default: **derive thermal headroom as a proxy from PUE/power**, following the DFI physics convention, and mark it approximated. Alternative: request real thermal fields be surfaced into the mock/real NOI path. *Which do you prefer?*

**Q5 — Same question for cooling load:** no numeric cooling-power field exists in the NOI mock. Default proxy: `facility_power − it_load` (implied by PUE). *Acceptable for v0?*

**Q6 — Repo home:** the shared workstream is currently scaffolded at `github.com/yunqilu/pib-llm-decision-intelligence` (personal account). Should it move into `leomutto-org2` so it lives with the other NOI repos and PR review can follow the normal workflow?

**Q7 — Tracker hygiene:** several Week 1 rows (Glossary inventory, joint task, this memo, constraint mapping) show "Pending / Behind" but are now done or delivered — I'll flip them to "Submitted for Review" today unless you'd rather update the tracker yourself.

---

## Part 3 — Where this leaves Week 2

With Q2–Q5 answered, I can immediately draft the constrained multi-objective formulation: decision variables = the five mock action targets; state = the eight numeric telemetry fields; objective = stranded-capacity recovery % traded against quality/latency/robustness; constraints = hand-encoded bounds (per Q3). The toy-instance prototype can run entirely against `blackbox_mock.py` + the `ACME` seed scenario (the only tenant safe to seed, enforced in `seed_pgvector.py`), keeping everything inside the Role 16 boundary.

*No Blackbox/BioCore internals were accessed or reasoned about in preparing this memo.*
