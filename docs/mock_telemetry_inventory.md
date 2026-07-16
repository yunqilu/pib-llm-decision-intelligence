# PhysaFlow NOI — Glossary & Mock Telemetry / Physical Constraints Inventory

**Prepared by:** Yunqi Lu — Decision Intelligence & Optimization Research Associate (Applied/Optimization)
**Tracker item:** Week 1 — "Read the Glossary; inventory the available mock telemetry and physical constraints (power, cooling, water, thermal)"
**Scope followed:** Role 16 boundary — research/modeling level only, on PhysaFlow's **open NOI repositories**, using **standard mock interfaces**. No access to, inspection of, or reasoning about the proprietary Blackbox / BioCore engine internals.
**Repos inspected:** `pib-llm-backend-main` (NOI backend, primary source), `physaflow-onboarding-main` (architecture context only, cited separately where DFI-side facts are used for background).

---

## 0. Methodology note — please read before using this doc

I could not find a file or document section literally titled **"Glossary"** anywhere in the repos I currently have access to (`pib-llm-backend-main`, `pilot-in-a-box-LLM-main`, `pib-llm-frontend-main`, `physaflow-onboarding-main` — checked by filename and by full-text grep). Two possibilities:

1. It's a resource Leo hasn't shared yet (Notion, Google Doc, or a repo I don't have access to).
2. It's meant to be assembled by us from the scattered term definitions already in the codebase/docs (`AGENTS.md`, `docs/architecture/`, `docs/specs/`).

**I'm treating this document as a working substitute for (1) or (2)** — Section 1 below is a glossary I compiled directly from source (every term cites where it's defined). **This is a flag to raise with Leo**, not a resolved item — see the open question in §4.

Everything below is sourced from actual code/docs I read, with file:line references so it can be verified. Where I note something is **absent**, I searched for it (grep) and confirmed it isn't there — that's a finding, not an oversight.

---

## 1. Glossary (compiled from source)

| Term | Definition | Source |
|---|---|---|
| **PhysaFlow / "Pilot-in-a-Box"** | The overall platform; helps data-center operators recover stranded AI capacity without new hardware, in shadow/read-only mode | `physaflow-onboarding-main/00-overview.md`; Role 16 brief |
| **NOI** | "Network Organism Intelligence" (product framing) — the conversational AI / decision-intelligence layer. Internally the code still says "Copilot" — legacy naming, **never used externally** | `pib-llm-backend-main/AGENTS.md:14-15`; Role 16 brief |
| **DFI** | Data Feed Infrastructure — the other ecosystem (`leomutto-org`): producers, ingest, dashboard. Out of this document's direct repo access, referenced only via onboarding docs | `physaflow-onboarding-main/00-overview.md` |
| **Blackbox / Core Vault / BioCore** | The proprietary optimization engine. **Strictly off-limits** — this doc never inspects its internals, only the mock/consumer-side contract | Role 16 brief, "Scope & Boundaries" |
| **Champion / Challenger** | Two parallel model versions the Blackbox runs for comparison (`runtime_version` field, e.g. `"0.4.1"` = Champion) | `services/blackbox_mock.py:12,103` |
| **EDD-Q1** | The canonical real Blackbox response schema (name of the schema spec doc). Migration target after Sprint 13 (card LLM-132) | `docs/specs/blackbox_mock_vs_real_schema_diff.md` |
| **Shadow / read-only mode** | The platform observes and recommends but never directly controls facility equipment | `physaflow-onboarding-main/00-overview.md`; `guardrails.shadow_read_only_enforced` field, `blackbox_mock.py:35` |
| **PUE** | Power Usage Effectiveness = facility power / IT power. Physically always **≥ 1.0**; a `0.0` reading in the data is a "not reported" placeholder, never a real value | `services/telemetry_chunker.py:46-52,91-96` (explicit code comment + bug history, NOI-213) |
| **IT load** | Power drawn by compute equipment itself (excludes cooling/overhead), in kW | `services/blackbox_mock.py` (`it_load_kw` action target) |
| **stranded capacity** | Installed capacity that's physically present but unusable due to poorly distributed power/cooling/thermal constraints (the central problem this internship targets) | Role 16 brief |
| **tenant** | A customer/site (e.g. `alcf`, `acme`). Regex-validated lowercase id; one blocked tenant exists (`alibaba`) | `services/tenant.py:34-45` |
| **chunk** | A windowed text summary of telemetry+recommendations, embedded and stored for retrieval | `services/telemetry_chunker.py` |
| **RAG** | Retrieval-Augmented Generation — retrieve relevant chunks, feed them to the LLM as grounding context | general NOI architecture |
| **reason code** | A short uppercase-underscore tag explaining *why* a recommendation was made (e.g. `HIGH_PUE`, `THERMAL_HEADROOM_AVAILABLE`) | `services/blackbox_mock.py:73`; `services/guardrails.py:108` |
| **golden question(s)** | A hand-curated evaluation dataset of Q&A pairs used to test RAG/grounding quality against known-correct schema paths | `docs/specs/golden_questions_v2.md` |
| **artificial / demo tenant** | A synthetic tenant (currently only `ACME`) safe to seed with fake data; seeding a real tenant is blocked in code to avoid contaminating its RAG | `scripts/seed_pgvector.py:18-41` |

---

## 2. Mock telemetry — what exists, where it lives, what it contains

The NOI backend does **not** generate raw physical telemetry itself (that's the DFI producers' job, out of this repo's scope). What NOI's own repo *does* ship, and what the Role 16 brief calls "the standard mock interfaces," are three layers:

### 2.1 Mock Blackbox recommendations — `services/blackbox_mock.py`

This is the primary mock interface: a hand-authored `_MOCK_DATA` dict keyed by tenant, shaped to match the **real EDD-Q1 schema** (migrated in card LLM-132, so mock and real have the same shape — good for us, it means anything prototyped against the mock transfers to real data later).

**4 mock tenants available today:**

| Tenant | Domain framing | Example action (decision variable) | Constraints attached (natural language) |
|---|---|---|---|
| `alcf` | HPC supercomputing (Argonne) | reduce `it_load_kw`: 280.5 → 250.0 kW | "Maintain HPC job queue SLA"; "Do not exceed thermal redline (28°C inlet)" |
| `acme` | Commercial datacenter | raise `chiller_setpoint_c`: 7.0 → 9.5 °C | "Maintain N+1 cooling redundancy"; "Operational window: 02:00–06:00 local time" |
| `tenant-001` | Legacy/compat (office building profile) | 3 recs: reduce `it_load_kw`, raise `setpoint_temperature_c`, reduce `pump_b12_duty_pct` | "No reducir por debajo de 18°C en zona de oficinas"; "Mantener presion minima en sistema HVAC"; pump maintenance window |
| `tenant-002` | Legacy/compat (production facility) | shift `production_load_pct`: 95.0 → 80.0% | "No afectar linea de produccion critica" |

**Full field schema per recommendation** (this is the actual "decision variable" shape available to prototype against):

```
action: { target, current, suggested, delta, units, type }
expected_impact: { pue_delta }
explainability: { reason_codes[], narrative }
constraints: [str]   # human-readable, not machine-parseable bounds
```

Response-level summary (per tenant, aggregated):
```
summary: {
  confidence, risk_score,
  estimated_impact_24h: { energy_kwh, energy_cost_usd, pue_delta, co2e_kg },
  guardrails: { shadow_read_only_enforced }
}
```
Source: `services/blackbox_mock.py:1-330`.

**Important limitation for optimization formalization:** `constraints` is a **list of free-text strings** (e.g. `"Do not exceed thermal redline (28°C inlet)"`), not structured `{variable, bound, direction}` triples. If Week 2's formulation needs machine-readable bounds, this is a gap to flag — the mock doesn't currently expose constraints as numeric objects, only as narrative.

### 2.2 Canonical schema reference — `docs/specs/blackbox_mock_vs_real_schema_diff.md` + `docs/specs/golden_questions_v2.md`

These two docs give the authoritative field list (I cross-checked them against the code and they match exactly):

```
response.recommendations[0].action.target / .current / .suggested / .delta / .units / .type
response.recommendations[0].expected_impact.pue_delta
response.recommendations[0].explainability.reason_codes / .narrative
response.summary.confidence / .risk_score
response.summary.estimated_impact_24h.energy_kwh / .energy_cost_usd / .pue_delta / .co2e_kg
response.summary.guardrails.shadow_read_only_enforced
```

Also documents **fields expected from the real engine but not yet in mock data**: `kpi_deltas`, `kpi_details`, `metric_status`, `audit`, `source_classification`, artifact URIs — worth knowing these exist in the target schema even though the mock doesn't fabricate them yet.

### 2.3 The numeric fields that actually flow through the pipeline — `services/telemetry_chunker.py`

This is the **live, structural mock telemetry vector** — the fixed set of numeric fields every windowed chunk carries (5min / 1h / 4h / 24h windows):

```python
_NUMERIC_FIELDS = (
    "cpu_avg", "it_load_raw", "pue_raw", "it_load_optimized", "pue_optimized",
    "impact_kwh_24h", "co2_kg", "cost_usd", "confidence", "risk",
)
```
Source: `services/telemetry_chunker.py:33-44`.

Notable code-level honesty rule baked in here: PUE values `< 1.0` are **excluded from averaging** and rendered as `"not available"`, because a physical PUE can never be below 1.0 — a `0.0` reading is a placeholder, not data (`_fmt_pue`, line 91-96; bug history NOI-213). Good precedent to follow in our own baselines/benchmarks: never silently average in a sentinel value.

### 2.4 Seed/demo chunk data with concrete numbers — `scripts/seed_pgvector.py`

For local RAG evaluation, only the **`ACME`** tenant may be seeded (code-enforced — seeding a real tenant like `alcf` is blocked to avoid contaminating its RAG, `scripts/seed_pgvector.py:18-41`). The seed script hardcodes one concrete example scenario, useful as a sanity-check reference for our own toy instance:

| Field | Value |
|---|---|
| `action.target` | `it_load_kw` |
| `action.current` → `action.suggested` | 420.0 kW → 378.0 kW |
| `action.delta` | −42.0 kW |
| `reason_codes` | `HIGH_PUE`, `THERMAL_HEADROOM` |
| `expected_impact.pue_delta` | −0.12 |
| `estimated_impact_24h.energy_kwh` | 1008.0 kWh saved |
| `estimated_impact_24h.energy_cost_usd` | $80.64 |
| `estimated_impact_24h.co2e_kg` | 389.1 kg |

Source: `scripts/seed_pgvector.py:53-177`.

### 2.5 Tenant-specific domain framing — `prompts/alcf.py`, `prompts/acme.py`

These aren't telemetry per se, but they define **which metrics each tenant profile is "in scope" for**, which matters for scenario/site-config design later:

- **ALCF** (HPC): `cpu_util_pct`, `it_load_kw_raw/optimized`, `pue_raw/optimized`, `carbon_intensity`; constraints = node maintenance windows, partition reservations, compute SLAs.
- **ACME** (commercial): `PUE`, `IT load (kW)`, `carbon_intensity`, `energy_cost`; constraints = commercial availability SLAs, maintenance windows, demand peaks, carbon reporting.

---

## 3. Physical constraints inventory, by category (power / cooling / water / thermal)

This directly answers the task's second half and should double as a head start on next task ("map which constraints and decision variables are observable through the mock interfaces").

### ⚡ Power

**Observable via NOI mock interfaces — yes, this is the best-covered category:**

| Variable | Where | Notes |
|---|---|---|
| `it_load_kw` (raw / optimized) | `telemetry_chunker.py` numeric fields; `blackbox_mock.py` action target | Primary decision variable across all 4 mock tenants |
| `pue_raw` / `pue_optimized` | same | Physically bounded ≥ 1.0 — enforced in mock-consuming code |
| `energy_kwh` (24h impact) | `blackbox_mock.py` `estimated_impact_24h` | Signed; negative = savings |
| `energy_cost_usd` | same | USD, tied to `energy_kwh` |
| `cpu_avg` / `cpu_util_pct` | `telemetry_chunker.py`; `prompts/alcf.py` | Proxy driver referenced but the actual CPU→power physics formula is **not** in this repo (lives in DFI's `pib-simulator`, per onboarding docs — out of scope here) |
| `co2_kg` / `co2e_kg` | both mock layers | Carbon, downstream of power, included as an "impact" not a standalone constraint |

**Not observable here (background only, DFI-side per onboarding docs, not fetchable through NOI's mock interfaces):** the actual non-linear power curve (SPECpower idle-factor model) lives in `pib-simulator/telemetry_engine.py`, a DFI repo we don't have. NOI's mock only sees the *outputs* (kW numbers), not the generating formula.

### ❄️ Cooling

**Partially observable — present as action targets and narrative constraints, not as a standalone numeric telemetry field:**

| Variable | Where | Notes |
|---|---|---|
| `chiller_setpoint_c` | `blackbox_mock.py` (ACME tenant) | A decision variable (action target), not a telemetry reading |
| `pump_b12_duty_pct` | `blackbox_mock.py` (tenant-001) | Cooling-pump duty cycle as a decision variable |
| "N+1 cooling redundancy" | `blackbox_mock.py` constraints (free text) | A real constraint, but **not machine-readable** — no numeric bound object |
| "FREE_COOLING_AVAILABLE" | reason code | Qualitative signal, not a number |
| Cooling *load* in kW, cooling energy (`cooling_kwh`) | **Not found in this repo** | Present on the DFI side (`baseline.cooling_kwh`, Parquet `cooling_load` column per onboarding docs) — not exposed through anything in `pib-llm-backend-main` |

**Gap worth flagging:** if the optimization formulation needs a numeric cooling-power variable (not just a setpoint lever), it isn't in the NOI mock today — would need to either derive one (e.g. `facility_power - it_load` as a proxy, following the DFI physics convention from the onboarding docs) or request it be added to the mock.

### 💧 Water

**Not observable via NOI mock interfaces at all — confirmed by exhaustive grep, this is a hard gap, not something I missed:**

I grepped the entire `pib-llm-backend-main` repo for `water` (case-insensitive) and found **zero** hits related to water consumption — the only matches were unrelated occurrences of the word "water**mark**" (the RAG poller's incremental-sync checkpoint, an unrelated concept).

For context (from `physaflow-onboarding-main`, DFI-side, **not accessible as a mock interface here**): water usage does exist as a concept in the platform — `baseline.water_liters` in the DFI ingest payload contract, and a `water_use_proxy` column in the raw Parquet telemetry the DFI producers write. But **none of that is piped into NOI's mock data or its RAG chunks**. If water needs to be part of our optimization formulation, it currently has **no mock representation to prototype against** — this should go straight into the Day-3 clarification memo to Leo.

### 🌡️ Thermal

**Present narratively/qualitatively; not present as a first-class numeric telemetry field in the mock:**

| Variable | Where | Notes |
|---|---|---|
| "thermal redline (28°C inlet)" | `blackbox_mock.py` constraint text (ALCF) | A real numeric threshold, but embedded in a free-text string, not a structured field |
| `THERMAL_HEADROOM` / `THERMAL_HEADROOM_AVAILABLE` | reason codes | Qualitative signal only — no accompanying numeric headroom value |
| `setpoint_temperature_c` | `blackbox_mock.py` action target (tenant-001) | A decision variable (target setpoint), not a sensor reading |
| Inlet/return air temp (`avg_inlet_temp_c`, `return_air_temp_c`), hotspot count | **Not found in this repo** | Present in the DFI Parquet schema (per onboarding docs) — not exposed through NOI's mock |
| Thermal recirculation model (ASHRAE) | **Not in this repo** | Lives in DFI's `pib-simulator`, out of scope |

**Practical implication for our formulation:** thermal, like water, mostly shows up in NOI's mock data as *narrative justification* for a power/cooling decision (e.g. "thermal headroom enables a load reduction"), not as an independently observable/controllable state variable. If the RL/optimization formulation wants thermal as a first-class state dimension, it will need either (a) a proxy derived from PUE/power (à la the DFI physics), or (b) a request to extend the mock — this is exactly the kind of question the tracker's "Map which constraints and decision variables are observable through the mock interfaces" task (next item) should resolve with Leo/James.

---

## 4. Summary table — coverage at a glance

| Category | Numeric telemetry in NOI mock? | Decision variables in NOI mock? | Structured (machine-readable) constraints? | Real physics/formula accessible to us? |
|---|:---:|:---:|:---:|:---:|
| Power | ✅ yes (`it_load_kw`, `pue`, `energy_kwh`, `co2_kg`) | ✅ yes | ⚠️ narrative only | ❌ no (DFI repo, out of scope) |
| Cooling | ❌ no dedicated field | ✅ yes (`chiller_setpoint_c`, `pump_duty_pct`) | ⚠️ narrative only | ❌ no |
| Water | ❌ none at all | ❌ none | ❌ n/a | ❌ no (DFI concept only) |
| Thermal | ❌ no dedicated field (temps live in constraint text) | ✅ yes (`setpoint_temperature_c`) | ⚠️ narrative only | ❌ no |

---

## 5. Open questions for Leo / the Day-3 clarification memo

1. **Where does the canonical "Glossary" actually live?** I couldn't find one in the repos I have access to — is it a Notion/Drive doc, or should this document become that reference?
2. **Water is entirely absent from NOI's mock interfaces.** Should our optimization formulation include it (using DFI's `water_liters`/`water_use_proxy` as a conceptual reference even though we can't query it directly), or is water explicitly out of scope for the Decision Intelligence workstream this summer?
3. **Constraints are free text, not structured bounds.** For the Week 2 constrained multi-objective formulation, do we get access to a structured constraints schema, or are we expected to hand-encode the numeric bounds we can parse out of strings like `"thermal redline (28°C inlet)"`?
4. **Thermal state variables** (inlet/return temp) exist on the DFI side but not in anything we can touch — is deriving a proxy from PUE acceptable, or is there a plan to surface real thermal fields into NOI's mock/real data path?

---

*All file:line references verified by direct inspection on 2026-07-12. No Blackbox/BioCore internals were accessed, inspected, or reasoned about in preparing this document — consistent with the Role 16 Scope & Boundaries.*
