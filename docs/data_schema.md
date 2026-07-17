# Data Schema: the env data contract (v0.2)

Yunqi Lu

**Consumers:** `env/simulator.py`, `optimization/` (MILP/LP), `rl/` (MDP), `shared/benchmark_harness.py`.
**Machine-checkable versions:** [`env/schemas/`](../env/schemas/) (JSON Schema, draft 2020-12). Validate any instance with:

```bash
python env/schemas/validate.py env/scenarios/alcf_toy.json
```

*(Canonical copy — repo layout: this doc `docs/data_schema.md`, schemas + validator `env/schemas/`, sample `env/scenarios/alcf_toy.json`.)*

This specifies the **four payloads** that cross the env boundary, `scenario_config` (in), `observation` (out), `action` (in), `info` (out) and the **provenance mapping** from PhysaFlow NOI's mock interfaces to every field. 

Sources: Week 1 mock inventory, Week 2 formulation §1, joint spec §5.

---

## 0. Conventions (binding)

1. **Units live in field names**: `_kw`, `_kwh`, `_c` (°C), `_pct` (0–100), `_usd`, `_kg`. No unitless physical quantities.
2. **Time**: arrays are length `T` (default 24), 0-indexed; index `t` covers wall-clock interval `[t, t+1)` hours. The formulation's `t ∈ {1..24}` maps to array index `t−1`.
3. **PUE honesty rule (NOI-213)**: `pue` is *report-only, never an input or constraint variable*. Value is a float `≥ 1.0` or `null` ("not reported"). Never `0.0`, never averaged over nulls.
4. **Provenance**: every parameter group carries `"provenance": "sourced" | "approximated" | "mixed"`, matching the formulation's tags. Hand-parsed free-text constraints keep the original string in `"source_text"`.
5. **Signs**: deltas are signed; negative = reduction/savings (mock convention).
6. **One tenant per scenario** (D6). Water fields do not exist in v0 (see §5).

---

## 1. `scenario_config`: input to `reset(scenario_config)`

The single source is read by both layers. The MILP consumes it **directly** (certainty-equivalent planning: baseline profile = realized profile); the simulator uses it to generate the episode the RL agent sees. Schema: [`scenario_config.schema.json`](../env/schemas/scenario_config.schema.json).

### 1.1 `meta`

| Field         | Type  | Notes                                                               |
| ------------- | ----- | ------------------------------------------------------------------- |
| `scenario_id` | str   | unique; cite in every result (provenance discipline, joint spec §4) |
| `tenant`      | str   | one of the mock tenants: `alcf`, `acme`, `tenant-001`, `tenant-002` |
| `T`           | int   | horizon steps, default 24                                           |
| `dt_hours`    | float | step length, default 1.0                                            |
| `seed`        | int   | controls all random draws (arrivals, load wobble)                   |

### 1.2 `facility`  Layer A parameters (formulation §1)

| Field                      | Symbol    | Type          | Provenance                                                                                     |
| -------------------------- | --------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `baseline_it_load_kw`      | `L̄_t`    | float[T]      | **sourced** : mock `it_load_kw` (e.g. ALCF 280.5)                                              |
| `p_cap_kw`                 | `P_cap`   | float         | approximated (scenario invention, Week 5 sensitivity)                                          |
| `sla_floor_kw`             | `L^SLA_t` | float[T]      | approximated, parsed from SLA constraint text                                                  |
| `t_max_inlet_c`            | `T_max`   | float         | sourced text, hand-parsed ("thermal redline (28°C inlet)")                                     |
| `price_usd_per_kwh`        | `p_t`     | float[T]      | sourced, mock `energy_cost_usd / energy_kwh` ≈ 0.08                                            |
| `carbon_kg_per_kwh`        | `γ_t`     | float[T]      | sourced, mock `co2e_kg / energy_kwh` ≈ 0.386                                                   |
| `stranded_denominator_kwh` | `D`       | float \| null | `null` ⇒ harness computes it (joint spec §3.1); both layers must report against the same value |

### 1.3 `facility.levers`  bounds for the 5 mock action targets (+ ramp)

Per lever `c` (chiller setpoint °C), `u` (pump duty %), `s` (zone setpoint °C), `q` (deferrable share %): `{min, max, delta_max, ref?}`. Plus `ramp_kw` (`R`, §3.6) and `q.min` (`q^min`). Provenance: mixed (mock ranges + text, e.g. `s.min = 18` from tenant-001).

### 1.4 `facility.surrogates`  declared coefficients [all approximated]

Synthetic stand-ins for physics we cannot observe (DFI-side). **Must be reported with every result.**

| Field              | Symbols                    | Used in                                                 |
| ------------------ | -------------------------- | ------------------------------------------------------- |
| `cooling_overhead` | `β₀..β₃`, `c_ref`          | `C_t = β₀ + β₁L_t − β₂(c_t−c_ref) + β₃(u_t/100)` (§3.1) |
| `inlet_temp`       | `a₀..a₂`                   | `T^in_t = a₀ + a₁L_t − a₂κ_t` (§3.2)                    |
| `cooling_effort`   | `κ₀, κ_c, κ_u, κ_max, rho` | `κ_t` aggregation + N+1 reserve `κ_t ≤ ρκ_max` (§3.5)   |

### 1.5 `facility.action_window`  the MILP part (§3.4)

| Field         | Type  | Notes                                                                                                                                                                  |
| ------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hours`       | int[] | `W`, e.g. `[2,3,4,5]` for "02:00–06:00" (sourced text, hand-parsed)                                                                                                    |
| `z_fixed`     | bool  | `true` ⇒ binaries fixed ⇒ Layer A is a pure **LP** (toy-instance config). Remember `core.relax_integer_vars` after fixing, or HiGHS still treats it as MIP (no duals). |
| `source_text` | str   | original constraint string                                                                                                                                             |
|               |       |                                                                                                                                                                        |

### 1.6 `hosts`  Layer B inventory **[owner: James  Q2]**

Synthetic; no mock field exists for any of this. James fills defaults, schema fixes the shape:

| Field                 | Type        | Notes                                                                         |
| --------------------- | ----------- | ----------------------------------------------------------------------------- |
| `resources`           | str[]       | the `M` resource axes, default `["cpu","mem","power","storage"]`              |
| `inventory[i]`        | object      | per host: `host_id`, `rated_power_kw` (**`P_i^rated`**), `rated` per resource |
| `initial_utilization` | float[N][M] | `h_{i,j} ∈ [0,1]` at t=0                                                      |

**Coupling contract (joint spec §3.1):** `Σ_i rated_power_kw` must be consistent with the facility scale implied by `baseline_it_load_kw`; the simulator asserts `Σ_i h_{i,pow}·P_i^rated ≈ baseline_it_load_kw[0]` at reset (relative tolerance `1e-3`).

### 1.7 `arrivals`  task arrival process **[owner: James  Q3]**

| Field | Type | Notes |
|---|---|---|
| `family` | enum | `"none"` (toy instance / certainty-equivalence check) \| `"poisson"` \| `"trace"` |
| `rate_per_hour` | float \| float[T] | for `poisson` |
| `demand` | object | distribution of `δ_k` per resource (family + params) |
| `trace_path` | str | for `family="trace"` |

Week 5 stress tests vary this block only — everything else stays fixed.

---

## 2. `observation`  output of `reset()` and `step(action)`

What a policy/solver may *see*. Schema: [`observation.schema.json`](../env/schemas/observation.schema.json).

### 2.1 `facility` view (both layers)

| Field                                    | Symbol            | Type                | Mock origin                                       |
| ---------------------------------------- | ----------------- | ------------------- | ------------------------------------------------- |
| `t`                                      | —                 | int                 | step index                                        |
| `it_load_kw`                             | `L_t`             | float               | chunker `it_load_raw/optimized`                   |
| `facility_power_kw`                      | `F_t`             | float               | derived `L_t + C_t` (no direct mock field)        |
| `pue`                                    | —                 | float ≥ 1.0 \| null | **report-only** (rule §0.3)                       |
| `inlet_temp_c`                           | `T^in_t`          | float               | surrogate output [approximated] — no mock sensor  |
| `cooling_effort`                         | `κ_t`             | float               | surrogate output [approximated]                   |
| `levers`                                 | `c_t,u_t,s_t,q_t` | object              | current setpoints (mock action targets)           |
| `admitted_kw`                            | `r_t` so far      | float               | mission variable (not in mock)                    |
| `budget_remaining_kw`                    | `r*_t − r_t`      | float               | Layer A plan → Layer B envelope (joint spec §3.1) |
| `price_usd_per_kwh`, `carbon_kg_per_kwh` | `p_t, γ_t`        | float               | echoed from config                                |
| `in_window`                              | `z_t`             | bool                | moves/migrations allowed this step                |

### 2.2 `hosts` view (Layer B only; `null` for the pure-LP optimizer)

| Field | Symbol | Type |
|---|---|---|
| `utilization` | `H_t` = `h_{i,j}` | float[N][M], each ∈ [0,1] |
| `queue[k]` | `v_k` | `{task_id, demand: {cpu, mem, power_kw, storage}, sla_protected: bool, arrived_t}` — `demand` is `δ_k` |

**Masking (joint spec §3.2):** the env returns `action_mask` inside `info`, derived from the hard constraints (power cap, thermal surrogate, SLA floor, windows). Hard constraints are **masked, never penalized**.

---

## 3. `action`  input to `step(action)`

Schema: [`action.schema.json`](../env/schemas/action.schema.json). A step accepts **either or both** layers' fields:

| Layer                | Field        | Content                                                                                                                                |
| -------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| A (plan / setpoints) | `facility`   | `{admit_kw (r_t), curtail_kw (d_t), chiller_setpoint_c (c_t), pump_duty_pct (u_t), zone_setpoint_c (s_t), deferrable_share_pct (q_t)}` |
| B (dispatch)         | `placements` | list of `{task_id, host_id}` <br>total placed power ≤ `budget_remaining_kw`                                                            |
| B (dispatch)         | `migrations` | list of `{task_id, from_host, to_host}` <br>**only when `in_window`**                                                                  |

The 5 facility levers are exactly the 5 mock Blackbox action targets; `admit_kw` is the one variable the mock does not have (D1, admission form).

---

## 4. `info`  diagnostics returned by `step`

Not for training on (except `action_mask`). Per step: objective-term increments `{f1_recovery_kwh, f2_cost_usd, f3_co2_kg, f4_risk}`, `recovery_pct` (running `f₁/D`), `action_mask`, `violations` (must stay empty. a non-empty list is a simulator bug, not a policy cost), and `provenance` echo (`scenario_id`, `schema_version`, surrogate coefficients, `λ` vector if RL).

---

## 5. Mock → schema provenance map (summary)

| Mock interface field                                                                                      | Schema field                                                           | Status                       |
| --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------- |
| `blackbox_mock` / chunker `it_load_kw`, `it_load_raw/optimized`                                           | `facility.baseline_it_load_kw`, obs `it_load_kw`                       | sourced                      |
| chunker `pue_raw/optimized`                                                                               | obs `pue` (report-only)                                                | sourced                      |
| chunker `cpu_avg` / `cpu_util_pct`                                                                        | `hosts.initial_utilization[:,cpu]` (aggregate anchor only)             | mixed                        |
| `estimated_impact_24h.energy_kwh` + `.energy_cost_usd`                                                    | `facility.price_usd_per_kwh` (ratio)                                   | sourced                      |
| `estimated_impact_24h.co2e_kg` (per kWh)                                                                  | `facility.carbon_kg_per_kwh` (ratio)                                   | sourced                      |
| `summary.confidence`, `summary.risk_score`                                                                | **not consumed** in v0 (mock-authored scalars, no physics behind them) | dropped                      |
| action targets `chiller_setpoint_c`, `pump_b12_duty_pct`, `setpoint_temperature_c`, `production_load_pct` | `facility.levers` bounds + `action.facility`                           | sourced                      |
| free-text constraints (28 °C redline, SLA, windows, N+1, 18 °C floor)                                     | hand-parsed numeric fields + `source_text`                             | sourced text, hand-parsed    |
| (absent in mock)                                                                                          | `hosts.*`, `arrivals.*`, all surrogate coefficients, `p_cap_kw`, `D`   | **synthetic** [approximated] |

**Hard gaps carried from Week 1 (unchanged):** 

1. water, zero mock representation, excluded from v0 (Q4→Leo); 
2. thermal sensors (inlet/return temp), surrogate only (§1.4); 
3. placement/migration actions, not in mock, hence `hosts`/`arrivals` are scenario inventions (Q1→Leo, Q2/Q3→James).

---

## 6. Versioning

1. Schema changes go through PR + a line in `docs/decisions/`; 
2. bump the version suffix in each JSON Schema's `$id`. 
3. Both tracks pin the schema version in their results (`info.provenance.schema_version`).
