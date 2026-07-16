# Shared Problem Specification — Constrained Multi-Objective Optimization Formulation

**Status:** DRAFT v0.1 (2026-07-15) — optimization-side formulation drafted by Yunqi.
Pending reconciliation with James's MDP formulation (Week 2 joint task) and review
by Leo / technical lead. Every constraint below is tagged **[sourced]** (numeric
value appears in the mock data) or **[approximated]** (hand-encoded from free-text
constraints or a synthetic surrogate — see §7).

Companion docs: `data_schema.md` (observation fields), the Week 1 mock-telemetry
inventory (variable provenance, file:line references).

---

## 1. Setting and scope

One site (tenant) per problem instance; instance parameters come from a scenario
config (matching `env/simulator.py: reset(scenario_config)`). Multi-period horizon
of `T = 24` hourly steps, `t ∈ {1, …, T}`, chosen to match the mock Blackbox's
`estimated_impact_24h` window and to align one optimization horizon with one RL
episode.

Out of scope for v0 (per the Week 1 inventory and pending Leo's answers to the
Day-3 memo): water as a modeled resource; structured thermal telemetry (inlet
temp is handled via a surrogate, §4.2); Blackbox/BioCore internals; real
production data.

## 2. Parameters (from scenario config)

| Symbol | Meaning | Units | Provenance |
|---|---|---|---|
| `L̄_t` | Baseline IT load profile | kW | mock `it_load_kw` (e.g. ALCF 280.5) [sourced] |
| `P_cap` | Facility power capacity | kW | scenario config [approximated] |
| `L̄^SLA_t` | Minimum IT load protected by SLA | kW | "Maintain HPC job queue SLA" [approximated] |
| `T_max` | Inlet-temperature redline | °C | "thermal redline (28°C inlet)" → 28 [sourced text, hand-parsed] |
| `W ⊆ {1,…,T}` | Operational window (steps where actions are allowed) | — | "Operational window: 02:00–06:00" [sourced text, hand-parsed] |
| `p_t` | Electricity price | $/kWh | derived from mock `energy_cost_usd / energy_kwh` ≈ 0.08 [sourced] |
| `γ_t` | Carbon intensity | kg CO₂e/kWh | derived from mock `co2e_kg / energy_kwh` ≈ 0.386 [sourced] |
| `β = (β₀, β₁, β₂, β₃)` | Cooling-overhead surrogate coefficients (§4.1) | — | synthetic [approximated] |
| `a = (a₀, a₁, a₂)` | Inlet-temperature surrogate coefficients (§4.2) | — | synthetic [approximated] |
| `ρ ∈ (0,1)` | N+1 cooling reserve factor (§4.5) | — | "Maintain N+1 cooling redundancy" [approximated] |
| `κ₀, κ_c, κ_u, κ^max` | Cooling-effort aggregation coefficients and cap (§3, §4.5) | — | synthetic [approximated] |
| `D` | Stranded-capacity denominator for recovery % | kWh | scenario config; computed once per scenario by the harness [approximated] |
| bounds `c^min/max, s^min/max, u^min/max, q^min, R, Δ^max` | lever bounds, ramp & per-step move limits | — | mock action ranges + text (e.g. `s^min = 18` °C from tenant-001) [mixed] |

## 3. Decision variables

All indexed by `t ∈ {1,…,T}` unless noted. The five levers are exactly the five
action targets observed in the mock Blackbox (Week 1 inventory §2.1).

| Variable | Domain | Meaning | Mock action target |
|---|---|---|---|
| `r_t` | `≥ 0` (continuous, kW) | **Recovered capacity**: newly admitted IT load above baseline | (the mission variable; not a mock action — see §5) |
| `d_t` | `≥ 0` (continuous, kW) | Curtailment of baseline IT load | `it_load_kw` (reduce) |
| `c_t` | `[c^min, c^max]` (°C) | Chiller supply setpoint | `chiller_setpoint_c` |
| `u_t` | `[u^min, u^max]` (%) | Cooling-pump duty | `pump_duty_pct` |
| `s_t` | `[s^min, s^max]` (°C) | Room/zone temperature setpoint | `setpoint_temperature_c` |
| `q_t` | `[q^min, 100]` (%) | Production/deferrable load share | `production_load_pct` (shift) |
| `z_t` | `{0,1}` | Action-window indicator (binary → MILP) | encodes operational/maintenance windows |

Derived (not free) quantities:

- Effective IT load: `L_t = L̄_t · (q_t / 100) − d_t + r_t`
- Cooling effort (aggregate lever, used by surrogates): `κ_t = κ₀ + κ_c (c^ref − c_t) + κ_u (u_t / 100)` — affine in levers [approximated]
- Facility power (affine surrogate, §4.1): `F_t = L_t + C_t`

## 4. Constraints

### 4.1 Power capacity [approximated surrogate, sourced bound shape]

Cooling/overhead power is an affine surrogate (keeps the model linear —
deliberately avoids the bilinear `PUE_t · L_t` form):

```
C_t = β₀ + β₁ L_t − β₂ (c_t − c^ref) + β₃ (u_t / 100)        ∀t
F_t = L_t + C_t ≤ P_cap                                       ∀t
C_t ≥ 0                                                       ∀t
```

`C_t ≥ 0` guarantees implied `PUE_t = F_t / L_t ≥ 1.0`, consistent with the
codebase's PUE honesty rule (`telemetry_chunker.py`, NOI-213). PUE is a *reported
output*, never a modeled ratio inside constraints.

### 4.2 Thermal redline [bound sourced from text; surrogate approximated]

Inlet temperature is not observable in the mock (Week 1 inventory §3), so a
declared linear surrogate stands in for it:

```
T^in_t = a₀ + a₁ L_t − a₂ κ_t ≤ T_max (= 28 °C)               ∀t
```

Coefficients `a` are synthetic and must be stated in every scenario config.
**Flagged assumption** — if DFI-side thermal fields ever surface, replace this.

### 4.3 SLA / service protection [approximated]

Curtailment may never cut into SLA-protected load, and production shifting may
not touch the critical line ("No afectar linea de produccion critica"):

```
L̄_t · (q_t / 100) − d_t ≥ L̄^SLA_t                            ∀t
q_t ≥ q^min                                                   ∀t
```

### 4.4 Operational / maintenance windows [sourced text, hand-parsed] — the MILP part

Actions are only allowed inside the window `W`; `M` is a big-M constant
(`M = max_t L̄_t` suffices):

```
z_t = 0                          ∀t ∉ W
d_t ≤ M z_t                      ∀t
|c_t − c_{t−1}| ≤ Δ^max_c z_t    ∀t      (idem for s_t, u_t, q_t)
```

**Note:** dropping §4.4 (or fixing `z_t` a priori, which is possible whenever `W`
is known in advance — true for all 4 mock tenants) reduces the whole model to a
pure **LP**. This is the planned "toy instance" configuration and feeds the
Week 2 solver-choice task.

### 4.5 Cooling redundancy (N+1) [approximated]

N+1 redundancy encoded as a reserve margin on cooling effort:

```
κ_t ≤ ρ · κ^max                                               ∀t
```

### 4.6 Ramp limits [approximated]

```
|L_t − L_{t−1}| ≤ R                                           ∀t ≥ 2
```

### 4.7 Physical bounds [mixed]

Variable domains of §3, including `s_t ≥ 18` °C (tenant-001, sourced text) and
`PUE ≥ 1.0` (implied by §4.1).

## 5. Objectives

Four objectives; all linear in the decision variables (absolute values here and
in §4.4/§4.6 are linearized with standard auxiliary variables).

```
f₁ = Σ_t r_t · Δt                      (kWh recovered)            → maximize
f₂ = Σ_t p_t · F_t · Δt                (energy cost, $)           → minimize
f₃ = Σ_t γ_t · F_t · Δt                (CO₂e, kg)                 → minimize
f₄ = Σ_t (w_d d_t + w_c |c_t − c^ref| + w_s |s_t − s^ref|) / T   (risk proxy)  → minimize
```

`Δt = 1 h`. **Recovery %** (the headline benchmark metric) is reported as
`f₁ / D` where `D` is the scenario's stranded-capacity denominator — kept as a
scenario parameter so the optimizer's objective stays linear and the harness owns
the normalization. `f₄` is a declared proxy (mock `risk` is an output score, not
a modeled function) [approximated].

### 5.1 Primary scalarization: ε-constraint

```
max f₁   s.t.   f₂ ≤ ε₂,  f₃ ≤ ε₃,  f₄ ≤ ε₄,  and §4.
```

Sweeping `(ε₂, ε₃, ε₄)` over a grid generates the Pareto front required in
Week 4 (capacity vs. quality vs. latency/robustness), including non-convex
segments, with no unit-commensuration of objectives needed.

### 5.2 Alternative scalarization: weighted sum (recorded, not primary)

```
max  λ₁ f₁ − λ₂ f₂ − λ₃ f₃ − λ₄ f₄,   λ ≥ 0.
```

Simpler single solve; kept as a fallback and as the natural bridge to a scalar
RL reward (§6). Known limitation: cannot reach non-convex Pareto points, and the
`λ` weights mix units.

### 5.3 Appendix: efficiency form (equivalent alternative framing)

Fix `r_t = 0` and minimize `f₂` (or `Σ_t F_t`): the classic "same load, less
facility power" framing that matches what the mock recommendations literally do.
Recovered capacity is then *derived* as `r̂_t = (F̄_t − F*_t) / PUE^ref` — the
freed facility power converted back to admissible IT load. The admission form
(§5.1) is primary because its objective *is* the mission metric; the efficiency
form is retained because (a) it needs no `P_cap` estimate and (b) it maps 1:1 to
the existing mock recommendations for validation.

## 6. Mapping to the MDP formulation (for reconciliation with James)

| This spec | MDP |
|---|---|
| observation fields (`data_schema.md`) + `t` | state `s_t` |
| `(d_t, c_t, u_t, s_t, q_t)` (and `r_t` if admission is an agent action) | action `a_t` |
| stage term of §5.2 weighted sum | reward `r(s_t, a_t)` |
| constraints §4 | feasibility: action masking / projection, or penalty terms in reward (James to choose) |
| horizon `T = 24` | episode length |
| scenario config | environment parameters at `reset()` |

Open reconciliation questions: (1) hard constraints vs. reward penalties on the
RL side; (2) is `r_t` an agent action or an environment response to freed
headroom; (3) shared random seeds / scenario IDs for head-to-head benchmarking.

## 7. Known gaps and declared assumptions

- **Water:** entirely absent from NOI mock interfaces (Week 1 inventory §3);
  excluded from v0. Revisit on Leo's answer to Day-3 memo Q2.
- **Thermal surrogate (§4.2)** and **cooling-overhead surrogate (§4.1)**:
  synthetic affine models; coefficients live in scenario configs and must be
  reported alongside any result.
- **Free-text constraints** hand-parsed into bounds; each tagged above. If a
  structured constraints schema materializes (Day-3 memo Q3), replace the
  hand-encoded values.
- **Risk (`f₄`)** is a proxy, not the mock's `risk` field.
- **`P_cap` and `D`** are scenario inventions — the mock exposes no capacity
  figure. Sensitivity to both must be part of Week 5 stress testing.
