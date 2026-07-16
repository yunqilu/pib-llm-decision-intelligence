# Decision Intelligence Workstream — Shared Scope & Repo Layout
**Status: DRAFT — starting point for the joint discussion with James, not a final decision.**
Prepared by Yunqi Lu, to kick off the Week 1 "JOINT with James" alignment task.

---

## 1. The one shared problem (proposed)

Both of us are studying the same thing from different angles: **how to recover stranded
data-center capacity** (power/cooling/thermal-constrained AI compute that's physically
present but unusable) — framed as a decision-making problem over time.

- **James's angle:** formalize it as an MDP, solve with RL (policy gradient / actor-critic).
- **My angle:** formalize it as a constrained multi-objective optimization problem, solve
  with baselines (greedy) and LP/MILP.

**The point of this doc:** make sure both formalizations describe *the same underlying
problem* (same state/observations, same actions/decision variables, same constraints,
same objective), so the two solution methods can be benchmarked head-to-head on identical
scenarios later (Week 4).

*(To confirm with James: does he agree with this framing, or is his mental model of the
problem different? This is the first thing to settle on the call.)*

## 2. Proposed repo layout

One shared repo (or a shared top-level folder if repos must stay separate — TBD with
James/Leo), so the environment and benchmark are never duplicated:

```
decision-intelligence/
├── README.md                     # problem statement, links to both tracks
├── docs/
│   ├── problem_spec.md           # THE single shared formalization (state, action, reward/objective, constraints)
│   ├── data_schema.md            # what the env exposes to both solvers (see §4)
│   └── decisions/                # short ADR-style notes, e.g. this file once finalized
├── env/                          # SHARED — single source of truth, nobody forks it
│   ├── simulator.py              # env API: reset(), step(action), observation/action space
│   ├── scenarios/                # site configs, load profiles, stress/shift scenarios
│   └── mock_telemetry_adapter.py # bridges NOI's mock interfaces -> env observations
├── shared/                       # SHARED tooling
│   └── benchmark_harness.py      # runs any policy/optimizer, computes recovery %, Pareto front
├── optimization/                 # my track — owned by Yunqi
│   ├── baselines/                # greedy, LP/MILP
│   └── results/
├── rl/                           # James's track — owned by James
│   ├── agents/                   # policy gradient / actor-critic
│   └── results/
└── memos/                        # weekly research memos to Leo (each of us writes our own)
```

**Ownership rule of thumb:** `env/`, `shared/`, and `docs/problem_spec.md` are edited only
by mutual agreement (PR + review from the other). `optimization/` and `rl/` are each
person's own space to move fast in.

*(To confirm with James: does a repo already exist that this should live in, or are we
creating a new one? I don't have visibility into that — need to ask Leo/James.)*

## 3. Env API contract (draft)

A minimal interface both a solver and an RL agent can consume:

```
reset(scenario_config) -> observation
step(action) -> observation, reward, done, info
observation_space: dict of the fields below
action_space: the decision variables below
```

## 4. Data schema — what the environment exposes (draft, from the mock telemetry inventory)

Based on the mock-interface inventory already done this week
(`glossary-and-mock-telemetry-inventory.md`), here's what's realistically available to
build `env/simulator.py` on day one:

**Observable state (numeric):** `it_load_kw` (raw/optimized), `pue` (raw/optimized, ≥1.0),
`cpu_util_pct`, `energy_kwh`, `energy_cost_usd`, `co2_kg`, `confidence`, `risk`.

**Decision variables (actions) seen in the mock so far:** `it_load_kw` (reduce),
`chiller_setpoint_c` (raise/lower), `pump_duty_pct` (reduce), `setpoint_temperature_c`
(raise/lower), `production_load_pct` (shift).

**Known gap — flag to James before building on it:** water is **not represented anywhere**
in NOI's mock interfaces, and thermal/cooling state (inlet/return temps, cooling load) only
shows up as free text inside `constraints` strings, not as structured numeric fields. We
need to jointly decide: (a) exclude water/detailed thermal from v0 scope, or (b) agree on a
synthetic proxy formula, before either of us builds around it.

**Constraints are currently free text, not structured bounds** (e.g. `"Do not exceed
thermal redline (28°C inlet)"`). Proposal: we hand-encode the ones we need as numeric
bounds in `problem_spec.md`, and note explicitly which are approximated vs. sourced.

## 5. Explicitly out of scope (both tracks)

- Blackbox / Core Vault / BioCore internals — hard boundary per Role 16, applies to both of us.
- Real production data / `pilotdb` access — mock interfaces only for v0.
- Water as a first-class modeled constraint — pending resolution of the open question above.

## 6. Open questions for the call with James

1. Does he agree the problem is "one shared MDP/optimization problem, two solution methods,"
   or does he see it differently?
2. Repo: new shared repo, or a folder inside an existing one?
3. Who writes `env/simulator.py` v0 — jointly, or does one of us own the first draft and the
   other reviews?
4. Sync cadence — weekly check-in, or tighter during the shared-deliverable weeks (2, 3, 4, 5, 6 per the tracker)?
5. His read on the water/thermal gap — does his RL formulation need those dimensions more
   than the optimization side does?

---
*Once we've talked, replace this file's status line with "AGREED — <date>" and summarize
the actual decisions in 3-5 bullets at the top, so it reads as the record of what we
settled on, not just my initial proposal.*
