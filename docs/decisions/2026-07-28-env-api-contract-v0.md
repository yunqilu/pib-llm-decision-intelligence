# Env API contract, v0 — finalized (James)

## Decision

`env/simulator.py: DecisionIntelligenceEnv` is the canonical environment.
Contract:

```python
env = DecisionIntelligenceEnv(scenario_config)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(action)
```

`scenario_config`, `observation`, and `action` are plain JSON-serializable
dicts, shaped by `env/schemas/*.schema.json` (already checked in by Yunqi,
`docs/data_schema.md` v0.2). This is now implemented, tested
(`tests/test_simulator.py`), and exercised end-to-end by two baseline
policies (`optimization/baselines/policies.py`) via the shared harness
(`shared/benchmark_harness.py`).

## Context: reconciling with the earlier Gym-style prototype

An earlier draft (`PhysaFlowDispatchEnv`, not in this repo) used
`gymnasium.spaces.Dict`/`MultiDiscrete`: fixed-size numpy arrays, tasks
addressed by queue-slot index, hosts addressed by integer index. That draft
predates the finalized v0.2 data contract that Yunqi and I signed off on
(`docs/data_schema.md`, `env/schemas/*.schema.json`), which instead uses:

- variable-length lists (`hosts.queue`, `action.placements`,
  `action.migrations`) instead of fixed-size padded arrays,
- string `task_id`/`host_id` instead of integer indices,
- plain JSON instead of numpy — because `scenario_config`/`observation`/
  `action` also have to cross the boundary into Yunqi's optimization track
  (Pyomo/HiGHS consumes `scenario_config` directly for the certainty-
  equivalent LP, joint spec §5.4) and into `shared/benchmark_harness.py`,
  neither of which should need a Gym dependency.

**Decision: the JSON/dict contract is canonical.** It is what
`env/schemas/*.schema.json` already validates, what the optimization track
already expects, and what the harness scores. Fixed-shape numpy/Gym spaces
are a poor fit for a queue whose length varies step to step — padding to
`max_queue_len` either silently drops tasks above the pad, or wastes a lot
of the action space on masked-out no-ops.

**The Gym-style interface is not thrown away — it becomes a thin adapter**,
not the source of truth: `rl/gym_wrapper.py` wraps
`DecisionIntelligenceEnv` behind a `gymnasium.Env`-compatible surface for
whichever RL library expects one, translating fixed-size padded arrays to
and from the JSON contract (padding/truncating `hosts.queue` to
`max_queue_len`, mapping host-index actions back to `host_id` strings). Any
translation bug is local to that one file and never touches
`env/simulator.py` or the harness.

## Consequences

- Optimization track, RL track, and the harness all consume literally the
  same `observation`/`action` payloads — no drift between what the LP sees
  and what the RL policy sees.
- `info["action_mask"]` (hard constraints: power cap, thermal surrogate, SLA
  floor, action windows — joint spec §3.2) is defined once, in
  `env/simulator.py`, and is authoritative for both a raw JSON-contract
  policy (`optimization/baselines/policies.py`) and anything routed through
  the Gym wrapper.
- RL code that wants fixed-shape tensors pays the translation cost in one
  adapter file, not in the environment itself.
- `reward` is *not* returned as a scalar by `env.step()` — see
  `env/simulator.py` module docstring and `docs/joint_problem_spec_checklist_james.md`
  ("Reporting": every result must disclose its `lambda` vector). Scalarizing
  is `shared/benchmark_harness.scalarize()`'s job, called explicitly by
  whoever is evaluating a policy, so the `lambda` used is always visible in
  that caller's code rather than buried inside the env.

## Follow-up: spec-conformance pass (2026-07-29)

A review against `docs/problem_spec.md` found that v0 accepted but silently
dropped several fields/constraints. Fixed:

- `curtail_kw` and `deferrable_share_pct` (`d_t`, `q_t`) are now wired into
  `L_t` per §3: `L_t = L_host*(q_t/100) - d_t`, where `L_host` is the
  coupling-identity quantity from host dispatch. Previously these fields
  were schema-accepted, clamped into `self.levers`, and never used again.
- **SLA floor (§4.3)** is now enforced: `_effective_load()` caps how much
  `curtail_kw` can actually reduce `L_t` so it never dips below
  `sla_floor_kw[t]`. Previously `sla_floor_kw` was loaded and length-checked
  at init and never referenced again.
- **N+1 cooling reserve (§4.5)** is now enforced: `_enforce_n_plus_one()`
  clamps `chiller_setpoint_c`/`pump_duty_pct` each step so `kappa_t <=
  rho*kappa_max` always holds. Previously `kappa_max`/`rho` were never
  referenced.
- **Thermal redline (§4.2)** is now a real hard mask on `info["action_mask"]`
  (checked per candidate placement, same mechanism as the power cap),
  instead of only being checked after the fact and logged to `violations`.
  This was the one genuine correctness gap relative to joint spec §3.2,
  which says thermal should be masked like power cap, not caught post-hoc.

Ramp limit (§4.6) remains deliberately unenforced as a hard constraint —
that's correct per the joint spec's own cross-mapping table (§3.2), which
marks ramp as soft and folds it into `f4_risk` via migration churn cost, not
a mask.

Verified with 4 new regression tests (`tests/test_simulator.py`): each fix
is tested against a case constructed so the constraint is genuinely
binding, not vacuously satisfied (e.g. the default surrogate coefficients
never actually reach the N+1 cap, so that test tightens `kappa_max`/`rho`
to force it). Full suite (30 tests) and the baseline comparison both still
pass with zero violations.

- `rl/gym_wrapper.py`'s `max_queue_len` padding needs a real max-arrival-rate
  analysis once Week 5 stress tests (arrivals family/params) are set; a
  too-small pad silently drops tasks the JSON contract would have exposed.
- Migration semantics in `env/simulator.py` v0 operate at the aggregate
  host-power level (move `min(movable, room)` kW between two hosts), not a
  per-task ledger of "which running task is on which host" — `action.
  migrations[].task_id` is schema-required but not yet consumed. Revisit if
  Layer B needs per-task migration tracking (e.g. for churn cost that
  depends on *which* task moved, not just how much power moved).
