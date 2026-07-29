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

## Open follow-ups

- `rl/gym_wrapper.py`'s `max_queue_len` padding needs a real max-arrival-rate
  analysis once Week 5 stress tests (arrivals family/params) are set; a
  too-small pad silently drops tasks the JSON contract would have exposed.
- Migration semantics in `env/simulator.py` v0 operate at the aggregate
  host-power level (move `min(movable, room)` kW between two hosts), not a
  per-task ledger of "which running task is on which host" — `action.
  migrations[].task_id` is schema-required but not yet consumed. Revisit if
  Layer B needs per-task migration tracking (e.g. for churn cost that
  depends on *which* task moved, not just how much power moved).
