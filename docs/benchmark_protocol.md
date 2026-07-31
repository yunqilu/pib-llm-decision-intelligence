# Benchmark protocol: measuring % stranded capacity recovered

Status: **DRAFT — James, 2026-07-30.** Operationalizes
`docs/joint_problem_spec.md` §5 (5 bullet points) into a concrete,
checkable procedure. Needs Yunqi's sign-off before any reported number
(RL vs. baseline vs. oracle) is treated as final, since the denominator
`D` and the oracle bound both live on her side of the spec.

## 1. The headline metric

**recovery % = f₁ / D**, where `f₁` = kWh of capacity actually recovered
(admitted) over the episode, and `D` = kWh of capacity stranded at baseline
(joint spec §0, §3.1, §4). This is the *only* number that goes in a
headline claim like "policy X recovers Y% of stranded capacity." Cost
(`f₂`), carbon (`f₃`), and risk (`f₄`) are always reported alongside it,
never folded into it silently.

## 2. What "honestly" means here, concretely

Four failure modes we're explicitly guarding against, each with a specific
check:

| Failure mode | Guard |
| --- | --- |
| Comparing runs against different `D` | `D` is computed once per `scenario_id` (`shared/benchmark_harness.compute_D`) and reused by every policy evaluated on that scenario. Never recompute `D` per-policy. |
| A policy "wins" by exploiting a masking bug instead of real dispatch skill | `info["violations"]` must be `[]` for every episode in a reported result. A non-empty list voids that run's result — it means the env has a bug, not that the policy is being penalized (see `env/simulator.py` module docstring). Re-run after fixing, don't just subtract a penalty. |
| A reward-scalarized number reported without its λ | Every reported reward (not recovery %, which doesn't need λ) must state its `lambda_vector` next to it — `shared/benchmark_harness.EpisodeResult.summary()`/`.as_dict()` do this automatically; don't hand-report a bare reward number. |
| Cherry-picked seed | See §4 (seeds and reporting). |

## 3. Fixed vs. reported-as-provisional

**Fixed today** (don't re-litigate per report):
- `f₁..f₄` formulas — `env/simulator.py`, matches `docs/problem_spec.md` §3-4.
- Hard-constraint enforcement — masking, not penalty (joint spec §3.2).
- `env/mock_telemetry_adapter.py`'s 4 tenant scenarios, schema-validated.

**Provisional — flag in every report until Yunqi reviews:**
- `D`'s exact value when `scenario_config.facility.stranded_denominator_kwh`
  is `null` (all 4 adapter-built tenant scenarios currently leave it null).
  `compute_D()`'s fallback (facility headroom + host fragmentation estimate)
  is *a* reasonable definition, not *the* agreed one — joint spec §3.1 says
  "harness owns the computation," and that's currently a placeholder
  implementation, not a reviewed one. Recovery % numbers reported before
  Yunqi signs off on `D` should be captioned "provisional D" or similar.
- The host inventory and arrival process (`env/mock_telemetry_adapter.py`
  `_default_host_inventory`/`_default_arrivals`) are `[approximated]`,
  owner James, per joint spec Q2/Q3 — not sourced from the mock telemetry.
  A number is only as meaningful as this synthetic setup is realistic.
- No oracle yet (§5 below) — until it exists, "% of oracle" can't be
  reported at all, only raw recovery %.

## 4. Seeds and reporting (operationalizing joint spec §5.5)

- **Minimum 10 seeds per (policy, scenario, λ) triple** before reporting a
  headline number. Report mean ± std, not a single run. (`optimization/
  baselines/run_baselines.py --episodes N` and `rl/agents/train.py`'s
  per-episode seeding both support this — episode/run seeds are derived
  deterministically from a single `--seed`, so "seed 0, 10 episodes" is
  fully reproducible, not "whatever the RNG happened to do.")
- Training runs (RL) additionally report the **last 10% of episodes**
  separately from the first 10% (`rl/agents/train.py`'s `summary.json`
  already does this) — a policy still learning shouldn't be scored on its
  early, near-random episodes.
- Every reported number traces back to a `rl/results/` or `optimization/
  results/` run directory (`shared/experiment_tracking.ExperimentTracker`):
  `config.json` (exact hyperparameters, git commit, scenario_id, seed),
  `metrics.jsonl` (per-episode), `summary.json`. A number in a memo or
  slide without a linked run directory doesn't count as reported yet.
- Cross-policy comparisons (e.g. "greedy beats random") must use the
  **same `scenario_id` and the same `D`** — `run_many()` in the shared
  harness already enforces the same scenario_config across policies in one
  call; don't hand-assemble a comparison from separately-run scripts with
  different scenario configs.

## 5. Oracle bound (joint spec §5.2) — not yet implemented

Per joint spec: solve the Layer A MILP with perfect foresight of the
realized episode (arrivals known in advance), giving an upper bound on
recovery for that specific episode. RL/baseline performance is then reported
as **% of oracle**, not raw recovery % alone (raw recovery % conflates "the
policy is bad" with "this scenario just doesn't have much recoverable
capacity available").

This isn't built yet — it's Yunqi's optimization-track deliverable
(`optimization/`), not something the env/harness can produce on its own.
`shared/benchmark_harness.oracle_relative_pct()` already has the plumbing
to consume it once available (takes a policy's `EpisodeResult` and an
`oracle_recovery_pct` and returns the ratio) — it's a placeholder pending
that solve. Until then, **do not report "% of oracle" numbers**; report raw
recovery % and note the oracle is pending.

## 6. Certainty-equivalence check (joint spec §5.4) — how to actually run it

1. Build the toy instance: `arrivals_enabled=False` (already what
   `env/scenarios/alcf_toy.json` and `mock_telemetry_adapter.build_scenario_config(..., arrivals_enabled=False)` give you) — fixed windows (`z_fixed=True`, already the adapter default).
2. Run `env/simulator.py` end to end holding levers static (see
   `tests/test_simulator.py::test_toy_instance_certainty_equivalence_zero_violations_zero_recovery`)
   — this is the "MDP turns into MILP" special case (joint spec §0).
3. Yunqi solves the same `scenario_config` as a pure LP (§5.4's other half —
   not yet run; needs her solver).
4. Compare: episode returns should match the LP optimum within tolerance.
   **This comparison hasn't been run yet** (step 3 doesn't exist). Until it
   is, the toy-instance test only confirms the env side is internally
   consistent (zero violations, zero recovery since nothing arrives to
   dispatch), not that it agrees with the LP.

## 7. Baseline / agent comparison table (what a report should look like)

A results write-up should be a table like:

| Policy | Scenario | Seeds | Recovery % (mean ± std) | % of oracle | Violations | λ | Run dir |
|---|---|---|---|---|---|---|---|
| random | alcf_stochastic | 10 | 0.57% ± 0.03 | pending oracle | 0/10 | (1.0, 0.01, 0.01, 0.01) | `optimization/results/...` |
| greedy | alcf_stochastic | 10 | 0.61% ± 0.02 | pending oracle | 0/10 | (1.0, 0.01, 0.01, 0.01) | `optimization/results/...` |
| actor_critic_v0 | alcf_stochastic | 10 | TBD | pending oracle | 0/10 | (1.0, 0.01, 0.01, 0.01) | `rl/results/...` |

(The random/greedy numbers above are illustrative, from a single seed=1
`optimization/baselines/run_baselines.py` run, not yet the required 10-seed
aggregate — flagging that gap is itself an example of §4's rule.)

## 8. Open questions for Yunqi

1. Sign off on `compute_D()`'s fallback formula, or provide the "real" `D`
   per scenario (joint spec §3.1: "harness owns the computation").
2. Timeline for the perfect-foresight MILP oracle solve (§5).
3. Timeline for the pure-LP solve needed to close the certainty-equivalence
   loop (§6 step 3 above).
4. Whether `env/mock_telemetry_adapter.py`'s synthetic host inventory/
   arrivals (Q2/Q3) need her review too, or stay solely James's call per the
   joint spec's ownership table.
