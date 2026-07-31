# pib-llm-decision-intelligence

Shared research repo for recovering stranded data-center capacity (power /
cooling / thermal constrained AI compute) via two complementary approaches,
under PhysaFlow NOI's decision-intelligence layer.

- **Optimization track** (`optimization/`) — constrained multi-objective
  optimization (greedy baselines, LP/MILP). Owner: Yunqi Lu.
- **RL track** (`rl/`) — MDP formulation, policy-gradient / actor-critic
  agents. Owner: James.

Both tracks consume the **same environment** (`env/`) and are evaluated on
the **same benchmark harness** (`shared/benchmark_harness.py`), so results
are directly comparable — one shared problem, not two separate research
tracks.

## Start here
- [`docs/problem_spec.md`](docs/problem_spec.md) — the single shared problem
  formalization (state, action, objective, constraints).
- [`docs/data_schema.md`](docs/data_schema.md) — what the environment exposes
  to both solvers.
- [`docs/decisions/`](docs/decisions/) — short decision records (ADR-style),
  e.g. the initial scope agreement with James.

## Scope boundary
No access to, inspection of, or reasoning about the proprietary Blackbox /
Core Vault / BioCore engine. Research/prototyping only, against PhysaFlow's
standard mock interfaces. See the Role 16 brief.

## Status
`env/simulator.py` (the shared env, v0.2 data contract), `shared/benchmark_harness.py`,
and two Layer B dispatch baselines (`optimization/baselines/policies.py`:
random, greedy) are implemented and tested (`tests/`, `pytest tests/ -v`,
40 tests). All six `docs/problem_spec.md` §4 constraints (power cap, thermal
redline, SLA floor, action windows, N+1 reserve, ramp) are enforced per the
joint spec's masking/soft split — see
`docs/decisions/2026-07-28-env-api-contract-v0.md` for the full contract
decision record and the 2026-07-29 spec-conformance follow-up.
`rl/gym_wrapper.py` adapts the env to fixed-shape Gym spaces for RL
training; `rl/agents/` has a working PyTorch actor-critic agent with full
experiment tracking (`shared/experiment_tracking.py`: seeds, configs,
per-episode logs). See `docs/benchmark_protocol.md` (draft, pending Yunqi
sign-off on `D` and the oracle) for how results get measured and reported.
Pending: Yunqi's review of the env implementation against her
MILP-side formulation, and a real LP/MILP oracle solve to report baseline
recovery % as % of oracle (data_schema.md §5).
