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
Layout scaffolded — problem spec, data schema, and env implementation are
still DRAFT pending joint sign-off between Yunqi and James.
