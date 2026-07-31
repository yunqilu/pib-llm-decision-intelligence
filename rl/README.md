# RL track

MDP formulation, policy-gradient / actor-critic agents. Owner: James.

- `gym_wrapper.py` — thin adapter exposing `env/simulator.py`'s canonical
  JSON contract as fixed-shape `gymnasium.spaces` (`Dict`/`MultiDiscrete`),
  for RL libraries that expect that surface. See
  `docs/decisions/2026-07-28-env-api-contract-v0.md` for why this is a
  wrapper, not the source of truth — the JSON contract is canonical.
- `agents/actor_critic.py` + `agents/train.py` — PyTorch actor-critic agent
  (Layer B dispatch only) with full experiment tracking. Run:
  `python rl/agents/train.py --tenant alcf --episodes 300 --seed 0`.
- `results/` — one directory per training run (`shared/experiment_tracking.py`),
  each with `config.json`, `metrics.jsonl`, `summary.json`, `model.pt`.

See `docs/benchmark_protocol.md` for how results from this track get
reported and compared against the baselines and (once it exists) the
optimization-track oracle.

Requires `gymnasium` and `torch` (optional dependencies, only for this
subpackage): `pip install gymnasium torch`.
