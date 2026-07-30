# RL track

MDP formulation, policy-gradient / actor-critic agents. Owner: James.

- `gym_wrapper.py` — thin adapter exposing `env/simulator.py`'s canonical
  JSON contract as fixed-shape `gymnasium.spaces` (`Dict`/`MultiDiscrete`),
  for RL libraries that expect that surface. See
  `docs/decisions/2026-07-28-env-api-contract-v0.md` for why this is a
  wrapper, not the source of truth — the JSON contract is canonical.
- `agents/` — policy-gradient / actor-critic implementations go here
  (scaffolded, not yet implemented).

Requires `gymnasium` (optional dependency, only for this subpackage):
`pip install gymnasium`.
