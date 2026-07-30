# Agents

Policy-gradient / actor-critic implementations go here.

- `actor_critic.py` — PyTorch actor-critic net + rollout/update logic (Layer B dispatch only; see docs/decisions/2026-07-28-env-api-contract-v0.md).
- `train.py` — CLI training entry point with full experiment tracking (shared/experiment_tracking.py). `python rl/agents/train.py --help`.
