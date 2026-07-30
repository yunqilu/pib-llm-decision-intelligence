"""
rl/agents/actor_critic.py — PyTorch actor-critic agent for
env.simulator.DecisionIntelligenceEnv, via rl/gym_wrapper.py's fixed-shape
translation.

Design notes:
  - Layer B (dispatch) only: the actor produces `P_t` (per-queue-slot host
    choice) and `M_t` (per-host migration target). Facility levers are held
    (see rl/gym_wrapper.py's _decode_action) -- a Layer-A-aware agent is a
    separate, later piece of work (see docs/decisions/2026-07-28-env-api-
    contract-v0.md follow-ups).
  - The policy is factorized: one independent categorical distribution per
    queue slot (over N+1 host choices, last = defer) and one per host (over
    N+1 migration targets, last = stand still). This is the standard way to
    handle a MultiDiscrete action space and matches gymnasium's own
    convention for it.
  - Masking (info["action_mask"]) is applied as -inf on logits before
    softmax, never as a post-hoc penalty -- consistent with the env's own
    "masked, not penalized" contract (joint spec §3.2). A masked-out choice
    has exactly zero probability, not a discouraged one.
  - Padding queue slots (Q_mask[k] == 0, i.e. fewer real tasks than
    max_queue_len this step) are excluded from the log-prob sum entirely --
    they aren't real decisions and shouldn't contribute gradient.
  - Full-episode Monte Carlo returns (episodes are only T=24 steps, so
    truncated n-step/bootstrap machinery buys little here) with a learned
    value baseline: standard REINFORCE-with-baseline / A2C, updated once per
    episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNet(nn.Module):
    def __init__(self, N: int, M: int, max_queue_len: int, hidden_dim: int = 256):
        super().__init__()
        self.N = N
        self.M = M
        self.max_queue_len = max_queue_len

        in_dim = N * M + max_queue_len * M + max_queue_len + 2 + 1  # H_t, Q_t, Q_mask, envelope(2), is_migration(1)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.placement_head = nn.Linear(hidden_dim, max_queue_len * (N + 1))
        self.migration_head = nn.Linear(hidden_dim, N * (N + 1))
        self.value_head = nn.Linear(hidden_dim, 1)

    def _featurize(self, obs: Dict[str, Any]) -> torch.Tensor:
        H = torch.as_tensor(obs["H_t"], dtype=torch.float32).flatten()
        Q = torch.as_tensor(obs["Q_t"], dtype=torch.float32).flatten()
        Qm = torch.as_tensor(obs["Q_mask"], dtype=torch.float32)
        env = obs["layer_A_envelope"]
        # log1p-normalize the two facility-scale fields: r_star_t /
        # budget_remaining_kw can be ~1e6 (the wrapper always proposes an
        # effectively unbounded admission budget -- see rl/gym_wrapper.py
        # _decode_action), while every other feature here is O(1)-O(30).
        # Feeding that raw into an untrained linear layer produces huge,
        # unstable initial value/logit estimates.
        envelope = torch.as_tensor(
            [
                np.log1p(max(0.0, float(env["r_star_t"][0]))),
                np.log1p(max(0.0, float(env["budget_remaining_kw"][0]))),
                float(env["is_migration_allowed"]),
            ],
            dtype=torch.float32,
        )
        return torch.cat([H, Q, Qm, envelope], dim=0)

    def forward(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._featurize(obs)
        h = self.trunk(x)
        placement_logits = self.placement_head(h).view(self.max_queue_len, self.N + 1)
        migration_logits = self.migration_head(h).view(self.N, self.N + 1)
        value = self.value_head(h).squeeze(-1)
        return placement_logits, migration_logits, value

    def act(self, obs: Dict[str, Any], mask: Dict[str, np.ndarray], greedy: bool = False):
        """Sample (or, if greedy, argmax) an action under the mask. Returns
        (action_dict_for_env, log_prob_sum, entropy_sum, value). Masked-out
        entries get -inf logits, so they have exactly 0 sampling probability
        -- not just low probability."""
        placement_logits, migration_logits, value = self.forward(obs)

        pm = torch.as_tensor(mask["placement_mask"], dtype=torch.bool)   # (max_queue_len, N+1)
        mm = torch.as_tensor(mask["migration_mask"], dtype=torch.bool)   # (N, N+1)
        placement_logits = placement_logits.masked_fill(~pm, float("-inf"))
        migration_logits = migration_logits.masked_fill(~mm, float("-inf"))

        q_valid = torch.as_tensor(obs["Q_mask"], dtype=torch.bool)  # real (non-padding) queue slots

        P_t, log_prob_p, ent_p = self._sample_rows(placement_logits, greedy)
        M_t, log_prob_m, ent_m = self._sample_rows(migration_logits, greedy)

        log_prob_sum = (log_prob_p * q_valid).sum() + log_prob_m.sum()
        entropy_sum = (ent_p * q_valid).sum() + ent_m.sum()

        action = {"P_t": P_t.numpy().astype(np.int64), "M_t": M_t.numpy().astype(np.int64)}
        return action, log_prob_sum, entropy_sum, value

    @staticmethod
    def _sample_rows(logits: torch.Tensor, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """logits: (rows, choices). Independent categorical per row (fully
        masked rows -- e.g. an empty queue slot with an all -inf row --
        fall back to uniform over the row so sampling doesn't NaN; that
        row's log-prob is excluded from the loss by the caller anyway)."""
        all_masked = torch.isinf(logits).all(dim=-1)
        safe_logits = logits.clone()
        safe_logits[all_masked] = 0.0  # uniform fallback, harmless: excluded from loss upstream
        dist = torch.distributions.Categorical(logits=safe_logits)
        actions = dist.mode if greedy else dist.sample()
        return actions, dist.log_prob(actions), dist.entropy()


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    returns = torch.zeros(len(rewards), dtype=torch.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def run_training_episode(
    env,
    net: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    seed: int = None,
) -> Dict[str, float]:
    """One on-policy actor-critic update: rollout a full episode, then a
    single backward pass over the whole episode (Monte Carlo returns +
    learned baseline). Returns episode-level metrics for logging."""
    obs, info = env.reset(seed=seed)
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[float] = []
    f1 = f2 = f3 = f4 = 0.0
    violations: List[str] = []

    terminated = False
    while not terminated:
        action, log_prob, entropy, value = net.act(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        rewards.append(reward)
        f1 += info["f1_recovery_kwh"]
        f2 += info["f2_cost_usd"]
        f3 += info["f3_co2_kg"]
        f4 += info["f4_risk"]
        violations.extend(info["violations"])

    returns = compute_returns(rewards, gamma)
    values_t = torch.stack(values)
    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)

    advantages = (returns - values_t).detach()
    # normalize advantages for training stability; skip if degenerate (near-constant rewards)
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = -(log_probs_t * advantages).mean()
    critic_loss = F.mse_loss(values_t, returns)
    entropy_bonus = entropies_t.mean()
    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "reward_total": float(sum(rewards)),
        "loss": float(loss.item()),
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy_bonus.item()),
        "f1_recovery_kwh": f1,
        "f2_cost_usd": f2,
        "f3_co2_kg": f3,
        "f4_risk": f4,
        "steps": len(rewards),
        "violations": len(violations),
    }
