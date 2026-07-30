"""
rl/gym_wrapper.py — thin Gymnasium-compatible adapter around
env/simulator.DecisionIntelligenceEnv, for RL libraries that expect fixed-
shape `gymnasium.spaces` rather than the canonical variable-length JSON
contract (see docs/decisions/2026-07-28-env-api-contract-v0.md for why the
JSON contract, not this wrapper, is the source of truth).

This wrapper pads/truncates the host queue to `max_queue_len` and maps
integer host indices back to `host_id` strings. Translation lives entirely
here; env/simulator.py never sees or produces numpy.

Usage:
    from env.mock_telemetry_adapter import build_scenario_config
    from rl.gym_wrapper import GymDispatchEnv

    cfg = build_scenario_config("alcf", scenario_id="rl_v0", arrivals_enabled=True)
    env = GymDispatchEnv(cfg, max_queue_len=32)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from env.simulator import DecisionIntelligenceEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except ImportError:  # pragma: no cover - gymnasium is an optional dependency for rl/
    gym = None
    spaces = None
    _HAVE_GYM = False


_Base = gym.Env if _HAVE_GYM else object


class GymDispatchEnv(_Base):
    """Fixed-shape wrapper. If gymnasium is installed, this class is
    additionally registered as a gymnasium.Env subclass at import time
    (see bottom of file); the reset()/step() surface is identical either
    way, so most RL loops work regardless of whether gymnasium is present."""

    metadata = {"render_modes": []}

    def __init__(self, scenario_config: Dict[str, Any], max_queue_len: int = 32, lambda_weights=(1.0, 0.0, 0.0, 0.0)):
        self._env = DecisionIntelligenceEnv(scenario_config)
        self.max_queue_len = max_queue_len
        self.lambda_weights = np.array(lambda_weights, dtype=np.float32)

        self.N = self._env.N
        self.M = self._env.M
        self._host_ids: List[str] = [h["host_id"] for h in self._env.inventory]
        self._queue_task_ids: List[Optional[str]] = [None] * self.max_queue_len

        if _HAVE_GYM:
            self.observation_space = spaces.Dict({
                "H_t": spaces.Box(low=0.0, high=1.0, shape=(self.N, self.M), dtype=np.float32),
                "Q_t": spaces.Box(low=0.0, high=1e4, shape=(self.max_queue_len, self.M), dtype=np.float32),
                "Q_mask": spaces.Box(low=0, high=1, shape=(self.max_queue_len,), dtype=np.int8),
                "layer_A_envelope": spaces.Dict({
                    "r_star_t": spaces.Box(low=0.0, high=1e7, shape=(1,), dtype=np.float32),
                    "budget_remaining_kw": spaces.Box(low=0.0, high=1e7, shape=(1,), dtype=np.float32),
                    "is_migration_allowed": spaces.Discrete(2),
                }),
            })
            self.action_space = spaces.Dict({
                # index N == "defer" (don't place this queue slot this step)
                "P_t": spaces.MultiDiscrete([self.N + 1] * self.max_queue_len),
                # index N == "stand still" for each host (no migration out)
                "M_t": spaces.MultiDiscrete([self.N + 1] * self.N),
            })

    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            # gymnasium.Env.reset(seed=...) convention: reseed this episode's
            # RNG. Previously silently ignored here -- important for
            # experiment reproducibility (shared/experiment_tracking.py),
            # since arrivals are stochastic and otherwise every reset() just
            # continues drawing from whatever RNG state env/simulator.py's
            # __init__ started with.
            self._env._rng = np.random.default_rng(seed)
        obs, info = self._env.reset()
        return self._encode_obs(obs), self._encode_info(info)

    def step(self, action: Dict[str, np.ndarray]):
        raw_action = self._decode_action(action)
        obs, _reward, terminated, truncated, info = self._env.step(raw_action)
        gym_obs = self._encode_obs(obs)
        gym_info = self._encode_info(info)
        reward = float(
            self.lambda_weights[0] * info["f1_recovery_kwh"]
            - self.lambda_weights[1] * info["f2_cost_usd"]
            - self.lambda_weights[2] * info["f3_co2_kg"]
            - self.lambda_weights[3] * info["f4_risk"]
        )
        return gym_obs, reward, terminated, truncated, gym_info

    # ------------------------------------------------------------------ #
    # JSON contract <-> fixed-shape numpy translation
    # ------------------------------------------------------------------ #

    def _encode_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        H_t = np.array(obs["hosts"]["utilization"], dtype=np.float32)
        queue = obs["hosts"]["queue"][: self.max_queue_len]
        Q_t = np.zeros((self.max_queue_len, self.M), dtype=np.float32)
        Q_mask = np.zeros((self.max_queue_len,), dtype=np.int8)
        self._queue_task_ids = [None] * self.max_queue_len
        for k, task in enumerate(queue):
            Q_t[k] = [task["demand"].get(r, 0.0) for r in self._env.resources]
            Q_mask[k] = 1
            self._queue_task_ids[k] = task["task_id"]
        # Tasks beyond max_queue_len aren't dropped from the underlying env
        # (env.queue keeps them; they reappear next observation if still
        # unplaced) -- only from *this step's* fixed-size view here.
        return {
            "H_t": H_t,
            "Q_t": Q_t,
            "Q_mask": Q_mask,
            "layer_A_envelope": {
                "r_star_t": np.array([self._env.r_star_t], dtype=np.float32),
                "budget_remaining_kw": np.array([obs["facility"]["budget_remaining_kw"]], dtype=np.float32),
                "is_migration_allowed": int(obs["facility"]["in_window"]),
            },
        }

    def _encode_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        mask = info["action_mask"]
        placement_mask = np.zeros((self.max_queue_len, self.N + 1), dtype=np.int8)
        placement_mask[:, self.N] = 1  # "defer" always legal
        for k, tid in enumerate(self._queue_task_ids):
            if tid is None:
                continue
            allowed_hosts = mask["placement_allowed"].get(tid, set())
            for host_id in allowed_hosts:
                if host_id in self._host_ids:
                    placement_mask[k, self._host_ids.index(host_id)] = 1
        migration_mask = np.zeros((self.N, self.N + 1), dtype=np.int8)
        migration_mask[:, self.N] = 1  # "stand still" always legal
        if mask["migration_allowed"]:
            migration_mask[:, : self.N] = 1
            for i in range(self.N):
                migration_mask[i, i] = 0  # can't migrate a host to itself

        return {
            **{k: v for k, v in info.items() if k != "action_mask"},
            "action_mask": {"placement_mask": placement_mask, "migration_mask": migration_mask},
        }

    def _decode_action(self, action: Dict[str, np.ndarray]) -> Dict[str, Any]:
        placements = []
        P_t = np.asarray(action["P_t"])
        for k, host_idx in enumerate(P_t):
            if host_idx >= self.N:
                continue  # defer
            tid = self._queue_task_ids[k] if k < len(self._queue_task_ids) else None
            if tid is None:
                continue  # this queue slot was padding, not a real task
            placements.append({"task_id": tid, "host_id": self._host_ids[int(host_idx)]})

        migrations = []
        M_t = np.asarray(action["M_t"])
        for i, target_idx in enumerate(M_t):
            if target_idx >= self.N or int(target_idx) == i:
                continue  # stand still
            migrations.append({
                "task_id": "n/a",  # v0 migration is aggregate host-power level, see ADR follow-ups
                "from_host": self._host_ids[i],
                "to_host": self._host_ids[int(target_idx)],
            })

        # Hold current facility levers; this wrapper is Layer-B-only (dispatch).
        # A Layer-A-aware RL agent should use env/simulator.py directly, not
        # this wrapper -- see docs/decisions/2026-07-28-env-api-contract-v0.md.
        levers = self._env.levers
        return {
            "facility": {
                "admit_kw": 1.0e6,  # open budget; env still gates on physical headroom
                "curtail_kw": 0.0,
                "chiller_setpoint_c": levers["chiller_setpoint_c"],
                "pump_duty_pct": levers["pump_duty_pct"],
                "zone_setpoint_c": levers["zone_setpoint_c"],
                "deferrable_share_pct": levers["deferrable_share_pct"],
            },
            "placements": placements,
            "migrations": migrations,
        }
