"""tests/test_gym_wrapper.py — rl/gym_wrapper.py translation-layer tests.

Requires gymnasium (optional dependency for rl/); tests auto-skip if it's
not installed, since env/simulator.py and everything else in this repo does
not depend on it (docs/decisions/2026-07-28-env-api-contract-v0.md).
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.mock_telemetry_adapter import build_scenario_config  # noqa: E402

gymnasium = pytest.importorskip("gymnasium")

from rl.gym_wrapper import GymDispatchEnv  # noqa: E402


def _defer_all_action(env: GymDispatchEnv):
    return {
        "P_t": np.full(env.max_queue_len, env.N, dtype=np.int64),
        "M_t": np.full(env.N, env.N, dtype=np.int64),
    }


def test_is_gymnasium_env_subclass():
    cfg = build_scenario_config("alcf", scenario_id="gym_test1", arrivals_enabled=False)
    env = GymDispatchEnv(cfg, max_queue_len=8)
    assert isinstance(env, gymnasium.Env)


def test_defer_all_full_episode_zero_violations():
    cfg = build_scenario_config("alcf", scenario_id="gym_test2", arrivals_enabled=True, seed=11)
    env = GymDispatchEnv(cfg, max_queue_len=16)
    obs, info = env.reset()
    total_violations = 0
    for _ in range(24):
        obs, reward, terminated, truncated, info = env.step(_defer_all_action(env))
        total_violations += len(info["violations"])
        if terminated:
            break
    assert terminated
    assert total_violations == 0


def test_observation_shapes_match_declared_spaces():
    cfg = build_scenario_config("alcf", scenario_id="gym_test3", arrivals_enabled=True, seed=2)
    env = GymDispatchEnv(cfg, max_queue_len=12)
    obs, info = env.reset()
    assert obs["H_t"].shape == (env.N, env.M)
    assert obs["Q_t"].shape == (12, env.M)
    assert obs["Q_mask"].shape == (12,)
    assert info["action_mask"]["placement_mask"].shape == (12, env.N + 1)
    assert info["action_mask"]["migration_mask"].shape == (env.N, env.N + 1)


def test_placing_on_masked_host_index_is_ignored_not_crashed():
    """A policy that ignores placement_mask and picks an arbitrary host index
    should not crash the wrapper or the underlying env -- it just doesn't
    get placed (masked, joint spec §3.2), same guarantee as the raw JSON
    contract."""
    cfg = build_scenario_config("alcf", scenario_id="gym_test4", arrivals_enabled=True, seed=9)
    env = GymDispatchEnv(cfg, max_queue_len=16)
    obs, info = env.reset()
    action = {
        "P_t": np.zeros(16, dtype=np.int64),  # blindly try host 0 for every slot, mask or no mask
        "M_t": np.full(env.N, env.N, dtype=np.int64),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["violations"] == []


def test_defer_action_never_admits_anything():
    cfg = build_scenario_config("alcf", scenario_id="gym_test5", arrivals_enabled=True, seed=3)
    env = GymDispatchEnv(cfg, max_queue_len=16)
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(24):
        obs, reward, terminated, truncated, info = env.step(_defer_all_action(env))
        total_reward += reward
        if terminated:
            break
    assert info["f1_recovery_kwh"] == 0.0
