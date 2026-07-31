"""tests/test_actor_critic.py — rl/agents/actor_critic.py and
shared/experiment_tracking.py tests.

Requires torch; auto-skips if not installed (torch is only needed for
rl/agents/, not the rest of the repo).
"""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.mock_telemetry_adapter import build_scenario_config  # noqa: E402

torch = pytest.importorskip("torch")
gymnasium = pytest.importorskip("gymnasium")

from rl.agents.actor_critic import ActorCriticNet, compute_returns, run_training_episode  # noqa: E402
from rl.gym_wrapper import GymDispatchEnv  # noqa: E402
from shared.experiment_tracking import ExperimentTracker, RunConfig, get_git_commit, set_seed  # noqa: E402


def _make_env(seed=1, max_queue_len=8):
    cfg = build_scenario_config("alcf", scenario_id="ac_test", arrivals_enabled=True, seed=seed)
    return GymDispatchEnv(cfg, max_queue_len=max_queue_len, lambda_weights=(1.0, 0.01, 0.01, 0.01))


def test_compute_returns_matches_manual_discounting():
    rewards = [1.0, 2.0, 3.0]
    gamma = 0.5
    returns = compute_returns(rewards, gamma)
    expected2 = 3.0
    expected1 = 2.0 + gamma * expected2
    expected0 = 1.0 + gamma * expected1
    assert returns.tolist() == pytest.approx([expected0, expected1, expected2])


def test_act_respects_mask_placement_choices_are_legal():
    torch.manual_seed(0)
    env = _make_env()
    obs, info = env.reset()
    net = ActorCriticNet(N=env.N, M=env.M, max_queue_len=8, hidden_dim=32)
    action, log_prob, entropy, value = net.act(obs, info["action_mask"])
    pm = info["action_mask"]["placement_mask"]
    for k, host_idx in enumerate(action["P_t"]):
        assert pm[k, host_idx] == 1, f"slot {k} chose host_idx {host_idx}, not in mask row {pm[k]}"
    mm = info["action_mask"]["migration_mask"]
    for i, target_idx in enumerate(action["M_t"]):
        assert mm[i, target_idx] == 1


def test_act_output_shapes():
    torch.manual_seed(0)
    env = _make_env(max_queue_len=12)
    obs, info = env.reset()
    net = ActorCriticNet(N=env.N, M=env.M, max_queue_len=12, hidden_dim=32)
    action, log_prob, entropy, value = net.act(obs, info["action_mask"])
    assert action["P_t"].shape == (12,)
    assert action["M_t"].shape == (env.N,)
    assert log_prob.dim() == 0  # scalar
    assert value.dim() == 0


def test_run_training_episode_produces_finite_loss_and_zero_violations():
    torch.manual_seed(0)
    env = _make_env()
    net = ActorCriticNet(N=env.N, M=env.M, max_queue_len=8, hidden_dim=32)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    metrics = run_training_episode(env, net, optimizer, gamma=0.99, seed=7)
    assert metrics["steps"] == 24
    assert metrics["violations"] == 0
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["reward_total"])


def test_gradients_flow_to_all_heads():
    """Regression guard: a masking or indexing bug that accidentally detaches
    a head (e.g. always choosing the 'defer'/'stand still' fallback) would
    silently produce zero gradient on that head forever. Confirm every head
    gets a nonzero gradient after one update."""
    torch.manual_seed(0)
    env = _make_env()
    net = ActorCriticNet(N=env.N, M=env.M, max_queue_len=8, hidden_dim=32)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    run_training_episode(env, net, optimizer, gamma=0.99, seed=7)

    for name, head in [("placement_head", net.placement_head), ("migration_head", net.migration_head),
                        ("value_head", net.value_head)]:
        grad_norm = sum(p.grad.norm().item() for p in head.parameters() if p.grad is not None)
        assert grad_norm > 0, f"{name} got zero gradient"


def test_seeded_env_reset_is_reproducible():
    """rl/gym_wrapper.py's reset(seed=...) fix: same seed -> same arrivals."""
    env1 = _make_env(seed=999)
    obs1, _ = env1.reset(seed=123)
    env2 = _make_env(seed=999)
    obs2, _ = env2.reset(seed=123)
    assert obs1["Q_t"].tolist() == obs2["Q_t"].tolist()
    assert obs1["Q_mask"].tolist() == obs2["Q_mask"].tolist()


def test_set_seed_reproducible_network_init_and_sampling():
    set_seed(42)
    net1 = ActorCriticNet(N=8, M=4, max_queue_len=8, hidden_dim=16)
    set_seed(42)
    net2 = ActorCriticNet(N=8, M=4, max_queue_len=8, hidden_dim=16)
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        assert torch.equal(p1, p2)


def test_get_git_commit_returns_string_or_none():
    commit = get_git_commit(repo_dir=str(pathlib.Path(__file__).resolve().parents[1]))
    assert commit is None or isinstance(commit, str)


def test_experiment_tracker_writes_expected_files(tmp_path):
    config = RunConfig(run_name="test_run", tenant="alcf", seed=1, lambda_vector=(1.0, 0.0, 0.0, 0.0), n_episodes=2)
    tracker = ExperimentTracker(base_dir=str(tmp_path), config=config)
    tracker.log_episode(0, {"reward_total": 1.0, "recovery_pct": 0.1})
    tracker.log_episode(1, {"reward_total": 2.0, "recovery_pct": 0.2})
    tracker.finalize({"final_recovery_pct": 0.2})

    assert (tracker.run_dir / "config.json").exists()
    assert (tracker.run_dir / "metrics.jsonl").exists()
    assert (tracker.run_dir / "summary.json").exists()

    lines = (tracker.run_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["episode"] == 0
    assert row0["reward_total"] == 1.0

    cfg_on_disk = json.loads((tracker.run_dir / "config.json").read_text())
    assert cfg_on_disk["config"]["run_name"] == "test_run"
    assert cfg_on_disk["config"]["seed"] == 1
    assert "python_version" in cfg_on_disk


def test_experiment_tracker_run_dir_unique_per_seed(tmp_path):
    c1 = RunConfig(run_name="r", tenant="alcf", seed=1, lambda_vector=(1, 0, 0, 0))
    c2 = RunConfig(run_name="r", tenant="alcf", seed=2, lambda_vector=(1, 0, 0, 0))
    t1 = ExperimentTracker(base_dir=str(tmp_path), config=c1)
    t2 = ExperimentTracker(base_dir=str(tmp_path), config=c2)
    assert t1.run_dir != t2.run_dir
    t1.finalize({})
    t2.finalize({})
