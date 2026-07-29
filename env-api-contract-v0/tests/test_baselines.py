"""tests/test_baselines.py — shared/benchmark_harness.py and
optimization/baselines/policies.py tests.

Run: pytest tests/ -v
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.mock_telemetry_adapter import build_scenario_config  # noqa: E402
from env.simulator import DecisionIntelligenceEnv  # noqa: E402
from optimization.baselines.policies import GreedyPolicy, RandomPolicy  # noqa: E402
from shared.benchmark_harness import compute_D, run_episode, run_many, scalarize  # noqa: E402

LAMBDA = (1.0, 0.01, 0.01, 0.01)


def _stochastic_env(seed):
    cfg = build_scenario_config("alcf", scenario_id="baseline_test", arrivals_enabled=True, seed=seed)
    return DecisionIntelligenceEnv(cfg)


def test_random_and_greedy_run_full_episode_no_violations():
    for policy in (RandomPolicy(seed=0), GreedyPolicy()):
        env = _stochastic_env(seed=1)
        result = run_episode(env, policy, policy_name=type(policy).__name__, lambda_vector=LAMBDA)
        assert result.steps == env.T
        assert result.violations == []


def test_greedy_recovers_at_least_as_much_as_random_on_average():
    """Not a hard per-seed guarantee (both are stochastic/heuristic), but
    greedy's best-fit + SLA-first + largest-first ordering should beat
    uniform random placement on average recovery % over enough episodes."""
    n = 20
    random_results = [
        run_episode(_stochastic_env(seed=s), RandomPolicy(seed=s), "random", LAMBDA)
        for s in range(n)
    ]
    greedy_results = [
        run_episode(_stochastic_env(seed=s), GreedyPolicy(), "greedy", LAMBDA)
        for s in range(n)
    ]
    avg_random = sum(r.recovery_pct for r in random_results) / n
    avg_greedy = sum(r.recovery_pct for r in greedy_results) / n
    assert avg_greedy >= avg_random


def test_compute_D_uses_declared_value_when_present():
    cfg = build_scenario_config("alcf", scenario_id="d_test")
    cfg["facility"]["stranded_denominator_kwh"] = 12345.0
    assert compute_D(cfg) == 12345.0


def test_compute_D_positive_fallback_when_null():
    cfg = build_scenario_config("alcf", scenario_id="d_test2")
    assert cfg["facility"]["stranded_denominator_kwh"] is None
    assert compute_D(cfg) > 0


def test_scalarize_matches_manual_formula():
    info = {"f1_recovery_kwh": 10.0, "f2_cost_usd": 2.0, "f3_co2_kg": 3.0, "f4_risk": 1.0}
    lam = (1.0, 0.5, 0.25, 2.0)
    expected = 1.0 * 10.0 - 0.5 * 2.0 - 0.25 * 3.0 - 2.0 * 1.0
    assert scalarize(info, lam) == pytest.approx(expected)


def test_run_many_returns_one_result_per_policy_per_episode():
    def factory():
        return _stochastic_env(seed=2)

    results = run_many(factory, {"random": RandomPolicy(seed=1), "greedy": GreedyPolicy()}, LAMBDA, n_episodes=3)
    assert len(results) == 6
    assert sum(1 for r in results if r.policy_name == "random") == 3
    assert sum(1 for r in results if r.policy_name == "greedy") == 3


def test_toy_scenario_zero_recovery_for_both_baselines():
    """Arrivals-off toy instance: nothing ever arrives to dispatch, so both
    baselines should report exactly 0 recovery (there is nothing to place),
    matching the certainty-equivalence check in test_simulator.py."""
    for policy in (RandomPolicy(seed=0), GreedyPolicy()):
        cfg = build_scenario_config("alcf", scenario_id="toy_test", arrivals_enabled=False)
        env = DecisionIntelligenceEnv(cfg)
        result = run_episode(env, policy, type(policy).__name__, LAMBDA)
        assert result.f1_recovery_kwh == 0.0
        assert result.recovery_pct == 0.0
        assert result.violations == []
