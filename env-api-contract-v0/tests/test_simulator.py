"""tests/test_simulator.py — env/simulator.py contract tests.

Run: pytest tests/ -v
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.mock_telemetry_adapter import TENANT_TELEMETRY, build_scenario_config  # noqa: E402
from env.simulator import DecisionIntelligenceEnv, SimulatorError  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def alcf_toy_config():
    with open(REPO_ROOT / "env" / "scenarios" / "alcf_toy.json") as f:
        return json.load(f)


def test_alcf_toy_schema_valid():
    import jsonschema
    from env.schemas.validate import cross_field_errors

    schema = json.loads((REPO_ROOT / "env" / "schemas" / "scenario_config.schema.json").read_text())
    cfg = json.loads((REPO_ROOT / "env" / "scenarios" / "alcf_toy.json").read_text())
    jsonschema.validate(cfg, schema)
    assert cross_field_errors(cfg) == []


@pytest.mark.parametrize("tenant", sorted(TENANT_TELEMETRY))
def test_adapter_scenarios_schema_valid(tenant):
    import jsonschema
    from env.schemas.validate import cross_field_errors

    schema = json.loads((REPO_ROOT / "env" / "schemas" / "scenario_config.schema.json").read_text())
    cfg = build_scenario_config(tenant, scenario_id=f"{tenant}_test")
    jsonschema.validate(cfg, schema)
    assert cross_field_errors(cfg) == []


def test_reset_enforces_coupling_identity(alcf_toy_config):
    env = DecisionIntelligenceEnv(alcf_toy_config)
    obs, info = env.reset()
    assert abs(obs["facility"]["it_load_kw"] - alcf_toy_config["facility"]["baseline_it_load_kw"][0]) < 1e-3


def test_reset_rejects_broken_coupling(alcf_toy_config):
    bad = json.loads(json.dumps(alcf_toy_config))  # deep copy
    bad["hosts"]["initial_utilization"][0][2] = 0.0  # zero out one host's power util -> breaks coupling
    with pytest.raises(SimulatorError):
        DecisionIntelligenceEnv(bad).reset()


def test_toy_instance_certainty_equivalence_zero_violations_zero_recovery(alcf_toy_config):
    """Joint spec §5.4: toy instance (arrivals off) should run cleanly with
    no host-level dispatch activity -- there's nothing in the queue to place,
    so recovery stays 0 and, crucially, violations stay empty for the whole
    episode (a hold-levers policy on a static baseline load never approaches
    p_cap or t_max_inlet_c in this scenario)."""
    env = DecisionIntelligenceEnv(alcf_toy_config)
    obs, info = env.reset()
    assert obs["hosts"]["queue"] == []  # arrivals family "none"

    total_violations = 0
    for _ in range(alcf_toy_config["meta"]["T"]):
        levers = obs["facility"]["levers"]
        action = {"facility": {
            "admit_kw": 0.0, "curtail_kw": 0.0,
            "chiller_setpoint_c": levers["chiller_setpoint_c"],
            "pump_duty_pct": levers["pump_duty_pct"],
            "zone_setpoint_c": levers["zone_setpoint_c"],
            "deferrable_share_pct": levers["deferrable_share_pct"],
        }}
        obs, reward, terminated, truncated, info = env.step(action)
        total_violations += len(info["violations"])
        assert obs["hosts"]["queue"] == []  # arrivals stay off all episode
        if terminated:
            break

    assert terminated
    assert total_violations == 0
    assert info["recovery_pct"] == 0.0


def test_step_before_reset_raises(alcf_toy_config):
    env = DecisionIntelligenceEnv(alcf_toy_config)
    with pytest.raises(SimulatorError):
        env.step({"facility": {
            "admit_kw": 0, "curtail_kw": 0, "chiller_setpoint_c": 7,
            "pump_duty_pct": 30, "zone_setpoint_c": 20, "deferrable_share_pct": 100,
        }})


def test_step_after_termination_raises(alcf_toy_config):
    env = DecisionIntelligenceEnv(alcf_toy_config)
    obs, info = env.reset()
    action = {"facility": {
        "admit_kw": 0.0, "curtail_kw": 0.0, "chiller_setpoint_c": 7.0,
        "pump_duty_pct": 30.0, "zone_setpoint_c": 20.0, "deferrable_share_pct": 100.0,
    }}
    terminated = False
    for _ in range(alcf_toy_config["meta"]["T"]):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    assert terminated
    with pytest.raises(SimulatorError):
        env.step(action)


def test_empty_action_rejected(alcf_toy_config):
    env = DecisionIntelligenceEnv(alcf_toy_config)
    env.reset()
    with pytest.raises(SimulatorError):
        env.step({})


def test_pue_never_zero_and_ge_one(alcf_toy_config):
    """NOI-213 honesty rule: pue is report-only, >= 1.0 or null, never 0.0."""
    env = DecisionIntelligenceEnv(alcf_toy_config)
    obs, info = env.reset()
    action = {"facility": {
        "admit_kw": 0.0, "curtail_kw": 0.0, "chiller_setpoint_c": 7.0,
        "pump_duty_pct": 30.0, "zone_setpoint_c": 20.0, "deferrable_share_pct": 100.0,
    }}
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        pue = obs["facility"]["pue"]
        assert pue is None or pue >= 1.0
        assert pue != 0.0


def test_placement_respects_admission_budget():
    """Layer B placements must never exceed the admission budget r*_t handed
    down by Layer A, even when several placements are submitted in one
    action batch (regression test for the intra-step cumulative-budget bug
    fixed while building the baselines)."""
    cfg = build_scenario_config("alcf", scenario_id="budget_test", arrivals_enabled=True, seed=7)
    env = DecisionIntelligenceEnv(cfg)
    obs, info = env.reset()

    # First step: open a small budget, then try to place every queued task
    # regardless of mask -- the env must not admit more than the budget.
    small_budget = 3.0
    levers = obs["facility"]["levers"]
    action = {
        "facility": {
            "admit_kw": small_budget, "curtail_kw": 0.0,
            "chiller_setpoint_c": levers["chiller_setpoint_c"],
            "pump_duty_pct": levers["pump_duty_pct"],
            "zone_setpoint_c": levers["zone_setpoint_c"],
            "deferrable_share_pct": levers["deferrable_share_pct"],
        },
        "placements": [
            {"task_id": t["task_id"], "host_id": cfg["hosts"]["inventory"][0]["host_id"]}
            for t in obs["hosts"]["queue"]
        ],
    }
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2["facility"]["admitted_kw"] <= small_budget + 1e-6
    assert info2["violations"] == []


def test_migrations_masked_outside_window():
    cfg = build_scenario_config("alcf", scenario_id="window_test", arrivals_enabled=False, seed=3)
    assert cfg["facility"]["action_window"]["z_fixed"] is True
    env = DecisionIntelligenceEnv(cfg)
    obs, info = env.reset()
    assert 0 not in env.window_hours  # t=0 is outside the [2,3,4,5] window in the default adapter config
    action = {"migrations": [{"task_id": "n/a", "from_host": "h00", "to_host": "h01"}]}
    obs2, reward, terminated, truncated, info2 = env.step(action)
    # masked -> silently ignored, never logged as a violation (joint spec §3.2)
    assert info2["violations"] == []
    assert obs2["hosts"]["utilization"] == obs["hosts"]["utilization"]  # nothing moved
