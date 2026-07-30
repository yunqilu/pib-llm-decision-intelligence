"""
shared/benchmark_harness.py — runs any policy (RL) or solver (optimization)
against env/simulator.py and reports recovery %, cost/carbon/risk trade-offs,
and per-episode diagnostics for both tracks.

Both tracks are evaluated the same way (data_schema.md §4, joint spec §5):
  1. Same scenario_config feeds both layers.
  2. D (stranded_denominator_kwh) is computed once per scenario, here, and
     both layers must report against that same value — this is the one
     place D gets resolved when scenario_config.facility.stranded_denominator_kwh
     is null.
  3. Every result must disclose its lambda vector (if a scalar reward was
     used to train/score the policy) alongside f1..f4 and recovery %.
  4. `violations` must stay empty; a non-empty list is surfaced as a hard
     failure of the run, not folded into the score (env bug, not policy cost).

Usage:
    from env.simulator import DecisionIntelligenceEnv
    from shared.benchmark_harness import run_episode

    env = DecisionIntelligenceEnv(scenario_config)
    result = run_episode(env, policy, lambda_vector=(1.0, 0.0, 0.0, 0.0))
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Action = Dict[str, Any]
Observation = Dict[str, Any]
Info = Dict[str, Any]

# A policy is any callable (observation, info) -> action. Kept as a plain
# protocol (not a base class) so optimization/ solvers and rl/ agents can
# both satisfy it without inheriting from a shared harness type.
Policy = Callable[[Observation, Info], Action]


def compute_D(scenario_config: Dict[str, Any]) -> float:
    """Stranded-capacity denominator D (data_schema.md §3.1, joint spec §3.1):
    kWh of capacity unusable at baseline due to (i) multi-resource
    fragmentation at the host level and (ii) power/thermal headroom at the
    facility level, integrated over the horizon.

    If `scenario_config.facility.stranded_denominator_kwh` is set, that value
    is authoritative (both layers must use it). Otherwise the harness derives
    it here — once — so RL and optimization runs on the same scenario_id are
    guaranteed to divide by the same number.
    """
    fac = scenario_config["facility"]
    given = fac.get("stranded_denominator_kwh")
    if given is not None:
        return float(given)

    T = scenario_config["meta"]["T"]
    dt = scenario_config["meta"]["dt_hours"]
    baseline = fac["baseline_it_load_kw"]
    p_cap = fac["p_cap_kw"]

    # Facility-level headroom: capacity between baseline load and the power
    # cap, integrated over the horizon (mirrors env.simulator's fallback so a
    # harness-run and a standalone env-run agree when D is unset).
    facility_headroom_kwh = sum(max(0.0, p_cap - l) for l in baseline) * dt

    # Host-level fragmentation: capacity stranded because per-host
    # multi-resource utilization can't be perfectly packed (each host's
    # *tightest* binding resource wastes the slack on every other axis).
    # Approximated as total rated power times the average non-power-axis
    # slack at t=0.
    hosts = scenario_config["hosts"]
    resources = hosts["resources"]
    inv = hosts["inventory"]
    util = hosts["initial_utilization"]
    pow_idx = resources.index("power")
    frag_kwh = 0.0
    if len(inv) == len(util):
        for row, host in zip(util, inv):
            non_power = [row[j] for j in range(len(resources)) if j != pow_idx]
            if non_power:
                slack = 1.0 - max(non_power)  # bottleneck axis determines strandable power
                frag_kwh += slack * host["rated_power_kw"]
        frag_kwh *= T * dt

    return max(1e-9, facility_headroom_kwh + frag_kwh)


def scalarize(info: Info, lambda_vector: Sequence[float]) -> float:
    """R = lambda1*f1_recovery - lambda2*f2_cost - lambda3*f3_carbon - lambda4*f4_risk
    (joint spec §2/§4). The env deliberately does not do this internally —
    see env/simulator.py module docstring — so every caller must be explicit
    about lambda and report it (checklist "Reporting")."""
    l1, l2, l3, l4 = lambda_vector
    return (
        l1 * info["f1_recovery_kwh"]
        - l2 * info["f2_cost_usd"]
        - l3 * info["f3_co2_kg"]
        - l4 * info["f4_risk"]
    )


@dataclass
class EpisodeResult:
    scenario_id: str
    policy_name: str
    lambda_vector: Tuple[float, float, float, float]
    D: float
    f1_recovery_kwh: float = 0.0
    f2_cost_usd: float = 0.0
    f3_co2_kg: float = 0.0
    f4_risk: float = 0.0
    reward_total: float = 0.0
    steps: int = 0
    violations: List[str] = field(default_factory=list)
    per_step: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def recovery_pct(self) -> float:
        return self.f1_recovery_kwh / self.D if self.D else 0.0

    def summary(self) -> str:
        return (
            f"{self.policy_name:>10s} | scenario={self.scenario_id:<16s} "
            f"recovery%={self.recovery_pct * 100:6.2f} "
            f"f1={self.f1_recovery_kwh:8.2f}kWh f2=${self.f2_cost_usd:7.2f} "
            f"f3={self.f3_co2_kg:7.2f}kgCO2 f4={self.f4_risk:7.2f} "
            f"reward={self.reward_total:9.2f} viol={len(self.violations)} "
            f"lambda={self.lambda_vector}"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "policy_name": self.policy_name,
            "lambda_vector": list(self.lambda_vector),
            "schema_version": "v0.2",
            "D_kwh": self.D,
            "recovery_pct": self.recovery_pct,
            "f1_recovery_kwh": self.f1_recovery_kwh,
            "f2_cost_usd": self.f2_cost_usd,
            "f3_co2_kg": self.f3_co2_kg,
            "f4_risk": self.f4_risk,
            "reward_total": self.reward_total,
            "steps": self.steps,
            "violations": self.violations,
        }


def run_episode(
    env,
    policy: Policy,
    policy_name: str = "policy",
    lambda_vector: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    record_per_step: bool = False,
) -> EpisodeResult:
    """Run one full episode of `env` under `policy`, accumulate f1..f4 and the
    scalarized reward, and return an EpisodeResult. `env` must already be
    constructed with a scenario_config (env.simulator.DecisionIntelligenceEnv
    or anything exposing the same reset()/step() contract)."""
    D = compute_D(env.config)
    result = EpisodeResult(
        scenario_id=env.scenario_id,
        policy_name=policy_name,
        lambda_vector=tuple(lambda_vector),
        D=D,
    )

    obs, info = env.reset()
    terminated = False
    while not terminated:
        action = policy(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        r = scalarize(info, lambda_vector)
        result.f1_recovery_kwh += info["f1_recovery_kwh"]
        result.f2_cost_usd += info["f2_cost_usd"]
        result.f3_co2_kg += info["f3_co2_kg"]
        result.f4_risk += info["f4_risk"]
        result.reward_total += r
        result.steps += 1
        result.violations.extend(info["violations"])
        if record_per_step:
            result.per_step.append({"t": obs["t"], "reward": r, **{
                k: info[k] for k in ("f1_recovery_kwh", "f2_cost_usd", "f3_co2_kg", "f4_risk")
            }})
    return result


def run_many(
    env_factory: Callable[[], Any],
    policies: Dict[str, Policy],
    lambda_vector: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    n_episodes: int = 1,
) -> List[EpisodeResult]:
    """Convenience runner: evaluate each policy in `policies` for
    `n_episodes` fresh envs (call env_factory() once per episode so
    stochastic arrivals get independent draws unless the scenario fixes a
    seed). Returns a flat list of EpisodeResult, one per (policy, episode)."""
    results: List[EpisodeResult] = []
    for name, policy in policies.items():
        for _ in range(n_episodes):
            env = env_factory()
            results.append(run_episode(env, policy, policy_name=name, lambda_vector=lambda_vector))
    return results


def oracle_relative_pct(policy_result: EpisodeResult, oracle_recovery_pct: float) -> Optional[float]:
    """% of oracle bound (data_schema.md §4/§5: "Oracle bound: Layer A MILP
    solved with perfect foresight ... RL performance is reported as % of
    oracle"). `oracle_recovery_pct` comes from the optimization track's
    perfect-foresight MILP solve on the same scenario_id; this harness does
    not compute it (that's optimization/, not shared/)."""
    if oracle_recovery_pct <= 0:
        return None
    return 100.0 * policy_result.recovery_pct / oracle_recovery_pct
