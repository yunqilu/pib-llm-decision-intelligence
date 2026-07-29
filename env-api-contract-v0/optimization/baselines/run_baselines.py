"""
optimization/baselines/run_baselines.py — run random and greedy baseline
dispatch policies against the shared simulator and print a comparison table.

Usage:
    python optimization/baselines/run_baselines.py
    python optimization/baselines/run_baselines.py --episodes 10 --out optimization/results/baselines_v0.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from env.mock_telemetry_adapter import build_scenario_config  # noqa: E402
from env.simulator import DecisionIntelligenceEnv  # noqa: E402
from optimization.baselines.policies import GreedyPolicy, RandomPolicy  # noqa: E402
from shared.benchmark_harness import run_many  # noqa: E402

LAMBDA = (1.0, 0.01, 0.01, 0.01)  # recovery-first weighting; disclosed alongside every result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5, help="episodes per (policy, scenario)")
    parser.add_argument("--tenant", default="alcf", choices=["alcf", "acme", "tenant-001", "tenant-002"])
    parser.add_argument("--out", default=None, help="optional path to write JSON results")
    args = parser.parse_args()

    scenarios = {
        "toy (arrivals off)": build_scenario_config(
            args.tenant, scenario_id=f"{args.tenant}_toy", arrivals_enabled=False, seed=42
        ),
        "stochastic (poisson)": build_scenario_config(
            args.tenant, scenario_id=f"{args.tenant}_stochastic", arrivals_enabled=True, seed=None
        ),
    }

    all_results = []
    for label, cfg in scenarios.items():
        print(f"\n=== {label} — tenant={args.tenant} ===")

        def env_factory(cfg=cfg):
            return DecisionIntelligenceEnv(cfg)

        policies = {
            "random": RandomPolicy(place_prob=0.7, migrate_prob=0.05, seed=0),
            "greedy": GreedyPolicy(),
        }
        results = run_many(env_factory, policies, lambda_vector=LAMBDA, n_episodes=args.episodes)
        for r in results:
            print(r.summary())
        all_results.extend(r.as_dict() for r in results)

        for name in policies:
            subset = [r for r in results if r.policy_name == name]
            avg_recovery = sum(r.recovery_pct for r in subset) / len(subset) * 100
            avg_viol = sum(len(r.violations) for r in subset) / len(subset)
            print(f"  -> {name:>7s} avg recovery% = {avg_recovery:6.2f}   avg violations = {avg_viol:.2f}")

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"lambda_vector": list(LAMBDA), "results": all_results}, indent=2))
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
