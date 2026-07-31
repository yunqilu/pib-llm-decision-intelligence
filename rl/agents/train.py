"""
rl/agents/train.py — train the actor-critic agent (rl/agents/actor_critic.py)
on env/simulator.py via rl/gym_wrapper.py, with full experiment tracking
(shared/experiment_tracking.py): every run's exact config, seed, and
per-episode metrics are logged to rl/results/, so "run X got Y% recovery"
is always reproducible and auditable later.

Usage:
    python rl/agents/train.py --tenant alcf --episodes 300 --seed 0
    python rl/agents/train.py --tenant alcf --episodes 300 --seed 0 --lambda 1.0 0.02 0.02 0.02
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from env.mock_telemetry_adapter import build_scenario_config  # noqa: E402
from rl.agents.actor_critic import ActorCriticNet, run_training_episode  # noqa: E402
from rl.gym_wrapper import GymDispatchEnv  # noqa: E402
from shared.benchmark_harness import compute_D  # noqa: E402
from shared.experiment_tracking import ExperimentTracker, RunConfig, set_seed  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default="alcf", choices=["alcf", "acme", "tenant-001", "tenant-002"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-queue-len", type=int, default=32)
    parser.add_argument("--lambda", dest="lambda_vector", type=float, nargs=4, default=[1.0, 0.01, 0.01, 0.01],
                         metavar=("L1_RECOVERY", "L2_COST", "L3_CARBON", "L4_RISK"))
    parser.add_argument("--run-name", default="actor_critic_v0")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "rl" / "results"))
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    lambda_vector = tuple(args.lambda_vector)

    config = RunConfig(
        run_name=args.run_name,
        tenant=args.tenant,
        seed=args.seed,
        lambda_vector=lambda_vector,
        arrivals_enabled=True,
        n_episodes=args.episodes,
        extra={
            "lr": args.lr, "gamma": args.gamma, "hidden_dim": args.hidden_dim,
            "entropy_coef": args.entropy_coef, "value_coef": args.value_coef,
            "max_queue_len": args.max_queue_len, "algo": "actor_critic_v0",
        },
    )
    set_seed(args.seed)  # seeds python/numpy/torch; env's own RNG is seeded per-episode below

    scenario_config = build_scenario_config(
        args.tenant, scenario_id=f"{args.tenant}_ac_train", arrivals_enabled=True, seed=args.seed,
    )
    config.scenario_id = scenario_config["meta"]["scenario_id"]
    D = compute_D(scenario_config)  # fixed once per run -- see shared/experiment_tracking.py + docs/benchmark_protocol.md

    env = GymDispatchEnv(scenario_config, max_queue_len=args.max_queue_len, lambda_weights=lambda_vector)
    net = ActorCriticNet(N=env.N, M=env.M, max_queue_len=args.max_queue_len, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    tracker = ExperimentTracker(base_dir=args.out_dir, config=config, repo_dir=str(REPO_ROOT))
    print(f"logging to {tracker.run_dir}")

    # Per-episode env seeds are derived deterministically from the run seed
    # (not left to whatever RNG state happens to carry over) so the whole
    # training run -- not just episode 0 -- is reproducible given `--seed`.
    episode_seed_rng = __import__("numpy").random.default_rng(args.seed)

    recoveries = []
    for ep in range(args.episodes):
        ep_seed = int(episode_seed_rng.integers(0, 2**31 - 1))
        metrics = run_training_episode(
            env, net, optimizer, gamma=args.gamma,
            entropy_coef=args.entropy_coef, value_coef=args.value_coef, seed=ep_seed,
        )
        recovery_pct = metrics["f1_recovery_kwh"] / D if D else 0.0
        metrics["recovery_pct"] = recovery_pct
        metrics["episode_seed"] = ep_seed
        tracker.log_episode(ep, metrics)
        recoveries.append(recovery_pct)

        if ep % args.log_every == 0 or ep == args.episodes - 1:
            recent = recoveries[-args.log_every:]
            print(f"ep {ep:4d}  reward={metrics['reward_total']:8.2f}  "
                  f"recovery%={recovery_pct*100:6.2f}  (avg last {len(recent)}: {sum(recent)/len(recent)*100:6.2f})  "
                  f"loss={metrics['loss']:8.2f}  viol={metrics['violations']}")

    torch.save(net.state_dict(), tracker.model_path())
    n_tail = max(1, args.episodes // 10)
    tracker.finalize({
        "final_avg_recovery_pct_last_10pct": sum(recoveries[-n_tail:]) / len(recoveries[-n_tail:]),
        "first_avg_recovery_pct_first_10pct": sum(recoveries[:n_tail]) / len(recoveries[:n_tail]),
        "D_kwh": D,
        "total_episodes": args.episodes,
    })
    print(f"done. run dir: {tracker.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
