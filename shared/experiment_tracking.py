"""
shared/experiment_tracking.py — seeding and run logging shared across
tracks (RL training, optimization solves, baseline sweeps), so "reproduce
run X" always means the same thing: same code commit, same config, same
seed, same scenario_id, logged in one place.

Usage:
    from shared.experiment_tracking import RunConfig, ExperimentTracker, set_seed

    config = RunConfig(run_name="ac_v0", tenant="alcf", seed=0,
                        lambda_vector=(1.0, 0.01, 0.01, 0.01),
                        extra={"lr": 3e-4, "gamma": 0.99})
    set_seed(config.seed)
    tracker = ExperimentTracker(base_dir="rl/results", config=config)
    for ep in range(n_episodes):
        ...
        tracker.log_episode(ep, {"reward": r, "recovery_pct": rp, ...})
    tracker.finalize({"final_recovery_pct": rp})
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every source of randomness this repo's code touches: Python's
    `random`, numpy, and torch (CPU + CUDA) if installed. Does NOT seed a
    specific env instance -- env/simulator.py takes its seed from
    scenario_config.meta.seed, set independently (see RunConfig.seed /
    ExperimentTracker, which thread the same seed through both by default
    unless overridden)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms where available; harmless no-op otherwise.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    """Best-effort short commit hash for provenance in every run's
    config.json. Returns None (not an error) if git isn't available or this
    isn't a git checkout -- reproducibility-by-commit-hash is a nice-to-have,
    not a hard requirement."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def get_git_dirty(repo_dir: Optional[str] = None) -> Optional[bool]:
    """True if there are uncommitted changes -- worth knowing, since a run
    logged against a dirty tree isn't fully reproducible from the commit
    hash alone."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return len(out.stdout.strip()) > 0
    except Exception:
        pass
    return None


@dataclass
class RunConfig:
    """Everything needed to reproduce a run. `extra` holds
    algorithm-specific hyperparameters (lr, gamma, hidden_dim, ...) so this
    dataclass doesn't need to change shape per algorithm."""
    run_name: str
    tenant: str
    seed: int
    lambda_vector: Tuple[float, float, float, float]
    scenario_id: Optional[str] = None
    arrivals_enabled: bool = True
    n_episodes: int = 200
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["lambda_vector"] = list(self.lambda_vector)
        return d


class ExperimentTracker:
    """One run = one directory: `{base_dir}/{timestamp}_{run_name}_seed{seed}/`
    containing:
      - config.json    : RunConfig + git commit/dirty flag + package versions
      - metrics.jsonl   : one JSON object per logged episode (append-only,
                          crash-safe -- a killed run still leaves partial data)
      - summary.json    : written once, at finalize()
      - model.pt        : optional, if the caller saves a torch checkpoint here
    """

    def __init__(self, base_dir: str, config: RunConfig, repo_dir: Optional[str] = None):
        self.config = config
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.run_dir = Path(base_dir) / f"{timestamp}_{config.run_name}_seed{config.seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "config": config.as_dict(),
            "git_commit": get_git_commit(repo_dir),
            "git_dirty": get_git_dirty(repo_dir),
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp_utc": timestamp,
        }
        try:
            import torch
            meta["torch_version"] = torch.__version__
        except ImportError:
            meta["torch_version"] = None
        meta["numpy_version"] = np.__version__

        (self.run_dir / "config.json").write_text(json.dumps(meta, indent=2))
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._metrics_file = open(self._metrics_path, "a")

    def log_episode(self, episode: int, metrics: Dict[str, Any]) -> None:
        row = {"episode": episode, "wall_time_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), **metrics}
        self._metrics_file.write(json.dumps(row) + "\n")
        self._metrics_file.flush()
        os.fsync(self._metrics_file.fileno())  # crash-safe: survive a killed process, not just a killed thread

    def model_path(self, filename: str = "model.pt") -> Path:
        return self.run_dir / filename

    def finalize(self, summary: Dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._metrics_file.close()

    def __del__(self):
        try:
            if not self._metrics_file.closed:
                self._metrics_file.close()
        except Exception:
            pass
