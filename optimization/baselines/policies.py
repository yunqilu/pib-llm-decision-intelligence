"""
optimization/baselines/policies.py — random and greedy baseline dispatch
policies over env/simulator.py's v0.2 contract.

Both are Layer B (host dispatch) policies: they leave the Layer A facility
levers untouched (hold current setpoints) and only decide placements/
migrations, using `info["action_mask"]` — never guessing at feasibility
themselves (joint spec §3.2: hard constraints are masked, not penalized;
a policy that ignores the mask can only get its action silently dropped by
the env, not "cheat" into an infeasible state).

Every policy here is a plain callable `(observation, info) -> action`,
matching shared.benchmark_harness.Policy, so run_episode()/run_many() work
unmodified for either baseline.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


# Deliberately large: admit_kw only sets this step's *budget* r*_t handed to
# Layer B (joint spec §3.1); the env still gates actual placements on real
# physical headroom (p_cap - F_t) via info["action_mask"]. A baseline that
# "holds levers" should still open the tap all the way and let the mask do
# the real gating -- echoing back the previous budget_remaining_kw would be
# self-referential (it's 0 until a facility action sets it) and silently
# starve every placement forever.
_UNBOUNDED_ADMIT_KW = 1.0e6


def _hold_levers_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """No facility-side setpoint change: keep chiller/pump/zone/deferrable
    share where they are, and open the admission budget so host-level
    `placements` below are gated only by physical headroom, not by an
    artificial zero budget."""
    levers = observation["facility"]["levers"]
    return {
        "admit_kw": _UNBOUNDED_ADMIT_KW,
        "curtail_kw": 0.0,
        "chiller_setpoint_c": levers["chiller_setpoint_c"],
        "pump_duty_pct": levers["pump_duty_pct"],
        "zone_setpoint_c": levers["zone_setpoint_c"],
        "deferrable_share_pct": levers["deferrable_share_pct"],
    }


class RandomPolicy:
    """For each queued task, place it on a uniformly random *allowed* host
    (per info["action_mask"]) with probability `place_prob`; otherwise defer
    it this step. Migrations: with probability `migrate_prob`, issue one
    random legal migration (only ever proposed when the window is open —
    an unnecessary migration outside the window would just be masked, but we
    skip drawing it so the policy's own behavior is legible in logs)."""

    def __init__(self, place_prob: float = 0.7, migrate_prob: float = 0.05, seed: Optional[int] = None):
        self.place_prob = place_prob
        self.migrate_prob = migrate_prob
        self._rng = random.Random(seed)

    def __call__(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        mask = info["action_mask"]
        placements: List[Dict[str, str]] = []
        host_ids_seen: List[str] = []
        for task in observation["hosts"]["queue"]:
            allowed = list(mask["placement_allowed"].get(task["task_id"], ()))
            for hid in allowed:
                if hid not in host_ids_seen:
                    host_ids_seen.append(hid)
            if not allowed:
                continue
            if self._rng.random() < self.place_prob:
                placements.append({"task_id": task["task_id"], "host_id": self._rng.choice(allowed)})

        migrations: List[Dict[str, str]] = []
        if mask["migration_allowed"] and len(host_ids_seen) >= 2 and self._rng.random() < self.migrate_prob:
            from_host, to_host = self._rng.sample(host_ids_seen, 2)
            # env/simulator.py v0 models migration at the aggregate
            # host-power level (moves min(movable, room) kW between hosts)
            # rather than tracking which specific running task moved, since
            # v0 has no "currently running task" ledger, only queue/util —
            # task_id is schema-required but not consumed by the env yet.
            migrations.append({"task_id": "n/a", "from_host": from_host, "to_host": to_host})

        action: Dict[str, Any] = {"facility": _hold_levers_action(observation)}
        if placements:
            action["placements"] = placements
        if migrations:
            action["migrations"] = migrations
        return action


class GreedyPolicy:
    """Greedy host dispatch: sort the queue by task power demand (largest
    first, i.e. greedily bank the biggest wins toward the admission budget
    r*_t while it lasts), and for each task place it on the *most-utilized
    still-feasible* host (best-fit: pack tightly, keep other hosts free for
    future large tasks) among `info["action_mask"]`'s allowed set. SLA-
    protected tasks are placed first regardless of size, since they cannot be
    deferred once admitted (hard mask already forbids deferring them; placing
    them early avoids them silently starving on a budget-exhausted step)."""

    def __call__(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        mask = info["action_mask"]
        util = observation["hosts"]["utilization"]

        # host_id -> current mean utilization (best-fit score), derived from
        # the mask's host universe (any task's allowed-set membership) and
        # the observation's utilization matrix, indexed positionally.
        placement_allowed = mask["placement_allowed"]
        host_ids_seen: List[str] = []
        for allowed in placement_allowed.values():
            for hid in allowed:
                if hid not in host_ids_seen:
                    host_ids_seen.append(hid)
        # utilization rows are positional (hosts.inventory order); recover
        # that order the same way the env does: sorted by host_id numeric
        # suffix if present, else lexicographic, matching env/scenarios/*.json
        # convention ("h00", "h01", ...). If the scenario's host_ids don't
        # follow that convention this falls back to lexicographic order,
        # which still gives a *consistent* (if not perfectly aligned) score
        # -- acceptable for a baseline, not for the oracle.
        ordered_hosts = sorted(host_ids_seen)
        host_util_mean = {hid: (sum(util[i]) / len(util[i]) if i < len(util) else 0.0)
                           for i, hid in enumerate(ordered_hosts)}

        queue = sorted(
            observation["hosts"]["queue"],
            key=lambda task: (not task["sla_protected"], -task["demand"].get("power", 0.0)),
        )

        placements: List[Dict[str, str]] = []
        for task in queue:
            allowed = placement_allowed.get(task["task_id"], set())
            if not allowed:
                continue  # already excludes anything over budget/headroom/capacity (info["action_mask"])
            best_fit = max(allowed, key=lambda hid: host_util_mean.get(hid, 0.0))
            placements.append({"task_id": task["task_id"], "host_id": best_fit})
            host_util_mean[best_fit] = host_util_mean.get(best_fit, 0.0) + 0.01  # discourage stacking everything on one host

        action: Dict[str, Any] = {"facility": _hold_levers_action(observation)}
        if placements:
            action["placements"] = placements
        # Greedy baseline does not migrate: migration's payoff (freeing a
        # host for a future large task) requires look-ahead this policy
        # doesn't do; a no-op here is more legible than a random guess.
        return action
