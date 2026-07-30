"""
env/simulator.py — shared environment. Single source of truth for both the
optimization track and the RL track. Do not fork; propose changes via PR.

API contract (finalized v0.2 — see docs/data_schema.md, ADR
docs/decisions/2026-07-28-env-api-contract-v0.md):

    env = DecisionIntelligenceEnv(scenario_config)
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(action)

`scenario_config`, `observation`, and `action` are plain JSON-serializable
dicts shaped by env/schemas/*.schema.json (see env/schemas/validate.py for
scenario_config; this module enforces required keys at runtime so a
malformed payload fails loudly instead of silently producing garbage
metrics).

This class runs Layer A (facility setpoints) and Layer B (host dispatch)
together, per docs/joint_problem_spec.md §0/§3. `hosts` is never `None`
here; Yunqi's pure-LP/MILP optimizer instead consumes `scenario_config`
directly and never instantiates this class (joint spec §5.4,
certainty-equivalence check).

Formulas (docs/problem_spec.md §3-4, joint spec §2-3):
    L_host  = sum_i h[i, pow] * rated_power_kw[i]                         # physical host-dispatch power, kW
    L_t     = L_host*(q_t/100) - d_t                                      # effective IT load (deferral + curtailment applied)
    C_t     = beta0 + beta1*L_t - beta2*(c_t - c_ref) + beta3*(u_t/100)   # cooling overhead, kW
    F_t     = L_t + C_t                                                   # facility power, kW
    kappa_t = kappa0 + kappa_c*(c_ref - c_t) + kappa_u*(u_t/100)          # cooling effort
    T_in_t  = a0 + a1*L_t - a2*kappa_t                                    # inlet temp surrogate, C

`L_host` is the coupling-identity quantity (joint spec §3.1): it's what
reset() checks against baseline_it_load_kw[0], and what admitted placements
raise. `L_t` (the symbol problem_spec.md §3-4 actually uses in every
formula) is the *effective* load after this step's deferrable_share_pct
(q_t) and curtail_kw (d_t) are applied — see `_effective_load()`.

Reward (joint spec §2, unified; James's checklist items applied: L->Psi,
weights->lambda, cost+CO2e added). The env reports the four raw objective
terms in `info`; it does NOT scalarize with a fixed lambda, because lambda
is a policy/evaluation choice, not an environment parameter (joint spec §4:
"any reported RL result must state its lambda vector"). Callers combine:

    R = lambda1*f1_recovery - lambda2*f2_cost - lambda3*f3_carbon - lambda4*f4_risk

Hard constraints are enforced by masking/clamping (info["action_mask"],
_enforce_n_plus_one(), _effective_load()'s curtailment cap), never by reward
penalty — joint spec §3.2, NOI-213 honesty norms:
    - power cap (§4.1)      -> info["action_mask"] gates placements on F_t <= p_cap
    - thermal redline (§4.2)-> info["action_mask"] gates placements on T_in_t <= t_max
    - SLA floor (§4.3)      -> _effective_load() caps curtailment; q_t >= q_min via lever bounds
    - action windows (§4.4) -> info["action_mask"]["migration_allowed"]
    - N+1 reserve (§4.5)    -> _enforce_n_plus_one() clamps chiller/pump levers each step
    - ramp limit (§4.6)     -> deliberately soft (joint spec §3.2 table): folded into
                                f4_risk via migration churn cost, not hard-masked
`info["violations"]` must stay empty; a non-empty list means this module has
a bug, not that the policy misbehaved (masked-out actions are silently
ignored, not penalized).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "v0.2"


class SimulatorError(Exception):
    """Raised for malformed scenario_config/action payloads."""


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _require_keys(d: Dict[str, Any], keys: List[str], name: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise SimulatorError(f"{name} missing required keys: {missing}")


def _sample_delta(rng: np.random.Generator, demand_spec: Dict[str, Any], resources: List[str]) -> Dict[str, float]:
    """Sample delta_k (per-resource task demand) from arrivals.demand.

    `demand_spec` is keyed by resource axis, each `{"family": "uniform",
    "low": ..., "high": ...}` or `{"family": "lognormal", "mean_log": ...,
    "sigma_log": ...}`. Unspecified axes default to 0 (schema allows sparse
    demand dicts — see observation.schema.json `hosts.queue[k].demand`).
    """
    out: Dict[str, float] = {}
    for r in resources:
        spec = demand_spec.get(r)
        if spec is None:
            out[r] = 0.0
            continue
        fam = spec.get("family", "uniform")
        if fam == "uniform":
            out[r] = float(rng.uniform(spec["low"], spec["high"]))
        elif fam == "lognormal":
            out[r] = float(rng.lognormal(spec["mean_log"], spec["sigma_log"]))
        else:
            raise SimulatorError(f"arrivals.demand[{r!r}].family unknown: {fam!r}")
    return out


class DecisionIntelligenceEnv:
    """PhysaFlow stranded-capacity dispatch+planning environment, v0.2 contract."""

    def __init__(self, scenario_config: Dict[str, Any]):
        _require_keys(scenario_config, ["meta", "facility", "hosts", "arrivals"], "scenario_config")
        self.config = copy.deepcopy(scenario_config)
        self.T = self.config["meta"]["T"]
        self.dt = self.config["meta"]["dt_hours"]
        self.seed = self.config["meta"]["seed"]
        self.scenario_id = self.config["meta"]["scenario_id"]

        fac = self.config["facility"]
        for prof in ("baseline_it_load_kw", "sla_floor_kw", "price_usd_per_kwh", "carbon_kg_per_kwh"):
            if len(fac[prof]) != self.T:
                raise SimulatorError(f"facility.{prof} length {len(fac[prof])} != meta.T {self.T}")

        self.resources: List[str] = self.config["hosts"]["resources"]
        self.inventory: List[Dict[str, Any]] = self.config["hosts"]["inventory"]
        self.N = len(self.inventory)
        self.M = len(self.resources)
        self.rated_power = np.array([h["rated_power_kw"] for h in self.inventory], dtype=np.float64)
        if "power" not in self.resources:
            raise SimulatorError("hosts.resources must include 'power' for the coupling identity")
        self._pow_idx = self.resources.index("power")
        self._host_idx = {h["host_id"]: i for i, h in enumerate(self.inventory)}
        self._rated_cap = np.array(
            [[h["rated"][r] for r in self.resources] for h in self.inventory], dtype=np.float64
        )

        self.D = fac["stranded_denominator_kwh"]  # may be None -> harness/env computes a fallback
        self.window_hours = set(fac["action_window"]["hours"])
        self.z_fixed = fac["action_window"]["z_fixed"]

        arr = self.config["arrivals"]
        self.arrival_family = arr["family"]
        if self.arrival_family == "poisson":
            self.arrival_rate = arr["rate_per_hour"]
            self.arrival_demand = arr["demand"]
        elif self.arrival_family == "trace":
            raise SimulatorError("arrivals.family='trace' not implemented in simulator v0")
        elif self.arrival_family != "none":
            raise SimulatorError(f"arrivals.family unknown: {self.arrival_family!r}")

        self._rng = np.random.default_rng(self.seed)
        self._task_counter = 0

        # mutable state, set by reset()
        self.t = 0
        self.H: Optional[np.ndarray] = None  # [N, M] utilization in [0,1]
        self.queue: List[Dict[str, Any]] = []
        self.levers: Dict[str, float] = {}
        self.r_star_t = 0.0
        self.curtail_kw = 0.0
        self.admitted_this_step = 0.0
        self._done = True  # must reset() before step()

    # ------------------------------------------------------------------ #
    # reset / step
    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.t = 0
        self._task_counter = 0
        self._done = False

        util = np.array(self.config["hosts"]["initial_utilization"], dtype=np.float64)
        if util.shape != (self.N, self.M):
            raise SimulatorError(f"initial_utilization shape {util.shape} != ({self.N},{self.M})")
        self.H = util.copy()

        # coupling identity (joint spec §3.1 / env/schemas/validate.py), rel tol 1e-3
        L0 = float(np.sum(self.H[:, self._pow_idx] * self.rated_power))
        base0 = self.config["facility"]["baseline_it_load_kw"][0]
        if abs(L0 - base0) > 1e-3 * max(base0, 1.0):
            raise SimulatorError(
                f"coupling violated at reset: sum h_pow*P_rated={L0:.4f} != baseline_it_load_kw[0]={base0}"
            )

        lv = self.config["facility"]["levers"]
        self.levers = {
            "chiller_setpoint_c": lv["c"].get("ref", (lv["c"]["min"] + lv["c"]["max"]) / 2),
            "pump_duty_pct": lv["u"]["min"],
            "zone_setpoint_c": lv["s"].get("ref", (lv["s"]["min"] + lv["s"]["max"]) / 2),
            "deferrable_share_pct": lv["q"]["max"],
        }
        self.r_star_t = 0.0
        self.curtail_kw = 0.0
        self.admitted_this_step = 0.0
        self._cumulative_f1 = 0.0
        self.queue = self._sample_arrivals()

        obs = self._make_observation()
        info = self._make_info(f1=0.0, f2=0.0, f3=0.0, f4=0.0, violations=[])
        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[float], bool, bool, Dict[str, Any]]:
        if self._done:
            raise SimulatorError("step() called after episode terminated; call reset()")
        if not action:
            raise SimulatorError("action must set at least one of facility/placements/migrations")

        violations: List[str] = []

        # --- Layer A: facility setpoints ------------------------------------
        # Must run BEFORE the mask is computed: r*_t (admission budget) is set
        # here, and the placement mask below gates on r*_t. Computing the mask
        # first would check this step's placements against *last* step's
        # budget (0 at reset), silently starving every placement forever.
        if "facility" in action:
            fac_a = action["facility"]
            _require_keys(
                fac_a,
                ["admit_kw", "curtail_kw", "chiller_setpoint_c", "pump_duty_pct",
                 "zone_setpoint_c", "deferrable_share_pct"],
                "action.facility",
            )
            lv = self.config["facility"]["levers"]
            self.levers["chiller_setpoint_c"] = self._clamp_lever(
                fac_a["chiller_setpoint_c"], self.levers["chiller_setpoint_c"], lv["c"])
            self.levers["pump_duty_pct"] = self._clamp_lever(
                fac_a["pump_duty_pct"], self.levers["pump_duty_pct"], lv["u"])
            self.levers["zone_setpoint_c"] = self._clamp_lever(
                fac_a["zone_setpoint_c"], self.levers["zone_setpoint_c"], lv["s"])
            self.levers["deferrable_share_pct"] = self._clamp_lever(
                fac_a["deferrable_share_pct"], self.levers["deferrable_share_pct"], lv["q"])
            self._enforce_n_plus_one()  # problem_spec.md §4.5: kappa_t <= rho*kappa_max
            # r*_t: this step's admission budget handed down to Layer B (joint spec §3.1)
            self.r_star_t = max(0.0, float(fac_a["admit_kw"]))
            # d_t: curtailment. Stored raw here; _effective_load() re-clamps it
            # against the live SLA floor every time it's used, so the SLA
            # constraint (problem_spec.md §4.3: L_bar_t*(q_t/100) - d_t >=
            # L_bar^SLA_t) is enforced by construction rather than checked
            # post-hoc as a violation.
            self.curtail_kw = max(0.0, float(fac_a["curtail_kw"]))

        mask = self._action_mask()

        # --- Layer B: placements ---------------------------------------------
        # `mask` (computed once above, now correctly *after* r*_t is set) is
        # a fast first-pass filter. It is NOT re-derived per placement, so
        # budget/headroom are tracked live here as each placement is applied
        # -- otherwise several placements that were each individually legal
        # against the start-of-step snapshot could collectively exceed this
        # step's admission budget r*_t or physical power headroom. Per-host
        # multi-resource
        # capacity is naturally self-enforcing via _apply_placement's clip,
        # but budget/headroom are facility-wide and must be tracked across
        # the whole batch.
        placements = action.get("placements", [])
        migrations = action.get("migrations", [])
        allowed_task_hosts = mask["placement_allowed"]       # {task_id: set(host_id)}
        migration_allowed = mask["migration_allowed"]        # bool

        p_cap = self.config["facility"]["p_cap_kw"]
        by_task = {task["task_id"]: task for task in self.queue}
        placed_ids: set = set()

        for p in placements:
            tid, hid = p["task_id"], p["host_id"]
            if tid in placed_ids or tid not in by_task:
                continue  # stale/duplicate id, not a masking issue: silently ignore
            if hid not in allowed_task_hosts.get(tid, set()):
                continue  # masked -> ignored, not penalized (joint spec §3.2)
            task = by_task[tid]
            dpow = task["demand"].get("power", 0.0)

            budget_remaining = self.r_star_t - self.admitted_this_step
            if dpow > budget_remaining + 1e-9:
                continue  # would exceed r*_t once earlier placements in this same batch are counted

            L_host_now = float(np.sum(self.H[:, self._pow_idx] * self.rated_power))
            L_eff_after = self._effective_load(L_host_now + dpow)
            _, F_after, _, T_after = self._facility_physics(L_eff_after)
            if F_after > p_cap + 1e-6:
                continue  # would exceed p_cap_kw once earlier placements in this same batch are counted
            if T_after > self.config["facility"]["t_max_inlet_c"] + 1e-6:
                continue  # would exceed t_max_inlet_c (problem_spec.md §4.2) once this batch is counted

            i = self._host_idx[hid]
            cap = self._rated_cap[i]
            delta = self._demand_vector(task)
            if np.any(self.H[i] + delta / np.maximum(cap, 1e-9) > 1.0 + 1e-9):
                continue  # host filled by an earlier placement in this same batch

            self._apply_placement(i, task)
            placed_ids.add(tid)
            self.admitted_this_step += dpow

        self.queue = [t for t in self.queue if t["task_id"] not in placed_ids]

        # --- Layer B: migrations ----------------------------------------------
        churn_kw = 0.0
        if migrations and migration_allowed:
            for m in migrations:
                fi, ti = self._host_idx.get(m["from_host"]), self._host_idx.get(m["to_host"])
                if fi is None or ti is None or fi == ti:
                    continue
                movable_kw = self.H[fi, self._pow_idx] * self.rated_power[fi]
                room_kw = (1.0 - self.H[ti, self._pow_idx]) * self.rated_power[ti]
                moved_kw = min(movable_kw, room_kw)
                if moved_kw <= 1e-9:
                    continue
                self.H[fi] = np.clip(self.H[fi] - (moved_kw / max(self.rated_power[fi], 1e-9)), 0.0, 1.0)
                self.H[ti] = np.clip(self.H[ti] + (moved_kw / max(self.rated_power[ti], 1e-9)), 0.0, 1.0)
                churn_kw += moved_kw

        # --- objectives ----------------------------------------------------------
        L_host_t = float(np.sum(self.H[:, self._pow_idx] * self.rated_power))
        L_t = self._effective_load(L_host_t)  # problem_spec.md L_t = Lbar*(q/100) - d_t (+ r_t via host dispatch)
        C_t, F_t, kappa_t, T_in_t = self._facility_physics(L_t)

        p_cap = self.config["facility"]["p_cap_kw"]
        if F_t > p_cap + 1e-6:
            violations.append(f"F_t={F_t:.3f} exceeds p_cap_kw={p_cap} despite masking")
        t_max = self.config["facility"]["t_max_inlet_c"]
        if T_in_t > t_max + 1e-6:
            violations.append(f"T_in_t={T_in_t:.3f} exceeds t_max_inlet_c={t_max} despite masking")

        t_idx = min(self.t, self.T - 1)
        p_t = self.config["facility"]["price_usd_per_kwh"][t_idx]
        gamma_t = self.config["facility"]["carbon_kg_per_kwh"][t_idx]
        f1_recovery = self.admitted_this_step * self.dt
        self._cumulative_f1 += f1_recovery
        f2_cost = p_t * F_t * self.dt
        f3_carbon = gamma_t * F_t * self.dt
        f4_risk = churn_kw + self._proximity_risk()

        D = self._resolved_D()
        recovery_pct = self._cumulative_f1 / D if D else 0.0

        self.t += 1
        terminated = self.t >= self.T
        truncated = False
        self._done = terminated

        if not terminated:
            # NOTE: r_star_t is deliberately NOT reset to 0 here. It's the
            # budget *level* Layer A last set (persists until a facility
            # action changes it again); admitted_this_step is the *this-step
            # consumption* counter and does reset. Zeroing r_star_t here was
            # a bug: it recreated the reset()-time "budget is 0, nothing can
            # be placed" deadlock on every single step, not just the first.
            self.admitted_this_step = 0.0
            self.queue.extend(self._sample_arrivals())

        obs = self._make_observation()
        info = self._make_info(f1=f1_recovery, f2=f2_cost, f3=f3_carbon, f4=f4_risk, violations=violations)
        info["recovery_pct"] = recovery_pct
        info["facility_power_kw"] = F_t
        info["inlet_temp_c"] = T_in_t

        # Reward is deliberately not returned as a fixed scalar here — see
        # module docstring. Policies/harness combine info's f1..f4 with their
        # own lambda and MUST log that lambda alongside every result
        # (joint spec §4, checklist "Reporting").
        reward = None
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # physics
    # ------------------------------------------------------------------ #

    def _effective_load(self, L_host: float) -> float:
        """problem_spec.md §3: L_t = Lbar_t*(q_t/100) - d_t (+ r_t, handled
        separately via host dispatch already baked into L_host). `q_t`
        (deferrable_share_pct) scales down how much of the dispatched host
        power is "in production" this step; `d_t` (curtail_kw) further cuts
        it. Both apply to L_host (the physical, host-dispatch-derived load),
        not the static baseline profile, so admitted/recovered capacity
        (which raises L_host via placements) still shows up in C_t/F_t/T_in_t.

        SLA floor (§4.3: Lbar_t*(q_t/100) - d_t >= Lbar^SLA_t) is enforced
        here by construction, not as a separate check: curtailment is capped
        so the result never dips below the scenario's sla_floor_kw at this
        step. (q_t itself is already bounded below by q^min via the normal
        lever clamp in step(), covering the other half of §4.3.)
        """
        q_frac = self.levers["deferrable_share_pct"] / 100.0
        provisional = L_host * q_frac
        t_idx = min(self.t, self.T - 1)
        sla_floor = self.config["facility"]["sla_floor_kw"][t_idx]
        max_curtail = max(0.0, provisional - sla_floor)
        curtail_applied = min(self.curtail_kw, max_curtail)
        return max(0.0, provisional - curtail_applied)

    def _enforce_n_plus_one(self) -> None:
        """problem_spec.md §4.5: kappa_t <= rho*kappa_max (N+1 cooling
        reserve). kappa_t depends only on the facility levers (chiller
        setpoint, pump duty), not on load, so this is enforced by clamping
        the just-set levers -- never by penalizing a placement. Reduce pump
        duty first (bounded at its own min), then raise chiller setpoint if
        that alone isn't enough (bounded at its own max); both stay inside
        their normal [min, max] bounds from facility.levers."""
        co = self.config["facility"]["surrogates"]["cooling_overhead"]
        ce = self.config["facility"]["surrogates"]["cooling_effort"]
        cap = ce["rho"] * ce["kappa_max"]

        def kappa_of(c: float, u: float) -> float:
            return ce["kappa0"] + ce["kappa_c"] * (co["c_ref"] - c) + ce["kappa_u"] * (u / 100.0)

        c_t = self.levers["chiller_setpoint_c"]
        u_t = self.levers["pump_duty_pct"]
        if kappa_of(c_t, u_t) <= cap + 1e-9:
            return

        lv = self.config["facility"]["levers"]
        if ce["kappa_u"] > 0:
            u_needed = (cap - ce["kappa0"] - ce["kappa_c"] * (co["c_ref"] - c_t)) / (ce["kappa_u"] / 100.0)
            u_t = _clamp(u_needed, lv["u"]["min"], u_t)  # only ever reduce, never raise, pump duty here
            self.levers["pump_duty_pct"] = u_t

        if kappa_of(c_t, u_t) > cap + 1e-9 and ce["kappa_c"] > 0:
            c_needed = co["c_ref"] - (cap - ce["kappa0"] - ce["kappa_u"] * (u_t / 100.0)) / ce["kappa_c"]
            c_t = _clamp(c_needed, c_t, lv["c"]["max"])  # only ever raise, never lower, chiller setpoint here
            self.levers["chiller_setpoint_c"] = c_t

    def _facility_physics(self, L_t: float) -> Tuple[float, float, float, float]:
        c_t, u_t = self.levers["chiller_setpoint_c"], self.levers["pump_duty_pct"]
        co = self.config["facility"]["surrogates"]["cooling_overhead"]
        C_t = co["beta0"] + co["beta1"] * L_t - co["beta2"] * (c_t - co["c_ref"]) + co["beta3"] * (u_t / 100.0)
        C_t = max(0.0, C_t)  # PUE honesty rule: C_t >= 0 => PUE >= 1.0 (NOI-213)
        F_t = L_t + C_t
        it = self.config["facility"]["surrogates"]["inlet_temp"]
        ce = self.config["facility"]["surrogates"]["cooling_effort"]
        kappa_t = ce["kappa0"] + ce["kappa_c"] * (co["c_ref"] - c_t) + ce["kappa_u"] * (u_t / 100.0)
        T_in_t = it["a0"] + it["a1"] * L_t - it["a2"] * kappa_t
        return C_t, F_t, kappa_t, T_in_t

    def _proximity_risk(self) -> float:
        # Psi(s_{t+1}): soft OOM/throttling hazard near 100% utilization (joint spec §3.2/§7)
        over = np.clip(self.H - 0.9, 0.0, None)
        return float(np.sum(over ** 2)) * 100.0

    def _resolved_D(self) -> Optional[float]:
        if self.D is not None:
            return self.D
        # Harness normally computes D once per scenario and both layers must
        # use that same value (data_schema.md §3.1). This fallback only
        # exists so a standalone env run still reports a sane recovery_pct.
        base = self.config["facility"]["baseline_it_load_kw"]
        p_cap = self.config["facility"]["p_cap_kw"]
        return max(1e-9, (p_cap - base[0]) * self.T * self.dt)

    # ------------------------------------------------------------------ #
    # host / queue mechanics
    # ------------------------------------------------------------------ #

    def _demand_vector(self, task: Dict[str, Any]) -> np.ndarray:
        return np.array([task["demand"].get(r, 0.0) for r in self.resources], dtype=np.float64)

    def _apply_placement(self, i: int, task: Dict[str, Any]) -> None:
        delta = self._demand_vector(task)
        self.H[i] = np.clip(self.H[i] + delta / np.maximum(self._rated_cap[i], 1e-9), 0.0, 1.0)

    def _in_window(self) -> bool:
        return (self.t in self.window_hours) if self.z_fixed else True

    def _clamp_lever(self, requested: float, current: float, bounds: Dict[str, float]) -> float:
        step = _clamp(requested - current, -bounds["delta_max"], bounds["delta_max"])
        return _clamp(current + step, bounds["min"], bounds["max"])

    def _sample_arrivals(self) -> List[Dict[str, Any]]:
        if self.arrival_family != "poisson":
            return []
        rate = self.arrival_rate
        rate_t = rate[self.t] if isinstance(rate, list) else rate
        n = int(self._rng.poisson(rate_t))
        out = []
        for _ in range(n):
            self._task_counter += 1
            demand = _sample_delta(self._rng, self.arrival_demand, self.resources)
            sla_prob = self.arrival_demand.get("sla_protected_prob", 0.0)
            out.append({
                "task_id": f"t{self.t:02d}-{self._task_counter:04d}",
                "demand": demand,
                "sla_protected": bool(self._rng.random() < sla_prob),
                "arrived_t": self.t,
            })
        return out

    # ------------------------------------------------------------------ #
    # masking / observation / info
    # ------------------------------------------------------------------ #

    def _action_mask(self) -> Dict[str, Any]:
        """Hard-constraint mask (joint spec §3.2): power cap (§4.1), thermal
        redline (§4.2), per-host multi-resource capacity, admission budget,
        action windows. N+1 (§4.5) is enforced separately, at the lever
        level, by _enforce_n_plus_one() -- it depends only on facility
        levers, not load, so it never needs to gate a placement. SLA floor
        (§4.3) is enforced inside _effective_load(). Recomputed fresh every
        call from current state."""
        p_cap = self.config["facility"]["p_cap_kw"]
        t_max = self.config["facility"]["t_max_inlet_c"]
        L_host_now = float(np.sum(self.H[:, self._pow_idx] * self.rated_power))
        budget_remaining = max(0.0, self.r_star_t - self.admitted_this_step)

        placement_allowed: Dict[str, set] = {}
        for task in self.queue:
            allowed: set = set()
            dpow = task["demand"].get("power", 0.0)
            if dpow <= budget_remaining + 1e-9:
                L_eff_after = self._effective_load(L_host_now + dpow)
                _, F_after, _, T_after = self._facility_physics(L_eff_after)
                if F_after <= p_cap + 1e-6 and T_after <= t_max + 1e-6:
                    delta = self._demand_vector(task)
                    for host_id, i in self._host_idx.items():
                        if np.all(self.H[i] + delta / np.maximum(self._rated_cap[i], 1e-9) <= 1.0 + 1e-9):
                            allowed.add(host_id)
            placement_allowed[task["task_id"]] = allowed

        return {
            "placement_allowed": placement_allowed,
            "migration_allowed": self._in_window(),
        }

    def _make_observation(self) -> Dict[str, Any]:
        L_host_t = float(np.sum(self.H[:, self._pow_idx] * self.rated_power))
        L_t = self._effective_load(L_host_t)
        _, F_t, kappa_t, T_in_t = self._facility_physics(L_t)
        t_idx = min(self.t, self.T - 1)
        return {
            "t": self.t,
            "facility": {
                "it_load_kw": L_t,
                "facility_power_kw": F_t,
                "pue": (F_t / L_t) if L_t > 1e-9 else None,  # report-only, NOI-213: never 0.0
                "inlet_temp_c": T_in_t,
                "cooling_effort": kappa_t,
                "levers": dict(self.levers),
                "admitted_kw": self.admitted_this_step,
                "budget_remaining_kw": max(0.0, self.r_star_t - self.admitted_this_step),
                "price_usd_per_kwh": self.config["facility"]["price_usd_per_kwh"][t_idx],
                "carbon_kg_per_kwh": self.config["facility"]["carbon_kg_per_kwh"][t_idx],
                "in_window": self._in_window(),
            },
            "hosts": {
                "utilization": self.H.tolist(),
                "queue": [dict(t) for t in self.queue],
            },
        }

    def _make_info(self, f1: float, f2: float, f3: float, f4: float, violations: List[str]) -> Dict[str, Any]:
        return {
            "f1_recovery_kwh": f1,
            "f2_cost_usd": f2,
            "f3_co2_kg": f3,
            "f4_risk": f4,
            "action_mask": self._action_mask(),
            "violations": violations,
            "provenance": {
                "scenario_id": self.scenario_id,
                "schema_version": SCHEMA_VERSION,
            },
        }
