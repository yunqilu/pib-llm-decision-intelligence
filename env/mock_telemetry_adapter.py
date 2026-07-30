"""
env/mock_telemetry_adapter.py — bridges PhysaFlow NOI's mock interfaces
(services/blackbox_mock.py, services/telemetry_chunker.py numeric fields, in
the pib-llm-backend-main repo) into this environment's scenario_config
format.

Scope note (Role 16 boundary): pib-llm-backend-main is a separate repo we do
not import at runtime here — this module is self-contained and re-encodes
only the already-published numeric fields documented, with file:line
citations, in docs/mock_telemetry_inventory.md (§2.1, §4). No Blackbox/
BioCore internals are touched. If/when this workstream gets direct access to
the mock service, swap TENANT_TELEMETRY below for a live call; the
scenario_config shape it produces does not change.

Fields marked [sourced] below are copied from docs/mock_telemetry_inventory.md;
everything else ([approximated]) is this workstream's own synthetic surrogate/
inventory (hosts §1.6, arrivals §1.7 — owner: James, joint spec Q2/Q3), kept
here so it's declared and versioned in one place per the provenance rule
(data_schema.md §0.4).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "v0.2"

# --------------------------------------------------------------------------- #
# [sourced] mock telemetry, per docs/mock_telemetry_inventory.md §2.1 / §4
# --------------------------------------------------------------------------- #

TENANT_TELEMETRY: Dict[str, Dict[str, Any]] = {
    "alcf": {
        "it_load_kw": 280.5,
        "energy_kwh": 1000.0,
        "energy_cost_usd": 80.0,        # -> price_usd_per_kwh ~= 0.08
        "co2e_kg": 386.0,               # -> carbon_kg_per_kwh ~= 0.386
        "thermal_redline_text": "Do not exceed thermal redline (28C inlet)",
        "t_max_inlet_c": 28.0,
        "sla_text": "Maintain HPC job queue SLA",
        "window_hours": None,           # ALCF mock has no window (inventory §2.1)
    },
    "acme": {
        "it_load_kw": 250.0,
        "energy_kwh": 1000.0,
        "energy_cost_usd": 80.0,
        "co2e_kg": 386.0,
        "sla_text": "Maintain N+1 cooling redundancy",
        "window_hours": [2, 3, 4, 5],   # "02:00-06:00 local time"
        "t_max_inlet_c": 30.0,          # no explicit ACME redline text; kept above ALCF's, [approximated]
    },
    "tenant-001": {
        "it_load_kw": 200.0,
        "energy_kwh": 1000.0,
        "energy_cost_usd": 80.0,
        "co2e_kg": 386.0,
        "sla_text": "No reducir por debajo de 18C en zona de oficinas",
        "s_min_c": 18.0,                # sourced text, hand-parsed
        "window_hours": None,
        "t_max_inlet_c": 28.0,
    },
    "tenant-002": {
        "it_load_kw": 320.0,
        "energy_kwh": 1000.0,
        "energy_cost_usd": 80.0,
        "co2e_kg": 386.0,
        "sla_text": "No afectar linea de produccion critica",
        "window_hours": None,
        "t_max_inlet_c": 28.0,
    },
}


def _profile(value: float, T: int) -> List[float]:
    return [round(value, 6)] * T


def _default_host_inventory(baseline_kw: float, n_hosts: int = 8) -> Dict[str, Any]:
    """[approximated] synthetic host inventory (Q2, owner: James).

    Evenly-rated hosts sized so `sum_i h_pow*P_rated == baseline_kw` exactly
    at 87.65625% average power utilization, matching Yunqi's placeholder
    convention in env/scenarios/alcf_toy.json so the coupling identity
    (data_schema.md §1.6) holds by construction for any baseline_kw.
    """
    rated_power_each = baseline_kw / n_hosts / 0.8765625
    inventory = [
        {
            "host_id": f"h{i:02d}",
            "rated_power_kw": round(rated_power_each, 6),
            "rated": {"cpu": 64, "mem": 512, "power": round(rated_power_each, 6), "storage": 20},
        }
        for i in range(n_hosts)
    ]
    initial_utilization = [[0.55, 0.6, 0.8765625, 0.4] for _ in range(n_hosts)]
    return {
        "resources": ["cpu", "mem", "power", "storage"],
        "inventory": inventory,
        "initial_utilization": initial_utilization,
        "provenance": {
            "tag": "approximated",
            "notes": (
                f"synthetic {n_hosts}x{rated_power_each:.3f}kW inventory, sized so "
                f"sum(h_pow*P_rated) == baseline_it_load_kw[0] == {baseline_kw} kW (Q2, owner: James)."
            ),
        },
    }


def _default_arrivals(enabled: bool) -> Dict[str, Any]:
    """[approximated] task arrival process (Q3, owner: James).

    Poisson arrivals, ~6 tasks/hour, resource demand uniform per axis, power
    draw sized so a handful of placements meaningfully move the admission
    budget without single tasks blowing past headroom. `sla_protected_prob`
    is an extra convenience field read by env/simulator.py (not schema-
    required; `demand` has `additionalProperties: true`).
    """
    if not enabled:
        return {
            "family": "none",
            "provenance": {
                "tag": "approximated",
                "notes": "Toy instance: arrivals off, validates certainty-equivalence (joint spec §5.4).",
            },
        }
    return {
        "family": "poisson",
        "rate_per_hour": 6.0,
        "demand": {
            "cpu": {"family": "uniform", "low": 1.0, "high": 8.0},
            "mem": {"family": "uniform", "low": 4.0, "high": 32.0},
            "power": {"family": "uniform", "low": 0.5, "high": 4.0},
            "storage": {"family": "uniform", "low": 1.0, "high": 10.0},
            "sla_protected_prob": 0.2,
        },
        "provenance": {
            "tag": "approximated",
            "notes": "Synthetic Poisson arrival process (Q3, owner: James); Week 5 stress tests vary this block only.",
        },
    }


def build_scenario_config(
    tenant: str,
    scenario_id: str,
    T: int = 24,
    seed: int = 42,
    arrivals_enabled: bool = False,
    n_hosts: int = 8,
    p_cap_kw: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a schema-valid scenario_config for `tenant` from documented mock
    telemetry (TENANT_TELEMETRY) plus this workstream's declared synthetic
    surrogates/inventory. Validate the result with `env/schemas/validate.py`
    before use (this module's `__main__` does that for all four tenants).
    """
    if tenant not in TENANT_TELEMETRY:
        raise ValueError(f"unknown tenant {tenant!r}; choose one of {sorted(TENANT_TELEMETRY)}")
    tel = TENANT_TELEMETRY[tenant]

    baseline_kw = tel["it_load_kw"]
    price = tel["energy_cost_usd"] / tel["energy_kwh"]
    carbon = tel["co2e_kg"] / tel["energy_kwh"]
    p_cap = p_cap_kw if p_cap_kw is not None else round(baseline_kw * 1.43, 2)  # ~ alcf_toy 280.5->400 ratio
    sla_floor = round(baseline_kw * 0.784, 2)  # ~ alcf_toy 280.5->220 ratio, [approximated]

    hours = tel["window_hours"] or [2, 3, 4, 5]
    z_fixed = True  # all 4 mock tenants have windows known a priori (docs/decisions/2026-07-16-solver-choice.md §1)

    scenario = {
        "meta": {
            "scenario_id": scenario_id,
            "tenant": tenant,
            "T": T,
            "dt_hours": 1.0,
            "seed": seed,
        },
        "facility": {
            "baseline_it_load_kw": _profile(baseline_kw, T),
            "p_cap_kw": p_cap,
            "sla_floor_kw": _profile(sla_floor, T),
            "t_max_inlet_c": tel["t_max_inlet_c"],
            "price_usd_per_kwh": _profile(round(price, 6), T),
            "carbon_kg_per_kwh": _profile(round(carbon, 6), T),
            "stranded_denominator_kwh": None,  # harness computes (data_schema.md §3.1)
            "levers": {
                "c": {"min": 5.0, "max": 12.0, "delta_max": 1.5, "ref": 7.0},
                "u": {"min": 30.0, "max": 100.0, "delta_max": 20.0},
                "s": {"min": tel.get("s_min_c", 18.0), "max": 27.0, "delta_max": 1.0},
                "q": {"min": 60.0, "max": 100.0, "delta_max": 10.0},
                "ramp_kw": 50.0,
            },
            "surrogates": {
                "cooling_overhead": {"beta0": 20.0, "beta1": 0.15, "beta2": 1.0, "beta3": 10.0, "c_ref": 7.0},
                "inlet_temp": {"a0": 18.0, "a1": 0.03, "a2": 2.0},
                "cooling_effort": {"kappa0": 1.0, "kappa_c": 0.15, "kappa_u": 1.0, "kappa_max": 5.0, "rho": 0.8},
            },
            "action_window": {
                "hours": hours,
                "z_fixed": z_fixed,
                "source_text": tel.get("sla_text", ""),
            },
            "provenance": {
                "tag": "mixed",
                "source_text": tel.get("sla_text", "") + "; " + tel.get("thermal_redline_text", ""),
                "notes": (
                    f"baseline/price/carbon sourced from mock ({tenant} it_load_kw={baseline_kw}; "
                    "ratios of estimated_impact_24h, docs/mock_telemetry_inventory.md §2.1); "
                    "p_cap, sla_floor, surrogate coefficients, lever bounds approximated."
                ),
            },
        },
        "hosts": _default_host_inventory(baseline_kw, n_hosts=n_hosts),
        "arrivals": _default_arrivals(arrivals_enabled),
    }
    return scenario


def build_all_tenant_scenarios(arrivals_enabled: bool = False) -> Dict[str, Dict[str, Any]]:
    return {
        tenant: build_scenario_config(tenant, scenario_id=f"{tenant}_v0", arrivals_enabled=arrivals_enabled)
        for tenant in TENANT_TELEMETRY
    }


if __name__ == "__main__":
    import json
    import pathlib
    import sys

    here = pathlib.Path(__file__).parent
    sys.path.insert(0, str(here / "schemas"))
    from validate import cross_field_errors  # type: ignore
    import jsonschema

    schema = json.loads((here / "schemas" / "scenario_config.schema.json").read_text())

    for tenant, cfg in build_all_tenant_scenarios().items():
        jsonschema.validate(cfg, schema)
        errs = cross_field_errors(cfg)
        status = "OK" if not errs else "INVALID: " + "; ".join(errs)
        print(f"{tenant:12s} {status}")
