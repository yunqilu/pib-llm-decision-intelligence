#!/usr/bin/env python3
"""Validate a scenario_config instance: JSON Schema + cross-field rules.

Usage:  python env/schemas/validate.py env/scenarios/alcf_toy.json
Rules beyond the schema (see docs/data_schema.md):
  - all facility profiles have length meta.T
  - action_window.hours within [0, T)
  - initial_utilization has shape [N][M]
  - coupling: sum_i h_{i,pow} * rated_power_kw ~= baseline_it_load_kw[0]
"""
import json
import pathlib
import sys

import jsonschema

HERE = pathlib.Path(__file__).parent
PROFILES = ("baseline_it_load_kw", "sla_floor_kw", "price_usd_per_kwh", "carbon_kg_per_kwh")
REL_TOL = 1e-3  # coupling tolerance, joint spec §3.1


def cross_field_errors(inst: dict) -> list[str]:
    errors = []
    T = inst["meta"]["T"]
    fac = inst["facility"]
    for f in PROFILES:
        if len(fac[f]) != T:
            errors.append(f"facility.{f}: length {len(fac[f])} != meta.T = {T}")
    bad = [h for h in fac["action_window"]["hours"] if not 0 <= h < T]
    if bad:
        errors.append(f"facility.action_window.hours out of [0, T): {bad}")

    hosts = inst["hosts"]
    n, m = len(hosts["inventory"]), len(hosts["resources"])
    u = hosts["initial_utilization"]
    if len(u) != n or any(len(row) != m for row in u):
        errors.append(f"hosts.initial_utilization: shape != [{n}][{m}]")
    elif "power" in hosts["resources"]:
        j = hosts["resources"].index("power")
        l0 = sum(row[j] * inv["rated_power_kw"] for row, inv in zip(u, hosts["inventory"]))
        base0 = fac["baseline_it_load_kw"][0]
        if abs(l0 - base0) > REL_TOL * max(base0, 1.0):
            errors.append(
                f"coupling: sum h_pow*P_rated = {l0:.3f} kW != baseline_it_load_kw[0] = {base0} kW"
            )
    for inv in hosts["inventory"]:
        missing = set(hosts["resources"]) - set(inv["rated"])
        if missing:
            errors.append(f"hosts.inventory[{inv['host_id']}].rated missing axes: {sorted(missing)}")
    return errors


def main(path: str) -> int:
    inst = json.loads(pathlib.Path(path).read_text())
    schema = json.loads((HERE / "scenario_config.schema.json").read_text())
    try:
        jsonschema.validate(inst, schema)
    except jsonschema.ValidationError as e:
        path = "$" + "".join(f"[{p!r}]" for p in e.absolute_path)
        print(f"INVALID (schema): {path}: {e.message}")
        return 1
    errors = cross_field_errors(inst)
    if errors:
        print("INVALID (cross-field):\n  " + "\n  ".join(errors))
        return 1
    print(f"OK: {path} is a valid scenario_config (T={inst['meta']['T']}, "
          f"tenant={inst['meta']['tenant']}, hosts={len(inst['hosts']['inventory'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
