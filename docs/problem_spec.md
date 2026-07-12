# Shared Problem Specification

**Status:** DRAFT — to be finalized jointly by Yunqi and James.

## State / observation space
(TBD — draft candidate fields from the mock telemetry inventory: it_load_kw,
pue, cpu_util_pct, energy_kwh, energy_cost_usd, co2_kg, confidence, risk.)

## Action / decision variables
(TBD — draft candidates: it_load_kw reduction, chiller_setpoint_c,
pump_duty_pct, setpoint_temperature_c, production_load_pct.)

## Objective
(TBD — % of stranded capacity recovered, traded off against quality/latency/
robustness.)

## Constraints
(TBD — currently only available as free text in the mock Blackbox data;
needs hand-encoding into numeric bounds here.)

## Known gaps
- Water is not represented in NOI's mock interfaces at all.
- Thermal state (inlet/return temp) is not a structured field — only appears
  narratively inside constraint text.
