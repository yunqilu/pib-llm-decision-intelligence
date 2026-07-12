# Data Schema — what the environment exposes to both solvers

**Status:** DRAFT.

Sourced from the Week 1 mock-telemetry inventory (NOI's
`services/blackbox_mock.py` and `services/telemetry_chunker.py`).

| Field | Type | Source |
|---|---|---|
| it_load_kw | float (kW) | mock Blackbox / telemetry chunker |
| pue | float (>=1.0) | mock Blackbox / telemetry chunker |
| cpu_util_pct | float | telemetry chunker |
| energy_kwh | float | mock Blackbox summary |
| energy_cost_usd | float | mock Blackbox summary |
| co2_kg | float | mock Blackbox summary |
| confidence | float [0,1] | mock Blackbox summary |
| risk | float [0,1] | mock Blackbox summary |

**Known gap:** water and structured thermal fields (inlet/return temp) are
not present in NOI's mock interfaces — see `docs/problem_spec.md`.
