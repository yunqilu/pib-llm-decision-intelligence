# Solver Choice & Tooling (Week 2)
#### By Yunqi Lu
---

## Decision

- **Model class: MILP, run as LP at the current stage** 
- **Solver: HiGHS** (open-source, via `highspy`).
- **Modeling layer: Pyomo** (interface `appsi_highs`).
- Install: `pip install pyomo highspy`  (no license, no binaries to manage).

## 1. Approach: MILP → LP (per D3)

The formulation is linear by construction: D5 replaced the bilinear `PUE·L` with an affine cooling-overhead surrogate, so there is no nonlinear or general-convex structure anywhere. **cvxpy-style convex programming buys nothing here**. The only integrality is the action-window indicators `z_t` (formulation §3.4). Hence:

- **Full model = MILP** with ≤ 24 binaries (one `z_t` per hour, single tenant per D6).
- **Toy instance = LP**: all 4 mock tenants have windows known a priori, so `z_t` is fixed and the binaries disappear (D3). This is the configuration we run first.
- Either way the instance is tiny (T = 24, a few hundred variables/constraints); *any* solver is fast enough. The choice is therefore driven by licensing, handoff, and workflow fit, not speed.

## 2. Solver: HiGHS

1. **License-clean.** MIT-licensed; no commercial procurement question for PhysaFlow. Gurobi is faster but unnecessary at this scale, and my academic license does not cover company work.
2. **Best-in-class among open solvers.** HiGHS leads the open-source field on standard LP/MILP benchmarks (Mittelmann); CBC and GLPK are older and effectively in maintenance mode.
3. **One `pip install`.** `highspy` ships wheels; same Python env as James's RL stack. No separate solver binary for him to install when he calls the MILP oracle.
4. **Duals for free.** For the LP configuration HiGHS returns duals/reduced costs, which we can report as shadow prices on capacity/thermal constraints (useful for Leo's "which constraint binds" question).

## 3. Modeling layer: Pyomo

1. **ε-constraint sweep is the main workload (D4).** Week 4 needs a grid sweep over `(ε₂, ε₃, ε₄)`. With Pyomo `mutable Param`, each sweep point is "update ε, re-solve," no model rebuild. In PuLP this means patching constraint constants by hand.
2. **Solver-agnostic escape hatch.** If instances ever grow (multi-tenant, longer horizon), the same model runs on Gurobi/CPLEX by changing one `SolverFactory` line, no rewrite.
3. **Spec traceability.** Named constraint blocks let the code mirror the spec section-by-section (§3.1 capacity, §3.4 windows, …), preserving the `[sourced]`/`[approximated]` tagging in comments.
4. **Standard for handoff.** Pyomo is the de facto OR standard in Python; easier for a future reader than PuLP idioms or a raw `highspy` API.

**Rejected:** *Gurobi/gurobipy* (licensing, overkill, see §2.1); *PuLP* (lighter, but awkward ε-sweeps and weaker structure); *cvxpy* (built for convex cones we don't have; MILP support is solver-dependent); *raw highspy* (no named constraints, poor traceability).

## 4. Verified (2026-07-16, clean env, Pyomo 6.10.1)

- `pip install pyomo highspy`; a T = 24 toy MILP with 24 binaries solves via `SolverFactory('appsi_highs')` in **0.008 s**.
- ε-sweep via `mutable Param`: update ε, re-solve, correct new optimum, no rebuild.
- LP configuration returns duals (shadow prices) as expected.
- Fixing `z_t` is not enough for duals. HiGHS still treats fixed binaries as a MIP. Apply `TransformationFactory('core.relax_integer_vars')` after fixing; then duals load fine.
