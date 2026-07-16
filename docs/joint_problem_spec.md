# PhysaFlow Joint Problem Spec (Week 2)  
Optimization × MDP Reconciliation
By Yunqi Lu

---

## 0. One problem, two layers

Both formulations' goal is to **recover stranded data-center capacity**, but they are currently at different granularities. This spec does not force one model to absorb the other, but it fixes a single shared problem statement and treats the two versions as complementary layers of it:

> For the next 24 hours, decide hour by hour how to run the facility (eg. how much new load to admit, how to set the cooling), so that recovered capacity (recovery %) is as large as possible, without breaking any rules (power cap, temperature redline, SLA, N+1 redundancy, allowed action windows), and without letting energy cost, CO₂e, or risk get out of hand.

| Layer            | Owner | Model                                                   | Granularity                                                                     | Role                                                                                                                                           |
| ---------------- | ----- | ------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **1.  Planning** | Yunqi | Multi-period MILP (LP with fixed windows), ε-constraint | Facility: whole-room totals `L_t` (kW), 5 mock levers                           | Draws up the optimal day-ahead plan assuming no surprises; maps out the trade-off (Pareto) curve; serves as the ceiling RL is measured against |
| **2.  Dispatch** | James | MDP, policy learned by RL                               | Host: which task on which machine (N×M utilization matrix), placement/migration | Real-time dispatch that reacts to surprises - tasks arriving, finishing, load jumping around                                                   |

The planning layer is simply the dispatch layer **with the surprises removed**: assume the day goes exactly as forecast (baseline `L̄_t`) and add all hosts up into one facility total, and MDP turns into MILP. The plan is the "no-surprises" special case of the dispatcher's problem, which also gives us a concrete way to check the two models against each other (**certainty-equivalent relaxation**).

---

## 1. Shared notation (collision resolution)

The two documents currently overload four symbols. Fixed as follows: **both docs should adopt these in v2**. All four renames are edits to **James's MDP doc**

| Symbol   | Reserved meaning                        | Was colliding with                | Resolution                                                                                     |
| -------- | --------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------- |
| `L_t`    | effective IT load (kW), Yunqi §2        | James's SLA/risk penalty `L(·)`   | Penalty renamed **`Ψ(s_{t+1})`**                                                               |
| `γ_t`    | carbon intensity (kg/kWh) [sourced]     | James's reward weight `γ`         | Reward weights `α, β, γ` renamed **`λ₁..λ₄`** (they *are* the weighted-sum bridge weights, §4) |
| `β₀..β₃` | cooling-overhead surrogate coefficients | James's migration-cost weight `β` | Covered by the `λ` rename                                                                      |
| `q_t`    | production/deferrable load share (%)    | James's queue item `q_k`          | Queue item renamed **`v_k`** with resource-demand vector **`δ_k ∈ ℝ^M`**                       |


---

## 2. Layer B restated in shared notation

James's model has four parts: (这部分可以稍微简单看看，重点在下面的3 changes)

1. **What the agent sees (state `s_t = (H_t, Q_t)`).** 
A snapshot of every machine: for each host, how full each resource is (CPU, memory, power, storage), as percentages `h_{i,j}`, plus the waiting line `Q_t`: tasks `v_k` that have arrived but not yet been placed, each with its resource needs `δ_k`. 

2. **What the agent can do (action `a_t = (P_t, M_t)`).** 
Two kinds of moves: put a waiting task onto a machine (placement `P_t`), or move a running task from one machine to another to free up space (migration `M_t`). 

3. **What happens next (transition `P(s_{t+1} | s_t, a_t)`).** 
Partly predictable, placing a task fills that machine by a known amount. Partly random. Tasks finish and free resources at unpredictable times, running loads wobble up and down, and new tasks keep arriving. 

4. **How the agent is scored (reward, unified, see §4).**
```
R(s_t, a_t) = λ₁ U(H_{t+1}) − λ₂ p_t F_t Δt − λ₃ γ_t F_t Δt − λ₄ [C(M_t) + Ψ(s_{t+1})]

```
Reading left to right: points for packing machines well (`U`), minus this hour's electricity bill, minus this hour's carbon, minus a penalty for shuffling tasks around too much (`C`) and for running machines dangerously close to full (`Ψ`). 


**!!!3 Changes applied based on James's v1:** 
(1) penalty `L` → `Ψ`; 
(2) weights → `λ`; 
(3) cost and CO₂e terms added - v1's reward only scored packing quality, churn, and overload risk, so it wasn't matching the same four objectives as Layer A.

---

## 3. Cross-layer mapping

### 3.1 State ↔ parameters/variables 

##### Variable / Naming Alignment Table

| Layer B (host)                  | Layer A (facility)                         | Coupling                                                                                                                                 |
| ------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `H_t` power column: `h_{i,pow}` | effective IT load `L_t`                    | `L_t = Σ_i h_{i,pow} · P_i^rated` — needs per-host rated power `P_i^rated` [approximated; **owner: James**, §6-Q2]                       |
| Queue `Q_t` admitted this step  | recovered capacity `r_t`                   | `r_t = Σ_{k placed at t} δ_k^pow`. Layer A's `r*_t` is an **admission budget** handed to Layer B: placements at step t may not exceed it |
| workload departures / deferrals | curtailment `d_t`, share `q_t`             | aggregate power freed by deferring/departing workloads                                                                                   |
| (no thermal state in MDP v1)    | `T^in_t` surrogate, levers `c_t, u_t, s_t` | Layer A owns cooling levers; Layer B treats the resulting `P_cap`-and-thermal-feasible envelope as given (§3.2)                          |

**Stranded capacity, one definition:** 
Layer B's implicit view (one resource at 100 % strands the rest of the host) and Layer A's denominator `D` are unified as: `D` = harness-computed kWh of capacity unusable at baseline due to 
(i) multi-resource fragmentation (host level) and 
(ii) power/thermal headroom limits (facility level). 

Harness owns the computation (unchanged from Week 2); 
This spec only fixes that **both layers report against the same `D`**.

### 3.2 Constraints ↔ MDP mechanics

Rule: **hard physical/contractual constraints are enforced in the MDP by action masking, not reward penalties**; only soft risk terms live in `Ψ`. (Penalty-only enforcement lets a learned policy trade SLA violations for reward, not acceptable per NOI-213 honesty norms.)

| Layer A constraint                             | Layer B enforcement                                                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| §3.1 power cap `F_t ≤ P_cap`                   | mask placements that would push `Σ_i h_{i,pow} P_i^rated + C_t` over `P_cap`; per-host `h_{i,j} ≤ 1`                           |
| §3.2 thermal redline `T^in_t ≤ 28 °C`          | mask via the same aggregate surrogate [approximated] — no host-level thermal model in v0                                       |
| §3.3 SLA floor `L̄_t(q_t/100) − d_t ≥ L^SLA_t` | hard mask on deferring SLA-protected workloads; *proximity-to-100 %* risk stays soft in `Ψ` (James's OOM/throttling rationale) |
| §3.4 action windows `W`, `z_t`                 | time-based mask: **migrations `M_t` allowed only for `t ∈ W`**. Same windows, same source text                                 |
| §3.5 N+1 redundancy `κ_t ≤ ρ κ^max`            | inherited through the thermal/power envelope (Layer A owns cooling)                                                            |
| §3.6 ramp `\|L_t − L_{t−1}\| ≤ R`              | soft: migration churn cost `C(M_t)` is the host-level analogue; both feed `λ₄`                                                 |

### 3.3 Time structure

One Layer A horizon (T = 24, Δt = 1 h) = **one RL episode**. 

MDP decision epochs may subdivide an hour (placement decisions arrive with the queue); 

Layer A's hourly setpoints `(c_t, u_t, s_t, q_t)` and budget `r*_t` are constant within the hour and act as the MDP's per-step envelope.

---

## 4. Objectives ↔ reward

Layer A keeps **ε-constraint** as primary (D4): `max f₁ s.t. f₂ ≤ ε₂, f₃ ≤ ε₃, f₄ ≤ ε₄`. Sweeping ε generates the Week 4 Pareto front, including non-convex segments.

The unified reward of §2 is the per-stage **weighted-sum bridge** already anticipated in Yunqi §4: `Σ_t R(s_t,a_t) = λ₁f₁ − λ₂f₂ − λ₃f₃ − λ₄f₄` under the mapping `U ↦ r_t Δt` (per-stage recovery), `C(M_t)+Ψ ↦ f₄` (risk proxy: churn + setpoint deviation + proximity risk). Two consequences both teams sign up to:

1. **Weighted sum reaches only the convex hull** of the Pareto front. RL policies (trained on some fixed `λ`) are therefore evaluated **against ε-constraint points, not against other weighted-sum solutions** (Yunqi §4 note, now binding).
2. Any reported RL result must state its `λ` vector alongside the scenario config, same provenance discipline as the surrogate coefficients. 这是因为换一组 λ 结果就不一样，不写明就没法复现、没法比。

`U(H_{t+1})` may additionally include James's alignment shaping (negative variance / cosine similarity between residual capacity and demand). Shaping terms must be reported separately from the headline metric, they exist to help learning, not to redefine the mission metric.

**Headline metric for both layers: recovery % = f₁ / D.**

---

## 5. Evaluation protocol (shared)

1. Same `scenario_config` feeds both layers (parameters of Yunqi §1 + host inventory `N, M, P_i^rated` + arrival process for `Q_t`).
2. **Oracle bound:** Layer A MILP solved with perfect foresight of the realized episode gives an upper bound on episode recovery; RL performance is reported as % of oracle.
3. **Pareto comparison:** RL policy outcomes `(f₁..f₄)` plotted against the ε-constraint front (Week 4).
4. Toy instance = windows fixed (`z_t` given) → Layer A is a pure LP; Layer B runs on the same instance with arrivals switched off to validate the certainty-equivalence claim of §0 (returns should match LP optimum within tolerance).
5. All four objective values + recovery % reported per run; surrogate coefficients and `λ` disclosed.

---

## 6. Divergences resolved & open questions

**What should be resolved in this spec:** 
	- notation collisions (§1); 
	- reward missing cost/CO₂e (§2); 
	- hard-vs-soft constraint handling (§3.2); 
	- weighted-sum vs ε-constraint evaluation (§4); 
	- shared `D` and headline metric (§3.1, §4).


| #   | Question                                                                                                                                                                                                                        | Owner |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| Q1  | The mock Blackbox exposes only the 5 facility levers. **No placement/migration actions exist in the mock**. Layer B needs either a host-level simulator v0 or an aggregation shim. Raise with Leo before Week 3 simulator work. | Leo   |
| Q2  | Per-host rated power `P_i^rated` and the `h_{i,pow} → kW` mapping: not in any mock field [approximated]. Propose synthetic host inventory in scenario config.                                                                   | James |
| Q3  | Stochastic arrival process for `Q_t`: distribution family + parameters go in scenario config (Week 5 stress tests will vary them).                                                                                              | James |
| Q4  | Water: still zero mock representation; stays out of v0 for both layers (D6), pending Leo's memo answer.                                                                                                                         | Leo   |

---

## 7. Assumptions inherited unchanged

From Yunqi §5: 
(1) synthetic affine surrogates (report coefficients with every result); 
(2) hand-parsed free-text constraints with provenance tags; 
(3)`P_cap`, `D` are scenario inventions → check whether these are reasonable through Week 5 sensitivity analysis. 

From James: 
(1) Live-migration cost is real (bandwidth + degradation), kept as `C(M_t)`; 
(2) OOM/throttling risk near 100 % utilization, kept in `Ψ`.
