# Checklist for James (from the Joint Problem Spec, 2026-07-16)

**Doc edits (v1 → v2)**

- [ ] Rename: `L` → `Ψ`, `α β γ` → `λ₁..λ₄`, `q_k` → `v_k` (§1)
- [ ] Reward: add cost + CO₂e terms (unified formula in §2)
- [ ] Hard constraints (power cap, thermal, SLA, windows) → action masking, not penalties; only soft risk stays in `Ψ` (§3.2)

**Modeling**

- [ ] Define `P_i^rated` so `L_t` / `F_t` are computable from your state (Q2)
- [ ] Placements per hour must fit Layer A's budget `r*_t`; migrations only in windows `W` (§3.1–3.3)
- [ ] Specify the `Q_t` arrival process for scenario config (Q3)

**Reporting**

- [ ] Every result: disclose `λ`, report `f₁..f₄` + recovery % (= f₁/D), compare vs ε-constraint points / % of oracle (§4–5)

**Together**

- [ ] Validation run with Yunqi: toy instance, no arrivals → should match LP optimum (§5.4)
- [ ] Ask Leo before Week 3: mock has no placement/migration actions (Q1)
