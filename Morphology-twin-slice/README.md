# Urban RAN Digital Twin — build workspace

Implements the design in `Urban_RAN_Digital_Twin_Brief_Design_v2.docx`:
interactive, offline multicell slice-policy evaluation using NVIDIA Sionna,
with a separately pretrained numerical RAN predictor. Tokenization excluded.

## Build stages (per brief §4)

| Stage | Goal | Gate |
| --- | --- | --- |
| A Platform check | Reproducible Sionna install; multicell example; feature inventory; resource profile | Identify missing packet/PRACH/slice functions and exact supported NR assumptions |
| B Interactive baseline | Heterogeneous multicell (macro + small cell, FDD+TDD carriers) scenario editor, persistent runs, slice budgets, rule-based borrowing | PRB accounting, queue conservation, interference response, overload behavior pass checks |
| C Pretrained twin | Versioned corpus; frozen checkpoint; six-step KPI forecasts; explicit uncertainty | Held-out prediction + calibration beat simple baselines; no area/time leakage |
| D Optimization study | Constrained allocation, adaptation curves, burst tests, downloadable comparisons | SLA/utilization trade-offs vs matched baselines; compute overhead quantified |

## Layout

- `src/urtwin/sim/` — Sionna-based reference simulator wrappers (SYS + PHY components)
- `src/urtwin/scenarios/` — scenario schema, generators, CSV/JSON import (cell locations, HetNet layout, FDD+TDD carriers, mixed mobility)
- `src/urtwin/learning/` — corpus generation, predictor training (temporal vs topology-aware Transformer), calibration
- `src/urtwin/opt/` — constrained model-predictive slice-budget allocation, reactive baselines, fallback policy
- `src/urtwin/api/` — small API + job queue (interface ↔ workers)
- `configs/` — pinned environment, scenario YAML examples
- `docs/input_formats.md` — standard inputtable formats: carrier topology CSV (WGS84 lat/lon) + scenario JSON
- `examples/` — `carrier_topology_example.csv`, `carrier_scenario_example.json` (real-topology mode)
- `tests/` — unit/integration tests incl. PRB accounting and queue-conservation checks
- `reports/` — Stage A–D reviewable outputs (feature inventory, resource profile, comparisons)

## Environment

Virtualenv in `.venv/` (system Python is PEP 668 externally managed).
Sionna runs on CPU here (no GPU on this VM); keep UE counts and batch sizes small.

```bash
.venv/bin/python -c "import sionna; print(sionna.__version__)"
.venv/bin/pytest tests/ -x -q
```

## Current status

- [x] Stage A: install + multicell smoke test + feature inventory → `reports/stage_a_platform_check.md` (GO for Stage B)
- [x] Stage B: baseline simulator (HetNet, FDD+TDD, mobility/HO, slicing, PRACH) → `reports/stage_b_baseline.md` — gates pass
- [x] Stage C: corpus_v1 (32 runs) + GRU predictor, 6-step forecasts w/ uncertainty → `reports/stage_c_predictor.md` — beats baselines on served traffic; calibration needs post-hoc fix
- [x] Stage D: gray-box action-conditioned predictor v5 + twin-assisted MPC vs reactive/fixed on truly matched (deterministic-channel) realizations → `reports/stage_d_comparison.md` — twin converges to reactive-like allocations (−0.16%); twin reframed as KPI trade-off laboratory, with per-claim code references for verification
