# Stage D — The Twin as a Slicing Laboratory: KPI Trade-offs Under Controlled Allocation

**Date:** 2026-10-03 (revised; v7 results with deterministic channels) ·
**Status:** Complete — the twin's purpose is characterized below.

## 1. Purpose (reframed)

This twin was not built to crown "predictive" over "reactive." It was built
as a **controlled laboratory for RAN slicing decisions**: change how PRBs
are divided among eMBB/URLLC/mIoT, hold every channel, arrival, and mobility
realization fixed, and measure what happens to each KPI. The
predictive-vs-reactive study was one experiment inside that lab — and its
most valuable output is not a winner but a **map of where the optimum sits
and how KPIs trade against each other**.

## 2. What the laboratory established

All results below are from **truly matched realizations**: same layout,
same seed ⇒ identical traffic, mobility, *and* channel fading across
policies (determinism verified bit-identical across repeat runs; see §6).
4 layouts (2 train + 2 held-out) × 2 seeds × 3 policies, 1000 slots each,
heavy-load regime (2–3× nominal arrivals) where slice budgets bind.

### 2.1 The head-to-head (v7, definitive)

| Policy | Served (kb) | Violations | kb / PRB | p5 UE thr (kbps) |
|---|---|---|---|---|
| Fixed (equal thirds) | 21,435 | 151.9 | 0.0470 | 4.83 |
| Reactive (backlog-driven) | 22,408 | 92.0 | 0.0350 | 4.76 |
| Twin-assisted MPC | 22,372 | 95.5 | 0.0358 | 4.74 |

Per-run, twin vs reactive served: −47, −22, +6, −18, −116, −2, −64, −22 kb
(−0.16% mean). **With an action-sensitive predictor, bold candidate
allocations, and identical channels, the optimizer converges to
reactive-like allocations.** The twin independently rediscovers that
backlog-proportional allocation is near-optimal for throughput — consistent
with MaxWeight throughput-optimality theory.

### 2.2 Where slicing decisions *do* move KPIs (the trade-off map)

A controlled single-variable sweep (same seed, fixed allocations) showed
budgets matter enormously when they bind:

| Allocation | Served (kb) | Violations |
|---|---|---|
| Equal thirds | 18,139 | 143 |
| eMBB-heavy (60/25/15) | 20,379 (+12%) | 12 (−92%) |
| URLLC-heavy (20/60/20) | 16,627 (−8%) | 198 (+38%) |

And across policies: conservative budgeting buys **+17–34% spectral
efficiency** (kb/PRB) at the cost of edge-user throughput; the fixed
equal-split policy is the most "efficient" (0.0470) and the worst on
violations (151.9). **This trade-off surface — not a leaderboard — is the
twin's product.**

### 2.3 Burst behavior

Under sudden 3× eMBB surges (epochs 20–30), twin and reactive tie
(±2% served, violations within ±7%). Nothing trained on history foresees
an unpredictable surge; both policies ride it out on real backlogs.

## 3. How the result was earned (diagnosis chain)

1. **v1 predictor ignored its action input** (±2σ budget shift →
   0.02-unit forecast change). Cause: half the training budgets were
   reactive — a deterministic function of backlog already in the history.
2. **Fix 1 (data)** — 100% randomized budgets: sensitivity stayed ~0.01.
   The late-fusion architecture let the network zero the action path.
3. **Fix 2 (architecture)** — gray-box `served = min(demand, budget × SE)`
   with auxiliary supervision of spectral efficiency from observed
   `served/prb_used`: sensitivity rose 10×, predicted SE realistic
   (~108 bits/PRB), budgets predicted binding 31% of steps.
4. **Fix 3 (methodology)** — Sionna's channel fading was unseeded, silently
   breaking "matched realizations." Fixed via `sionna.phy.config.seed`
   per compute call; determinism verified bit-identical.
5. **Fix 4 (action space)** — an oracle rollout showed ±15% tilts move true
   6-epoch outcomes by <1% (below model noise). Candidates were widened to
   bold reallocations (60%-to-one-slice, all-min, equal, demand-prop).

Even after all four fixes, the twin ties reactive — which *is* the finding:
in this regime the optimum is flat around backlog-proportional allocation,
and the twin proves it instead of assuming it.

## 4. Limitations (for open publication)

- 2 seeds/layout: pilot-grade; publication needs 5–10 + confidence intervals.
- Poisson arrivals have no predictable temporal structure; a fair
  proactive test needs diurnal/periodic patterns.
- MPC scoring weights (λ=4) and candidate design were not ablated.
- Indoor UEs excluded (Stage A Sionna edge case); all-outdoor UMa.

## 5. Compute overhead

MPC decision ≈ 10–27 ms/epoch on CPU; 1000-slot run 4–6 s vs 3–4 s
reactive. Negligible for offline study.

## 6. Verification — code outside this publication

Every claim above is reproducible from the workspace code and data; the
report itself contains no unverifiable numbers.

| Claim | Code | Data | Reproduce |
|---|---|---|---|
| Deterministic matched realizations | `src/urtwin/sim/channel.py` (`config.seed`), `baseline.py` (seed → provider) | — | run any sim twice, same seed ⇒ bit-identical KPIs |
| v7 head-to-head table | `src/urtwin/control/compare.py`, `src/urtwin/control/mpc.py` | `data/stage_d_comparison_v7.json` | `compare.py --ckpt checkpoints/predictor_v5.pt --scales 2.0,3.0 --out …` |
| Gray-box predictor + sensitivity | `src/urtwin/learning/models_structured.py`, `train_structured.py`, `dataset_v2.py` (aux) | `checkpoints/predictor_v5.pt`, `data/corpus_v4/` | `train_structured.py --corpus data/corpus_v4` |
| Budgets matter (sweep) | `baseline.py` (`set_budgets`, `budget_policy='external'`) | — | §2.2 script pattern in `compare.py::run_fixed` |
| Oracle candidate-spread test | `src/urtwin/control/oracle_test.py` | — | `python oracle_test.py` |
| Recalibrated uncertainty (v2, 89.6% PI90) | `src/urtwin/learning/train_v2.py` | `checkpoints/predictor_v2.pt` | `train_v2.py --corpus data/corpus_v2` |
| Corpora | `src/urtwin/learning/corpus.py` | `data/corpus_v2/`, `v3/`, `v4/` (+ manifests) | `generate(..., budget_policy='explore', traffic_scales=(2.0,3.0))` |

Pipeline order: `corpus.py` → `train_structured.py` → `compare.py`.
Checkpoints: `checkpoints/predictor_v2.pt` (black-box, recalibrated),
`predictor_v5.pt` (gray-box, v7). The twin-assisted controller falls back
to reactive when forecast uncertainty exceeds its calibrated p90
(measured fallback rate in v7: 0% — it acted on its forecasts throughout).
