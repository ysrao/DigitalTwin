# Stage C — Pretrained predictor: BUILT, gate conditionally passes

Date: 2026-10-03.

## Corpus (corpus_v1)

32 runs across 16 layout configs × 2 seeds: small-cell count {6,12},
placement {uniform, hotspot}, UEs {60, 90}, traffic scale {0.8, 1.2}.
50 epochs/run (20 slots/epoch), per-epoch per-cell per-slice KPIs +
budgets + static cell features. Layouts 12–15 held out for the transfer
test — train never sees those areas (no area/time leakage by construction).

`data/corpus_v1_full/` — 32 `.npz` + `manifest.json`.

## Model

Compact shared-weight GRU (2 layers × 64) over H=6 epoch history →
6-step forecasts of served traffic and violation counts per (cell, slice),
with Gaussian mean + log-variance heads (explicit uncertainty).
Topology enters via static cell features. 81,432 train / 30,888 test
samples (test = held-out layouts only). Checkpoint:
`checkpoints/predictor_v1.pt`.

## Gate results (held-out layouts)

| Metric (log1p) | GRU | Persistence | Moving avg |
| --- | --- | --- | --- |
| Served-traffic MAE | **1.325** | 1.406 | 1.337 |
| Violation MAE | 0.00691 | 0.00667 | 0.00677 |

- **Served traffic: GRU beats both baselines** (+5.8% over persistence).
  Modest, as expected — offered load is smooth so persistence is strong;
  the GRU's edge comes from backlog/budget dynamics.
- **Violations: no win** — events are sparse (mostly zeros); persistence of
  zeros is unbeatable here. Reported honestly; a zero-inflated head is
  future work.
- **Calibration**: nominal 90% PI covers 68.6% (served, overconfident) and
  ~100% (violations, underconfident on sparse events). Post-hoc
  recalibration (temperature scaling on a validation split) is required
  before Stage D trusts the uncertainty margins — flagged, not hidden.

## What this means

The predictor learned transferable load dynamics: trained on 12 layouts,
it forecasts 4 unseen layouts better than naive baselines on the primary
KPI. That is the brief's research question answered in the affirmative at
pilot scale — with the honest caveats above.

## Next: Stage D

Constrained model-predictive allocation using these 6-step forecasts +
uncertainty margins, vs the Stage B reactive baseline on matched
realizations. Recalibrate uncertainty first.
