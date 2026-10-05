# Changelog

## v2.0 (2026-10-04): antenna fix, all results regenerated

- FIX `src/urtwin/sim/channel.py`: macro cells use the TR 38.901 element (65 deg HPBW, 8 dBi)
  oriented to each sector azimuth (bs_orientations); pico/femto use an omni element.
  In v1.0-v1.2 every cell used the 38.901 element facing azimuth 0, with an analytic sector
  pattern added on top. Sectors at 150/270 deg were 14-19 dB weak on their own boresight,
  and small cells were directional (about 20 dB front-to-back).
  Verified: identical boresight loss for all sectors (UMa 126.9 dB, RMa 125.6 dB at 200 m);
  pico loss independent of direction within 0.6 dB.
- `scripts/run_grid.py`: one runner for every experiment (calib_urban, calib_suburban, urban,
  suburban, rural_uniform, rural_profile, ablation, trace), split across workers, resumable.
- Calibration on the corrected channel: load factors 5.0/6.0 for every morphology
  (`data/v2_calibration_choice.json`).
- New results (`data/v2_*.json`): urban, suburban, rural uniform and rural profile grids
  (4 x 360 runs), ablation (180), trace-driven (180), PRACH probe on the corrected channel
  (60-480 UEs), and summaries (`data/v2_summary_*.json`).
- `scripts/run_prach_v2.py`, `scripts/figs_v2.py` (Figs. 1-5; Fig. 1 now has URLLC, eMBB,
  mIoT and total panels).
- v1.x results moved to `data/v1_superseded/`, and old figures to `paper/figs/v1_superseded/`.
- Paper: `paper/RAN_Digital_Twin_Slice_Dimensioning_v8.docx`. Section VI (predictor/MPC) was
  not regenerated; it still uses v1.x results.

## v1.2 (2026-10-04)

Adds the rural morphology, so the twin now covers urban, suburban and rural.

- `scripts/run_dimensioning_suburban.py`: new options `SUB_ISD` (inter-site distance),
  `SUB_SC` (small-cell count; 0 for rural), `SUB_MORPH` (label), and
  `RURAL_PROFILE=1` (UE mix 30/20/50 eMBB/URLLC/mIoT, eMBB arrival rate x0.5).
- `scripts/calibrate_suburban.py`: `CAL_SEEDS` sets the calibration seed count.
- Rural, traffic profile (ISD 1732 m, no small cells, RMa, load 5.0/6.0 with profile):
  `data/stage_e_dimensioning_suburban_ruralprofile_w{0,1}.json`
- Rural, uniform load (same load 8.0/9.0 and seeds as the recalibrated suburban grid):
  `data/stage_e_dimensioning_suburban_rural_w{0,1}.json`
- Rural calibration (5 seeds): `data/calibration_suburban_rural_*.json`,
  choice in `data/calibration_choice_rural.json`
- Summaries: `data/summaries/rural_profile.json`, `data/summaries/rural_uniform.json`
- `scripts/fig_morphology.py`: five-dataset figure (Fig. 5)
- Paper: `paper/RAN_Digital_Twin_Slice_Dimensioning_v7.docx` (three morphologies; Appendix A)

## v1.1 (2026-10-04)

Adds a second morphology, suburban, with the TR 38.901 RMa macro channel.

### Simulator (opt-in, urban results unchanged)
- `src/urtwin/sim/channel.py`: RMa model; `tier_scenario(type, propagation)` maps
  macro cells to RMa when `radio.propagation: RMa` (street width 20 m,
  building height 10 m). Small cells stay UMi-Street Canyon.
- `src/urtwin/sim/baseline.py`: per-tier scenario selection uses `tier_scenario`.
- `src/urtwin/sim/hetnet.py`: optional `scenario.macro_height_m` (default 25 m).
- Verified: the published urban run (layout 0, seed 3000, 30%/10%) reproduces
  bit-identically after the change.

### New experiments
- `scripts/run_dimensioning_suburban.py`: 9 structures x 2 layouts x 20 seeds,
  ISD 1000 m, 35 m macro masts, RMa. Two grids:
  - same traffic as urban (load 5.0/6.0) -> `data/stage_e_dimensioning_suburban_w{0,1}.json`
  - recalibrated load 8.0/9.0 -> `data/stage_e_dimensioning_suburban_heavy_w{0,1}.json`
- `scripts/calibrate_suburban.py`: load sweep with the urban calibration criterion
  -> `data/calibration_suburban_*.json`, choice in `data/calibration_choice_suburban.json`.
- `scripts/analyze_dimensioning.py`: means, bootstrap CIs, paired Wilcoxon tests,
  exchange rate (ratio of mean paired changes), earnings and lambda sweep
  -> `data/summaries/*.json`.
- `scripts/fig_morphology.py`: urban-vs-suburban figure -> `paper/figs/morphology.png`.

### Paper
- `paper/RAN_Digital_Twin_Slice_Dimensioning_v6.docx`: adds Section V-G and
  corrects several urban statistics against the result JSONs (exchange rate
  21.0 [11.9, 45.6]; paired earnings gaps $19.1 and $60.4 per run; lambda
  break-points 14.4 and 38.5; 27-cell layout).

## v2.1 (2026-10-05) — revision r4
- Added scripts/run_revision_r4.py (long: 10,000-slot urban runs at 30%/10% and 30%/30%, 2 layouts x 10 seeds; hour: stationary trough/median/peak hour snapshots, 2 layouts x 10 seeds) and scripts/analyze_revision_r4.py.
- Results: data/r4_long_w*.json, data/r4_hour_w*.json, data/r4_summary.json (Tables I and II of paper v2).
- The time-compressed 24 h ramp (v2_trace_*.json) is retained but no longer used for claims.
- Paper: paper/IEEE-RAN-Digitaltwin-v2.{docx,pdf}, built by paper/build_r4.py.
