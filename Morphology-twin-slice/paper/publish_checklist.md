# Publishability Checklist — assessed 2026-10-03

Status key: DONE / PARTIAL / TODO. Effort: S/M/L.

## 1. Scale of evidence — TODO (L)
*Checklist (edited):* 4–6 morphologies (at least one from a real carrier CSV
via the topology import), 20+ seeds per morphology, 95% CIs, paired tests
across structures.
- Current: 2 synthetic layouts × 5 seeds. Pilot now reports 95% bootstrap CIs (v3 §V) and Wilcoxon signed-rank on paired per-run earnings (20% vs 10%: p=0.17; 20% vs 30%: p=0.08 — not significant at pilot scale).
- Compute is fine: 90 runs ≈ 6.5 min → 6 morphs × 20 seeds × 9 structures ≈
  80 min on the local PC.
- Blocker: a *real* carrier topology has to come from a carrier — the import
  path is built and tested (SF example), but we cannot fabricate "real".
- Edit made: clarified "20+ seeds **per morphology**" and that the real
  topology is an external dependency.

## 2. Correct 38.901 scenarios per tier + indoor/O2I — PARTIAL (M)
*Checklist (edited):* UMa for macros, UMi for pico/femto; add an indoor UE
share (suggest 20–30%) with O2I penetration (low/high).
- DONE 2026-10-03: tier-matched channels implemented — macro→UMa,
  pico/femto→UMi-Street Canyon, one provider per (carrier, tier); Stage B
  gates pass. 5G-Advanced radio: FR1 (2.1/3.5 GHz, mu=1) + FR3 7.125 GHz /
  200 MHz / mu=2; per-carrier-frequency coupling (Sionna switches O2I
  handling above 6 GHz — a single-frequency shortcut would be ~6 dB off).
- Indoor: still outdoor-only. 38.901 UMa assumes 80% indoor; Stage-A found
  zero-gain indoor links in Sionna 2.2.0. Fresh Oct-3 probes show plausible
  O2I values at short range but elevated clipping at long range — discrepancy
  unresolved, so indoor stays a defined re-run experiment, not silently built
  on an unvalidated path.

## 3. Formal model definitions + parameter table — TODO (S)
*Checklist (edited):* formally define the SLA model (deadline-expiry
violations; reliability now measured, see baseline.py), the traffic model
(Poisson arrivals — state rates), and the link abstraction (SINR → bits via
capped Shannon with gap). Add one parameter table (slots, SCS, PRB tables,
traffic rates, PRACH params, slice SLAs).
- Current: all in code/config; not formalized in paper text.
- Mostly a writing task; constants already pinned in configs/.

## 4. Analytical/prior dimensioning baseline + 25 refs — TODO (M)
*Checklist (edited):* compare against at least one analytical dimensioning
baseline — suggest Erlang-C per-slice guarantee sizing from offered load, or
a deterministic LP (minimize weighted violations s.t. PRB totals). Complete
related work to 25+ real references (replace placeholder [5]).
- Current: baselines are allocation policies (reactive, fixed-equal), not
  dimensioning methods. 6 references.
- The analytical baseline is the intellectually load-bearing item: it tests
  whether the twin beats textbook dimensioning.

## 5. Control study: structured traffic or demote — PARTIAL (S)
- Current: Poisson traffic; the study is already demoted to Section VI
  "Supporting Theory".
- Remaining: either rerun with diurnal/periodic traffic or formally demote
  to a labeled sanity check (one paragraph + the v7 tie numbers). Recommend
  the latter — cheapest and honest.
- Edit made: demotion counts; full rerun optional.

## 6. Table 1 + earnings sensitivity + "6G" — PARTIAL (S/M)
*Checklist (edited):* Table 1 needs 95% CIs (depends on item 1); add an
earnings **sensitivity analysis** — sweep the assumed prices/penalties and
show the interior optimum persists (or map where it breaks).
- Table 1 itself is fixed (percentages, header, v3 earnings model).
- Sensitivity runs are cheap (reuse stage_e data, recompute).
  - **Decision made 2026-10-03:** "6G" removed from the title — no 6G-specific radio configs were applied (FR1 config only).
  - **Update 2026-10-03:** radio is now 5G-Advanced (3GPP Rel-18/19) — FR3 7.125 GHz / 200 MHz / mu=2 added with per-carrier coupling; folded into the re-run plan. Title stays generation-free.

## 7. Code + data with DOI — TODO (awaiting user)
- Standing plan: paper first → Gymnasium env → GitHub + Zenodo DOI after
  user approves. Nothing published yet.

## Long poles
1 (needs carrier data), 2 (channel-model code), 4 (analytical baseline + lit
review). Items 3, 5, 6 are writing/cheap compute. Item 7 is a decision.
