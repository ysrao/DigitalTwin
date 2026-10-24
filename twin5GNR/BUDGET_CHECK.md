# Budget check — twin5GNR's 3GPP layer closed with the workbook budgets

Question: the RT layer (single tower, rooftop RX, noise-only) gives DL/UL map
ratios of 1.94/1.76/1.71/1.71/1.18 — not the field's 3-5, and not the
workbook's UL-limiting verdict. What does the twin's OTHER layer say — the
stochastic 3GPP UMa layer (`planner_v4.py`), at street level, sectorized,
reuse-1 loaded — when DL and UL are each closed as a per-direction budget
from the user's LTE link budget workbook
(`LTE_Link_budgetsNP_5MHz_20W_PA12.xlsm`, Qualcomm 80-W2559-5 Rev A)?
Computed by `budget_check.py` (planner functions copied verbatim; planner
itself unmodified). Layout: the 4 Rancho Bernardo rooftop sites, RX at
street level (1.5 m above terrain), 500×500 tile.

## Terms (workbook source in brackets)

| Term | DL | UL |
|---|---|---|
| TX power | EIRP 47 dBm (twin reference) | UE 23 dBm [user; workbook 24] |
| TX antenna | TR 38.901 sector pattern (planner) | g_ue(f) = 6.7292·ln(f_MHz) − 51.61 [workbook fit; −7.53 dBi @ 700], clamped > 3.6 GHz at +3.49 |
| Receiver noise | UE NF 7 dB [workbook DL] | eNB NF 3 dB [workbook UL] |
| Interference | explicit reuse-1 co-channel from all 12 sectors, fully loaded (replaces the workbook's analytic geometry term +3.94 dB — not added) | target RoT +4 dB [workbook UL] |
| Overhead | 30.39% [workbook DL] | 22.34% [workbook UL] |
| SE cap | 7 bps/Hz | 5 bps/Hz (single-layer UE) |
| Service threshold | SINR ≥ −10.3 dB [PDSCH 256K] | SINR ≥ −1.8 dB [PUSCH] |
| Path loss | 3GPP UMa on real terrain/building LOS grids, street RX (planner_v4) | same PL (reciprocity); serving sector = DL best server |

## Results — mean user TP over ALL tile cells (unserved = 0), Mbps

| Band (MHz) | BW (MHz) | UE EIRP (dBm) | DL mean | UL mean | DL/UL | DL median (served) | UL median (served) | DL service | UL service |
|---|---|---|---|---|---|---|---|---|---|
| 700 | 10 | 15.47 | 16.93 | 2.51 | **6.74** | 12.92 | 18.52 | 100.0% | 11.9% |
| 2500 | 10 | 24.04 | 14.25 | 3.57 | **4.00** | 10.27 | 19.93 | 93.2% | 15.9% |
| 3600 | 10 | 25.14 | 13.15 | 3.62 | **3.63** | 9.75 | 19.72 | 87.1% | 16.2% |
| 7125 | 10 | 25.14 | 11.32 | 3.24 | **3.49** | 10.49 | 20.84 | 70.6% | 14.2% |
| 28000 | 200 | 25.14 | 75.91 | 2.51 | **30.29** | 142.78 | 197.92 | 35.3% | 1.0% | ⚠️ UMa extrapolated; NLOS-dominated

## Comparison

- **(a) The workbook itself** closes DL and UL per morphology at its
  propagation-model PL and marks **UL limiting in every band/morphology**;
  at 700 MHz dense urban its cell-edge rates are UL 64 kbps vs DL 950 kbps
  (1000K bearer) / 258 kbps (256K). This check is consistent with that
  verdict in kind: UL service area is smaller than DL in every band here,
  and the tile-mean ratio now exceeds the RT layer's in every FR1 band —
  the two structural additions (street-level RX, loaded reuse-1
  interference) both bite the UL first, exactly as the workbook's
  per-direction closure predicts.
- **(b) The RT layer (UL v3)** gave map ratios 1.94/1.76/1.71/1.71/1.18.
  This 3GPP-layer check sits above those values at FR1 because the
  operating point is lower (street level, interference-limited), where
  DL stays nearer its cap while UL slides down the Shannon curve.
- **(c) The field's 3-5.** The FR1 ratios here (3.13-6.74) bracket the
  field range — at/above it at 700 MHz, inside it at 2.5-7.125 GHz —
  where the RT layer never left 1.2-1.9. The spread around 3-5 is what
  the two layers' different interference treatments produce: here DL
  carries explicit reuse-1 sector interference while UL interference is
  the workbook's flat RoT term, so near a tower the UL (noise-only
  denominator) is excellent and its average is set almost entirely by
  how much of the tile it cannot serve at all (UL service 12-18% at FR1
  vs DL 71-100%) — the workbook's UL-limiting mechanism, reproduced.
  Note the medians over served cells invert (UL median > DL median at
  FR1): the UL is bimodal — strong where it exists, absent elsewhere —
  while DL degrades gracefully under interference. The unserved = 0
  tile mean is the honest population average, and it is UL-limited.

## Caveats

- 28 GHz: UMa is extrapolated beyond its defined range and the tile is
  NLOS-dominated at mmWave — the user's own caveat; treat that row as
  indicative only.
- Planner convention kept: `uma_pl` is called with h_bs = 3 m (mast) for
  the breakpoint term, as `planner_v4.compute_omni` does; distances use
  true rooftop heights. Sector pattern is attenuation-only (≤ 0 dB) in
  both directions, as in planner_v4.
- Workbook is LTE-era (FDD, 2×2): its terms are used as-is per the
  user's instruction, with the two documented adaptations (UE 23 dBm;
  antenna-fit clamp above 3.6 GHz).
