# Uplink (UL) model for twin5GNR — v3, the link-budget method

**Everything on this page is MODELLED, not measured.** The uplink layer
closes a per-direction link budget — the structure of the user's own
LTE link-budget workbook — on top of the downlink's ray-traced path
loss. Generator: `ul_post.py`. Outputs:
`data_twin_tower_<site>_<band>_ul.npz` (`ul_rsrp`, `ul_snr`,
`ul_cap_mbps`, `los`, `meta`) for all 20 site-bands, plus
`ul_stats.json`.

## Why v3 (v1 and v2 were rejected, and rightly)

- **v1** was a reciprocal shortcut: the DL composition with the
  transmit power swapped (23 dBm), an idealised 0 dBi UE antenna at
  every frequency, and ad-hoc noise terms (BS NF 4 dB + 3 dB rise,
  DL's 7 bps/Hz cap). No per-direction budget was ever closed, and it
  produced DL/UL throughput ratios of ~1.0 — not credible.
- **v2** tuned two constants (interference rise 10 dB, UL cap 5) until
  the output moved toward the field's 3–5 ratio (it reached ~1.5).
  Tuning constants to fit an expected answer is not a link budget, so
  v2 was rejected too.
- **v3** changes no path loss (the RT PL was audited and cleared —
  below) and instead adopts, term by term, the budget structure of
  the workbook the user supplied. Nothing in v3 was adjusted to chase
  the 3–5 field ratio; where the twin's numbers land is reported as
  computed.

## The workbook and its method

The reference is the user's LTE link-budget workbook, **Qualcomm
80-W2559-5 Rev A** (`LTE_Link_budgetsNP_5MHz_20W_PA12.xlsm`; sheets
`700_DU`, `UL_Combined`, `Summary`, `Summary table`, `Tables`). Its
method is the classical one:

- Every direction and service (PUSCH, PBCH, RS, PDCCH, PDSCH at
  defined rates) gets its **own budget**: transmitter power expressed
  **per subcarrier** (PSD), the *receiving* end's noise floor and
  required SNR, antenna gains at both ends, and a margin stack
  (building penetration + log-normal fading at a stated cell-edge
  reliability).
- Each budget closes into a **MAPL** (maximum allowable path loss)
  and a final path loss to the cell border, per direction — so DL and
  UL are compared at the *same* propagation loss, with their own
  transmitters, receivers and margins.
- Its `Summary table` marks the **uplink as the limiting direction in
  every band/morphology column it covers** (700 MHz and AWS;
  dense-urban, urban, suburban, rural). Its own service-level edge
  numbers at 700 MHz dense-urban: UL (1 RB PUSCH) 37–74 kbps across
  its three CQI mappings, DL PDSCH-1000k average ≈ 950 kbps — noting
  that this compares a **1-RB UL allocation against a full-channel DL
  service**, an asymmetry that is part of the workbook's service
  definitions, not a like-for-like user rate.

## The v3 budget (per cell, on the map)

The twin keeps its audited ray-traced path loss in place of the
workbook's morphology propagation model. Per cell:

```
PL        = 47 + 0 - RSRP_DL                          [dB, effective]
ul_rsrp   = UE_EIRP(f) - PL + 8 + 12.04
          = UE_EIRP(f) + 10*log10(P_dir) + 12.04      [dBm at the BS]
ul_snr    = ul_rsrp - (-174 + 10*log10(BW) + 3 + 4)   [dB]
ul_cap    = BW * min(5, log2(1 + 10^(ul_snr/10))) * 0.7766   [Mbps]
```

`P_dir` is the same best-sector directional channel power the DL
post-processing composed (TR 38.901 pattern on the stored paths +
4×4 array). The workbook's separate 18 dBi eNB antenna line is **not**
added on top: in twin terms that panel-plus-combining gain already
lives inside this composition (element +8 dBi inside the pattern,
+12.04 dB array) — adding it again would double-count ~18 dB.

**PL derivation, cross-checked.** `PL = 47 − RSRP_DL` removes exactly
the known DL budget terms (47 dBm reference EIRP, 0 dBi UE gain);
internally it is identical to `8 − 10·log10(P_dir)` (residual
0.00 dB — the composition is exact). Against the independent
isotropic sum `−10·log10(Σ pk_pw)` on SE 3.6 GHz (2,139 served
cells): median isotropic PL 107.7 dB vs median derived PL 112.4 dB;
the derived (budget-effective) PL sits a median **+3.18 dB** above
the isotropic sum (mean +3.67, p10 +0.23, p90 +8.12). That offset is
the sector-pattern roll-off on the dominant paths — near zero for
boresight LOS cells (p10), several dB where the serving paths arrive
off-boresight — i.e. the derived PL is the isotropic PL *as the
budget sees it through the sector antenna*, which is the correct
quantity to close a budget against.

### Line items: workbook vs twin v3

| Term | Workbook (80-W2559-5) | Twin v3 | Note |
|---|---|---|---|
| UE transmit power | 24 dBm | **23 dBm** | user-set value; noted difference |
| UE antenna gain | g(f) = 6.7292·ln(f_MHz) − 51.61 dBi | same formula | handset fit: −7.53 dBi @ 700, +1.04 @ 2500, +3.49 @ 3600 MHz |
| UE antenna above 3.6 GHz | (LTE workbook — no such bands) | **clamped to the 3600 MHz value, +3.49 dBi** | explicit adaptation: unclamped, the LTE handset fit runs away (+8.09 dBi @ 7.125 GHz, +17.30 dBi @ 28 GHz) |
| UE EIRP @ 700 MHz | 16.47 dBm (24 − 7.53) | 15.47 dBm (23 − 7.53) | the 1 dB is the TX-power difference above |
| BS receive antenna | 18 dBi − 0.5 dB losses | inside the reciprocal composition (see above) | not added twice |
| eNB noise figure | 3 dB (700 sheets; 2.5 dB on AWS sheets) | 3 dB | |
| Target RoT | +4 dB | +4 dB on the noise floor | replaces v1's NF 4 + IoT 3, v2's NF 4 + IoT 10 |
| Thermal noise | −173.98 dBm/Hz | −174 dBm/Hz | twin convention, 0.02 dB |
| UL overhead | 22.3413% (`UL_Combined`) | 22.34% → payload ×0.7766 | |
| SINR → throughput | CQI efficiency tables (36.942 / vendor) | Shannon form, cap 5 bps/Hz, × 0.7766 | **labelled hybrid** — see below |
| Margins (edge) | BPL 11 dB + log-normal | same stack, two variants | § Edge model |

The one deliberate hybrid: the workbook maps SINR to efficiency
through CQI tables; the twin keeps its Shannon convention (as on the
DL side) and applies the workbook's overhead and a 5 bps/Hz
single-layer UE cap on top. At the workbook's own UL design point
(SINR −1.8 dB) its 36.942 table gives 0.265 bps/Hz where the Shannon
form gives 0.73 — the hybrid overstates low-SINR efficiency, in both
directions' edge math equally; it is used for consistency with the
twin's DL, and stated here rather than hidden.

Because the budget terms now differ per direction, the v1/v2
constant −3.96 dB shift is gone. The UL−DL RSRP shift is
`g_ue(f) − 3.96` dB: **−11.49 dB at 0.7 GHz** (the workbook's handset
antenna is the single largest UL penalty there), −2.92 dB at 2.5 GHz,
−0.47 dB at 3.6 GHz and above (clamped fit). In SNR terms,
`ul_snr − dl_snr = g_ue(f) − 1.96` dB: the BS receiver (NF 3 + RoT 4
= 7 dB over thermal) is quieter than the UE receiver in the DL layer
(NF 9 dB), which buys back 2 dB.

## Edge model (statistics only — not a map)

Mirroring the workbook's cell-border construction, per site-band:

- **Border PL (proxy):** the 90th percentile of `PL` over served
  cells. Labelled proxy: the twin tile is one tower serving a
  2×2 km grid, so its PL tail runs far past any real cell border —
  PL90 here is 148–208 dB, versus the workbook's own border PLs of
  117–129 dB and its RSRP detection threshold of −121 dBm (which at
  the twin's 47 dBm EIRP corresponds to PL ≈ 168 dB).
- **Margin stack (the workbook's):** BPL mean 11 dB + log-normal
  fading, applied to the edge SINR. Main variant (from `700_DU`):
  combined σ = √(10² + 8²) = 12.81 dB at 95% cell-edge reliability →
  **32.06 dB**. Variant (from `UL_Combined`): combined σ = 10 dB at
  90% → **23.82 dB**. (The twin's RX points are outdoor at 6 m, so
  applying the workbook's building-penetration stack is
  conservative; it is the workbook's stack, applied as specified.)
- **UL edge:** a 1-RB (180 kHz) PUSCH allocation carrying the full
  UE power — the workbook's UL service definition. Noise over
  180 kHz with NF 3 + RoT 4; TP = 180 kHz · min(5, log2(1+SINR)) ·
  0.7766.
- **DL edge (comparison only — published DL maps are untouched):**
  the workbook's DL structure at the same border PL: per-subcarrier
  EIRP from 43 dBm over the channel (occupied BW = 0.903 × nominal,
  the workbook's 4.515/5 MHz ratio), UE NF 7 dB, other-cell
  interference tied to the margined own signal at Ioc/Ior =
  +3.94 dB (its geometry −4.4 dB at 90% load), DL overhead 30.39%,
  2×2 MIMO with SU-MIMO gain 1 (the gain the workbook's own rate
  blocks use), Shannon-form SE capped at the workbook's top CQI
  efficiency 5.5547 bps/Hz.

### Edge results (pooled across the 4 sites; TP in kbps)

| Band | PL90 (dB) | UL edge, 95% margin | DL edge, 95% | DL/UL | UL edge, 90% margin | DL edge, 90% | DL/UL |
|---|---|---|---|---|---|---|---|
| 0.7 GHz (n28, FDD) | 152 | 0.118 | 5.88 | 49.8× | 0.786 | 38.63 | 49.1× |
| 2.5 GHz (n41) | 167 | 0.024 | 1.21 | 49.9× | 0.162 | 8.06 | 49.8× |
| 3.6 GHz (n48/n77) | 173 | 0.014 | 0.68 | 49.9× | 0.091 | 4.56 | 49.8× |
| 7.125 GHz (FR3) | 181 | 0.002 | 0.09 | 49.9× | 0.012 | 0.60 | 49.9× |
| 28 GHz (n257) | 200 | 0.00003 | 0.001 | 49.9× | 0.0002 | 0.01 | 49.9× |

**How to read this table — honestly.** At the PL90 proxy border both
directions are tens of dB below their noise floors (edge SINRs run
from −21 dB to −92 dB): neither link closes anything usable, and the
absolute TPs are effectively zero. Deep in that Shannon tail,
SE ≈ SINR/ln 2 for both directions, so PL and margin cancel in the
ratio and it pins to the constant set by the two budgets' per-hertz
terms and the 15 kHz-subcarrier vs 180 kHz-RB normalisation —
≈ 49.9×, in every band, under both margins. That constant is a
property of the two budget structures, not a measured field ratio.
The information the edge exercise actually carries is: (1) the twin
tile extends far beyond a workbook cell border (PL90 148–208 dB vs
the workbook's 117–129 dB borders); (2) at any such over-extended
border the UL is the direction that dies first, by the budget
constant above. The workbook's *own* edge comparison, at *its*
borders, is the meaningful one: UL limiting everywhere, with its
1-RB UL service at 37–74 kbps against its full-channel DL services.

## Map results (viewer-default bandwidths)

Coverage = % of all 2,601 grid cells with RSRP ≥ −90 dBm (no-path
cells count as not covered). Capacities are means (medians in
parentheses) over served cells, single user on the carrier: DL =
BW·min(7, log2(1+SNR)) (the existing DL convention, unchanged);
UL = BW·min(5, log2(1+SNR))·0.7766 (v3).

| Site-band | BW (MHz) | DL cov | UL cov | DL cap mean (med) | UL cap mean (med) | DL/UL | PL90 (dB) |
|---|---|---|---|---|---|---|---|
| NW — 0.7 GHz (n28, FDD) | 10 | 61.2% | 52.7% | 51 (70) | 26 (39) | 1.94 | 152.9 |
| NW — 2.5 GHz (n41) | 10 | 51.0% | 49.2% | 42 (70) | 24 (39) | 1.74 | 167.9 |
| NW — 3.6 GHz (n48/n77) | 10 | 48.2% | 48.1% | 40 (58) | 24 (39) | 1.68 | 172.6 |
| NW — 7.125 GHz (FR3) | 10 | 42.9% | 42.6% | 35 (32) | 21 (29) | 1.64 | 181.9 |
| NW — 28 GHz (n257) | 200 | 31.6% | 31.3% | 212 (2) | 183 (2) | 1.16 | 199.8 |
| NE — 0.7 GHz (n28, FDD) | 10 | 66.1% | 55.7% | 54 (70) | 28 (39) | 1.94 | 147.9 |
| NE — 2.5 GHz (n41) | 10 | 53.1% | 51.9% | 45 (70) | 26 (39) | 1.77 | 164.0 |
| NE — 3.6 GHz (n48/n77) | 10 | 51.1% | 50.8% | 43 (70) | 25 (39) | 1.74 | 168.9 |
| NE — 7.125 GHz (FR3) | 10 | 48.7% | 48.4% | 41 (70) | 24 (39) | 1.75 | 177.7 |
| NE — 28 GHz (n257) | 200 | 44.9% | 44.8% | 309 (192) | 264 (191) | 1.17 | 195.9 |
| SW — 0.7 GHz (n28, FDD) | 10 | 61.6% | 51.2% | 50 (70) | 26 (39) | 1.96 | 160.0 |
| SW — 2.5 GHz (n41) | 10 | 49.1% | 47.8% | 42 (70) | 24 (39) | 1.76 | 175.0 |
| SW — 3.6 GHz (n48/n77) | 10 | 46.6% | 46.4% | 40 (70) | 23 (39) | 1.72 | 181.0 |
| SW — 7.125 GHz (FR3) | 10 | 44.4% | 44.3% | 37 (52) | 22 (39) | 1.72 | 189.4 |
| SW — 28 GHz (n257) | 200 | 39.4% | 39.3% | 287 (27) | 239 (29) | 1.20 | 207.9 |
| SE — 0.7 GHz (n28, FDD) | 10 | 67.4% | 58.6% | 55 (70) | 28 (39) | 1.93 | 148.0 |
| SE — 2.5 GHz (n41) | 10 | 56.6% | 54.6% | 46 (70) | 26 (39) | 1.75 | 163.0 |
| SE — 3.6 GHz (n48/n77) | 10 | 53.0% | 52.6% | 44 (70) | 25 (39) | 1.72 | 168.0 |
| SE — 7.125 GHz (FR3) | 10 | 48.9% | 48.8% | 41 (70) | 24 (39) | 1.73 | 176.7 |
| SE — 28 GHz (n257) | 200 | 44.1% | 43.8% | 297 (140) | 250 (142) | 1.19 | 195.2 |

**Per-band pooled** (means of the 4 site means):

| Band | DL mean | UL mean | DL/UL | DL median | UL median | DL/UL (median) |
|---|---|---|---|---|---|---|
| 0.7 GHz (n28, FDD) | 52 | 27 | **1.94** | 70 | 39 | 1.80 |
| 2.5 GHz (n41) | 44 | 25 | **1.76** | 70 | 39 | 1.80 |
| 3.6 GHz (n48/n77) | 42 | 24 | **1.71** | 67 | 39 | 1.73 |
| 7.125 GHz (FR3) | 38 | 22 | **1.71** | 56 | 36 | 1.54 |
| 28 GHz (n257) | 276 | 234 | **1.18** | 90 | 91 | 0.99 |

## Honest reading, against the field's 3–5

- The map ratio is now **1.7–1.9 at FR1** (was ~1.0 in v1, ~1.5 in
  v2), driven by real budget terms: at 0.7 GHz the workbook's
  −7.53 dBi handset antenna alone costs the UL 8.5–10.4 coverage
  points against DL and halves its mean throughput. That is the
  workbook's own physics, not a fitted constant.
- At 28 GHz the pooled *median* ratio is **0.99** — the median served
  cell's UL throughput equals DL's. The arithmetic: with the clamped
  +3.49 dBi antenna fit and the quieter BS receiver, the UL SNR at a
  median mmWave cell is ~1.5 dB *above* the DL SNR, and at mmWave
  SNRs (both directions far below their caps over most of the grid)
  that outweighs the cap and overhead penalties; the mean ratio
  (1.18) is pulled up by the high-SNR cells near the tower, where the
  5 bps/Hz cap and 0.7766 payload bite. Reported as computed.
- Still short of the field's 3–5 — and v3 does not pretend otherwise.
  What a per-direction budget on a **single isolated tower** cannot
  contain is the loaded multi-cell geometry (neighbour-cell
  interference distributions, scheduler sharing, TDD duty structure)
  that produces the field ratio; the workbook reaches its
  "UL limiting" verdict through MAPL comparisons at its own borders
  and service definitions, not through a single-cell user-rate ratio
  either. The twin's UL is now built the way field budgets are built;
  reproducing field *ratios* remains a planner-layer (multi-site,
  loaded) question, exactly as the provenance note in the README
  scopes it.
- What v3 does establish, in the workbook's own terms: the UL budget
  closes with less margin everywhere; UL coverage is below DL in
  every site-band (decisively at 0.7 GHz); and at any common border
  the UL service dies first.

## Path-loss audit (retained from v2 — why the PL was never the problem)

Ray tracing is reciprocal, so a PL error could not have produced a
DL-only discrepancy — but it was audited anyway, on the SE 3.6 GHz
grid (median PL in distance bins, isotropic channel power over the
stored paths): LOS cells track FSPL and 3GPP UMa LOS within ~1–2 dB
from 50 m to 1.5 km (200–400 m bin: RT 94.9 vs FSPL 93.6 / UMa LOS
94.1 dB); NLOS cells are *harsher* than UMa NLOS (median RT 144 dB vs
UMa ~130–141 dB); LOS fraction among served cells is 48%. The PL was
reciprocal and consistent throughout — the v1 problem was always in
the budget layer, which is what v3 replaces.

## FDD approximation (n28, 0.7 GHz)

n28 is FDD: the paired uplink band sits ~55 MHz below the downlink
band. The UL layer **reuses the DL path gains** — no frequency
correction is invented. At 0.7 GHz the free-space difference over the
duplex gap is ~0.6 dB and material behaviour is nearly identical, so
the approximation is small and is stated rather than hidden. All
other twin bands (n41, n48/n77, FR3, n257) are TDD, where the path
gains are reciprocal at the same frequency.

## Reproduce

```bash
python ul_post.py        # writes all 20 *_ul.npz + ul_stats.json (v3)
python build_viewer.py   # regenerates twin_viewer.html with the DL/UL toggle
```

## Comparison: ray-traced layer vs link-budget layer

Same workbook budget terms, two propagation layers, average user
throughput (Mbps, mean of the 4 sites). The ray-traced layer receives
at a 6 m rooftop CPE; the budget layer (the twin's stochastic 3GPP UMa
planner, `budget_check.py` / `budget_single.py`) receives at a 1.5 m
handset at street level — the workbook's own UE height — and counts
unserved cells as zero.

| Band | RT twin DL | RT twin UL | RT ratio | Budget layer, single tower DL | UL | ratio | Budget layer, loaded network DL | UL | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 GHz | 52.4 | 26.9 | 1.94 | 13.3 | 0.8 | 17.0 | 16.9 | 2.5 | 6.74 |
| 2.5 GHz | 43.8 | 24.9 | 1.76 | 13.9 | 2.4 | 5.9 | 14.3 | 3.6 | 4.00 |
| 3.6 GHz | 41.7 | 24.3 | 1.71 | 15.0 | 3.4 | 4.4 | 13.2 | 4.0 | 3.25 |
| 7.125 GHz | 38.4 | 22.4 | 1.71 | 17.8 | 4.4 | 4.0 | 11.3 | 3.6 | 3.13 |

Reference points: the workbook marks the uplink **limiting in every
band and morphology** (700 MHz dense-urban edge: UL 64 kbps vs DL
950/258 kbps); field experience puts the DL/UL user-throughput ratio
at **3–5**. The budget layer brackets that range at FR1/FR3; the
ray-traced layer does not, because its receiver sits on a rooftop —
the difference is the operating point (height and clutter), not the
budget and not the averaging.

28 GHz is excluded from this table: the traced tile is NLOS-dominated
at mmWave and the budget layer's UMa model is extrapolated beyond its
defined range there (RT ratio 1.18; budget-layer 22.6, not credible).
The 0.7 GHz single-tower budget figure (17.0) is inflated by geometry:
from these rooftops the 700 MHz DL footprint covers 81% of the tile —
far past a planned 700 MHz cell border — while the handset antenna
(−7.53 dBi) leaves the UL serving 4% of it.
