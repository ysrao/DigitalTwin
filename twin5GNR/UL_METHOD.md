# Uplink (UL) model for twin5GNR

**Everything on this page is MODELLED, not measured.** The uplink layer
reuses the downlink's ray-traced paths and applies an uplink link budget.
Generator: `ul_post.py`. Outputs: `data_twin_tower_<site>_<band>_ul.npz`
(`ul_rsrp`, `ul_snr`, `ul_cap_mbps`, `los`, `meta`) for all 20 site-bands.

## How it works

The DL grids were composed in post-processing from the stored top-32 paths
per cell (`pk_pw`, `pk_th`, `pk_ph` in each solver file): per sector, the
path powers are weighted by the 3GPP TR 38.901 65° element pattern
(4° electrical downtilt, sector azimuths 90/210/330°) and summed; RSRP is
then `P_COND + 10·log10(channel power) + array gain`, best sector wins.

The UL layer changes **only the link direction**, nothing else:

```
ul_rsrp_sector = UE_P + 10*log10( sum_paths pk_pw * 10^(g_db/10) ) + G_ARRAY
ul_rsrp        = max over the 3 sectors          (NaN where no path)
ul_snr         = ul_rsrp - (-174 + 10*log10(BW_Hz) + NF_BS + RISE)
ul_cap         = BW * min(7, log2(1 + 10^(ul_snr/10)))     [Mbps]
```

The receive pattern weight `g_db` is applied on the same stored departure
angles the DL used: by reciprocity, the angle of arrival at the base
station for a reversed path equals the DL departure angle (the first/last
segment is the same straight line, traversed in reverse), so the BS
antenna weights each path identically in both directions.

## Assumptions (all labelled, all stated)

| Term | Value | Status |
|---|---|---|
| UE transmit power | 23 dBm | assumption (typical UE power class 3) |
| UE antenna | omni, 0 dBi | assumption (no UE pattern/body loss) |
| BS receive pattern | TR 38.901 65° element, tilt 4°, same sectors as DL | reciprocity, as DL |
| BS array gain | 12.04 dB (4×4, ideal beamforming) | same stated assumption as DL |
| BS noise figure | 4 dB | assumption |
| Interference rise | +3 dB over thermal | assumption (single tower: no modelled interferers; this is an allowance, not a computed interference) |
| Capacity | Shannon, capped at 7 bps/Hz | same convention as DL |
| Scheduler / power control / TTI bundling | not modelled | — a single "best user alone on the carrier" budget, like the DL layer |

## The constant-shift check

DL RSRP per sector is `P_COND + 10·log10(P_dir) + G_ARRAY` with
`P_COND = 47 − 8 − 12.04 = 26.96 dBm`; UL RSRP per sector is
`23 + 10·log10(P_dir) + G_ARRAY` over the *same* `P_dir`. Both take the max
over the same sectors, so the model predicts

```
ul_rsrp − dl_rsrp = 23 − 26.96 = −3.96 dB   on every served cell.
```

Measured on the produced grids: the shift is **exactly −3.96 dB
(median = min = max) on all 20 site-bands**, and the UL no-path (NaN)
cells are exactly the DL ones. This confirms the UL composition mirrors
the DL post-processing with no hidden term. It also means UL coverage is
DL coverage shifted by one constant: the −3.96 dB is the net of a much
weaker transmitter (23 dBm vs 47 dBm EIRP) largely bought back by the
element + array gain on receive (20.04 dB).

Note the UL *SNR* is not simply the DL SNR shifted: the UL noise floor
uses NF 4 dB + 3 dB rise (= 7 dB over thermal) vs the DL's 9 dB, so
`ul_snr = dl_snr(at same BW) − 3.96 + 2 = dl_snr − 1.96 dB`.

## FDD approximation (n28, 0.7 GHz)

n28 is FDD: the paired uplink band sits ~55 MHz below the downlink band.
The UL layer **reuses the DL path gains** for the uplink — no frequency
correction is invented. At 0.7 GHz the free-space difference over the
duplex gap is ~0.6 dB and material behaviour is nearly identical, so the
approximation is small; it is stated here rather than hidden. All other
bands in the twin (n41, n48/n77, FR3, n257) are TDD, where reciprocity of
the path gains is exact at the same frequency.

## Results (viewer-default bandwidths)

Coverage = % of all 2,601 grid cells with RSRP ≥ −90 dBm (no-path cells
count as not covered). Mean capacity = mean over served cells of
BW·min(7, log2(1+SNR)) at the stated BW, single user on the carrier.

| Site-band | BW (MHz) | DL cov ≥ −90 dBm | UL cov ≥ −90 dBm | Mean DL cap (Mbps) | Mean UL cap (Mbps) | Median UL−DL RSRP shift (dB) |
|---|---|---|---|---|---|---|
| NW — 0.7 GHz (n28, FDD) | 10 | 61.2% | 57.7% | 51 | 50 | -3.96 |
| NW — 2.5 GHz (n41) | 10 | 51.0% | 48.7% | 42 | 41 | -3.96 |
| NW — 3.6 GHz (n48/n77) | 10 | 48.2% | 45.1% | 40 | 38 | -3.96 |
| NW — 7.125 GHz (FR3) | 10 | 42.9% | 39.9% | 35 | 33 | -3.96 |
| NW — 28 GHz (n257) | 200 | 31.6% | 30.1% | 212 | 175 | -3.96 |
| NE — 0.7 GHz (n28, FDD) | 10 | 66.1% | 61.9% | 54 | 52 | -3.96 |
| NE — 2.5 GHz (n41) | 10 | 53.1% | 51.4% | 45 | 44 | -3.96 |
| NE — 3.6 GHz (n48/n77) | 10 | 51.1% | 49.5% | 43 | 43 | -3.96 |
| NE — 7.125 GHz (FR3) | 10 | 48.7% | 47.6% | 41 | 41 | -3.96 |
| NE — 28 GHz (n257) | 200 | 44.9% | 44.3% | 309 | 256 | -3.96 |
| SW — 0.7 GHz (n28, FDD) | 10 | 61.6% | 58.2% | 50 | 49 | -3.96 |
| SW — 2.5 GHz (n41) | 10 | 49.1% | 47.3% | 42 | 41 | -3.96 |
| SW — 3.6 GHz (n48/n77) | 10 | 46.6% | 45.6% | 40 | 39 | -3.96 |
| SW — 7.125 GHz (FR3) | 10 | 44.4% | 43.3% | 37 | 37 | -3.96 |
| SW — 28 GHz (n257) | 200 | 39.4% | 37.8% | 287 | 240 | -3.96 |
| SE — 0.7 GHz (n28, FDD) | 10 | 67.4% | 64.6% | 55 | 53 | -3.96 |
| SE — 2.5 GHz (n41) | 10 | 56.6% | 53.9% | 46 | 45 | -3.96 |
| SE — 3.6 GHz (n48/n77) | 10 | 53.0% | 50.7% | 44 | 43 | -3.96 |
| SE — 7.125 GHz (FR3) | 10 | 48.9% | 47.7% | 41 | 40 | -3.96 |
| SE — 28 GHz (n257) | 200 | 44.1% | 42.1% | 297 | 247 | -3.96 |

Reading: at FR1 the UL mean capacity sits within ~1–2 Mbps of DL because
most served cells are at the 7 bps/Hz Shannon cap in both directions. The
gap widens at 28 GHz / 200 MHz, where more cells are noise-limited and the
−1.96 dB relative SNR bites. UL coverage (≥ −90 dBm) runs 1–4 points
below DL everywhere — the direct footprint of the −3.96 dB shift.

## Reproduce

```bash
python ul_post.py        # writes all 20 *_ul.npz + ul_stats.json
python build_viewer.py   # regenerates twin_viewer.html with the DL/UL toggle
```
