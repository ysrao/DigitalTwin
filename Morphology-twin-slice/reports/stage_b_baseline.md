# Stage B — Interactive baseline: BUILT, gates pass

Date: 2026-10-03. Reference baseline simulator for the Urban RAN Digital Twin.

## What was built (`src/urtwin/`)

| Module | Implements |
| --- | --- |
| `scenarios/loader.py` | Schema-v1 loading + validation (shares sum to 1, TDD needs pattern, min-shares feasible) |
| `sim/numerology.py` | 3GPP 38.104 FR1 N_PRB table, slot durations, TDD DL fraction from pattern |
| `sim/hetnet.py` | 7-site × 3-sector macro hex + pico/femto underlay (uniform/hotspot/explicit); per-cell FDD+TDD PRB budgets |
| `sim/channel.py` | Sionna UMa coupling loss for arbitrary HetNet geometry via explicit `set_topology` tensors; analytic TR 38.901 sector pattern on top |
| `sim/mobility.py` | Mixed classes (stationary/pedestrian/vehicular), random-direction movement, A3 handover (hysteresis + TTT), 6 dB CRE bias to small cells |
| `sim/traffic.py` | Per-slice generators (eMBB Poisson/12kb, URLLC Poisson/256b) |
| `sim/prach.py` | Slotted PRACH, 64 preambles, collision + backoff; access KPIs separate from payload |
| `sim/slicing.py` | Guaranteed PRB budgets + reactive borrowing from shared pool |
| `sim/baseline.py` | Slot loop: budgets → mobility/HO → arrivals → PRACH → PF scheduling → expiry; full KPI report |

## Gate results (brief §4) — ALL PASS

- **PRB accounting**: 9,900 slot-cells checked, allocated ≤ usable total everywhere.
- **Queue conservation**: arrived 1375.9 kb = served 1371.1 + dropped 0.0 + queued 4.8 kb.
- **Overload**: flooded demand → 676 eMBB violations counted and reported, never hidden.
- **Interference response**: serving-cell selection sane (SINR mean +1.3 dB).

## Demo headline (60 UEs, 33 cells, 1000 slots = 0.5 s, 4.6 s wall-clock)

| Slice | Served | Violations | Delay p50/p95 |
| --- | --- | --- | --- |
| eMBB | 6993 kb | 33 | 0.0 / 3.0 ms |
| URLLC | 7.6 kb | 1 | 0.0 / 0.3 ms |
| mIoT | 631 kb | 0 | 5.0 / 12.5 ms |

18 handovers, PRACH 812/816 successes.

## Bugs found and fixed during build

1. **Cell selection used argmin on RSRP** (higher-is-better) — every UE attached
   to its *worst* cell; SINR mean was −46 dB. Fixed → +1.3 dB.
2. **Borrowing lent URLLC's idle guarantee** — bursty URLLC arrivals mid-epoch
   found zero budget and missed 5 ms deadlines (15 violations). Guarantees of
   latency-critical slices (deadline ≤ 2 epochs) are now held. → 1 violation.

## Abstractions (documented in code)

Sionna supplies wideband coupling loss; per-slot SINR adds thermal noise and
full-load interference analytically. Spectral efficiency = Shannon with 3 dB
gap, capped 7.5 bps/Hz. TDD duty cycle folded into average DL PRB budget.
Single-element panels; sectorization via analytic 65° pattern. Indoor UTs
excluded pending the Stage A edge-case investigation.

## Next: Stage C

Corpus generation (many layouts × traffic × seeds → KPI trajectories) and the
pretrained predictor. The reactive borrowing policy built here is the baseline
Stage D must beat.
