> v2.0: use scripts/run_grid.py and the v2_*.json results (see README.md and CHANGELOG.md). The commands below are the original v1.x notes and refer to superseded results.

# Urban RAN Digital Twin — reproducible package

Complete code + data + checkpoints for the paper
"A 5G-Advanced RAN Digital Twin for Morphology-Aware Slice Dimensioning
on Existing Carrier Networks."

## Contents
- `src/urtwin/` — full simulator + learning code
- `scripts/` — runnable experiment scripts
- `docs/` — input formats, topology CSV schema
- `examples/` — carrier topology CSV + scenario JSON examples
- `data/` — corpora (corpus_fr3 = 32-run FR3 training corpus), all
  result JSONs (stage_e_*), trace_synthetic_24h.csv (parsed from the
  user's Synthetic_data.xlsx, 96 x 15-min rows)
- `checkpoints/` — trained predictors (black-box + gray-box FR3)

## Environment (paper numbers were produced with this exact stack)
- Linux x86_64, CPU-only, Python 3.12
- torch (CPU build), numpy, scipy, pandas, openpyxl, python-docx
- **Sionna 2.2.0** (sionna.phy channel models for UMa/UMi/RMa)
- **Gymnasium** (only if the env packaging was built; not required for paper results)

Install:
    pip install torch numpy scipy pandas openpyxl python-docx sionna gymnasium

## v1.1 and v1.2: suburban and rural morphologies (RMa)
See CHANGELOG.md. Run with one thread per worker:

    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/run_dimensioning_suburban.py 0 2   # and worker 1 2
    SUB_TAG=_heavy SUB_SCALES=8.0,9.0 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/run_dimensioning_suburban.py 0 2
    python scripts/analyze_dimensioning.py suburban data/stage_e_dimensioning_suburban_w0.json data/stage_e_dimensioning_suburban_w1.json
    # rural, traffic profile and uniform load (v1.2):
    RURAL_PROFILE=1 SUB_TAG=_ruralprofile SUB_SCALES=5.0,6.0 SUB_ISD=1732 SUB_SC=0 SUB_MORPH=rural-RMa-profile OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/run_dimensioning_suburban.py 0 2
    SUB_TAG=_rural SUB_SCALES=8.0,9.0 SUB_ISD=1732 SUB_SC=0 SUB_MORPH=rural-RMa OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/run_dimensioning_suburban.py 0 2
    python scripts/fig_morphology.py

## Reproduce the key paper results
1. Main dimensioning grid (360 runs, FR3 config):
       PYTHONPATH=src python scripts/run_dimensioning.py   # -> data/stage_e_dimensioning_fr3_20seed.json
2. Trace-driven dimensioning (180 runs, diurnal trace from Synthetic_data.xlsx):
       PYTHONPATH=src python scripts/run_tracedimensioning.py  # -> data/stage_e_dimensioning_tracedriven.json
3. Train predictors on the FR3 corpus:
       PYTHONPATH=src python -m urtwin.learning.train_v2 --corpus data/corpus_fr3 --out checkpoints/predictor_fr3_black.pt
       PYTHONPATH=src python -m urtwin.learning.train_structured --corpus data/corpus_fr3 --out checkpoints/predictor_fr3_gray.pt
4. Carrier topology import (no real topology needed; example CSV validates the path):
       PYTHONPATH=src python -c "from urtwin.scenarios.topology import load_topology; print(load_topology('examples/carrier_topology_example.csv').shape)"

## Configuration notes (important)
- Radio: 2100 MHz FDD 20 MHz (num 1), 3500 MHz TDD 100 MHz DDDSU (num 1),
  FR3 7125 MHz TDD 200 MHz DDDDDDSSUU (num 2)
- Macro -> 3GPP TR 38.901 UMa (urban) or RMa (suburban, v1.1); pico/femto -> UMi-Street Canyon; outdoor-only
- Sionna above-6-GHz quirk: distance_2d_in must be shaped [batch, n_ut]
- Trace traffic: TrafficGen(trace_csv=...) maps normalized active_ues as a load multiplier
