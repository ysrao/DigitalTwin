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
- **Sionna 2.2.0** (sionna.phy channel models for UMa/UMi)
- **Gymnasium** (only if the env packaging was built; not required for paper results)

Install:
    pip install torch numpy scipy pandas openpyxl python-docx sionna gymnasium

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
- Macro -> 3GPP TR 38.901 UMa; pico/femto -> UMi-Street Canyon; outdoor-only
- Sionna above-6-GHz quirk: distance_2d_in must be shaped [batch, n_ut]
- Trace traffic: TrafficGen(trace_csv=...) maps normalized active_ues as a load multiplier
