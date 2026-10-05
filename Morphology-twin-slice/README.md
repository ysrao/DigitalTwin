# Morphology-twin-slice (v2.0)

Code, data and results for the paper
**"A 5G-Advanced RAN Digital Twin for Morphology-Aware Slice Dimensioning on Existing Carrier Networks"**
(Rao S. Yenamandra). Appendix A of the paper maps every table and figure to its command and result file.

The twin is an offline, deterministic RAN digital twin. It couples Sionna 3GPP TR 38.901 channels (UMa, UMi, RMa) with a slot-level downlink slicing simulator, and it is evaluated on three morphologies: urban, suburban and rural (1800 grid runs, v2.0 corrected antenna model).

## Setup (Linux, CPU, Python 3.12+)

    pip install torch numpy scipy pandas pyyaml matplotlib openpyxl python-docx sionna==2.2.0
    PYTHONPATH=src python -m pytest tests -q

## Reproduce (paper v8, Appendix A)

    PYTHONPATH=src python scripts/run_grid.py urban 0 1          # 360 runs; also: suburban, rural_uniform, rural_profile, ablation, trace
    PYTHONPATH=src python scripts/run_prach_v2.py 0 1            # PRACH probe
    PYTHONPATH=src python scripts/analyze_dimensioning.py urban data/v2_urban_w0.json data/v2_urban_w1.json
    python scripts/figs_v2.py                                    # Figs. 1-5

Use `0 2` and `1 2` with `OMP_NUM_THREADS=1` to run two workers in parallel. See `CHANGELOG.md` for the v2.0 antenna fix.

## Contents

- `src/urtwin/`: simulator (HetNet, channels, slicing and borrowing, PRACH, mobility) and learning code
- `scripts/`: experiments, calibration, analysis and figures
- `data/`: v2.0 result files (`v2_*.json`), calibration sweeps, summaries, FR3 corpus, 24 h trace; superseded v1.x results in `data/v1_superseded/`
- `checkpoints/`: trained predictors
- `paper/`: paper v8 and Figs. 1-5
- `CHANGELOG.md`, `README_FOR_CLAUDE.md` (detailed reproduction notes), `README_build.md` (build history)
