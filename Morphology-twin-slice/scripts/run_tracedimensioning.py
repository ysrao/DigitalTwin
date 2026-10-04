"""Trace-driven dimensioning: 9 structures x 2 layouts x 10 seeds with
diurnal trace traffic. Tests whether the URLLC floor trade-off holds
under temporally correlated load."""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from urtwin.control.dimension import run_one, GRID

SCALES = tuple(json.loads(Path("data/calibration_choice.json").read_text()))
TRACE = "data/trace_synthetic_24h.csv"
out_path = Path("data/stage_e_dimensioning_tracedriven.json")
results = []
for layout in (0, 1):
    for s in range(10):
        seed = 7000 + layout * 100 + s
        for embb_min, urllc_min in GRID:
            r = run_one(layout, seed, embb_min, urllc_min, scales=SCALES,
                        trace_csv=TRACE)
            r["traffic_scales"] = list(SCALES)
            results.append(r)
            print(f"layout{layout} seed{s} e={embb_min:.2f}/u={urllc_min:.2f}: "
                  f"viol={r['violations']} (e:{r['viol_embb']} u:{r['viol_urllc']})",
                  flush=True)
    out_path.write_text(json.dumps(results))
out_path.write_text(json.dumps(results, indent=1))
print(f"-> {out_path} ({len(results)} runs)")
