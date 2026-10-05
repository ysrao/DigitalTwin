"""Suburban load calibration, same criterion as scripts/calibrate_load.py:
sweep traffic_scale at the (0.30, 0.10) structure to reach the heavy-load
regime (~70 violations/run, URLLC ~2-3 at the 10% floor)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_dimensioning_suburban import run_one_suburban
tss = [float(x) for x in sys.argv[2].split(",")]
out = []
for ts in tss:
    for s in range(int(__import__("os").environ.get("CAL_SEEDS", "3"))):
        r = run_one_suburban(0, 2000 + s, 0.30, 0.10, scales=(ts, ts))
        out.append({"traffic_scale": ts, **r})
        print(f"ts={ts} s{s}: viol={r['violations']} e={r['viol_embb']} u={r['viol_urllc']} util={r['prb_util']}", flush=True)
Path(f"data/calibration_suburban_{sys.argv[1]}.json").write_text(json.dumps(out, indent=1))
