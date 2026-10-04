"""Load calibration on the new FR3 config: sweep traffic_scale at the
(0.30, 0.10) structure to recover the pilot heavy-load regime
(~70 violations/run, URLLC ~2-3 at 10% floor)."""
import json, sys, statistics
from pathlib import Path
sys.path.insert(0, "src")
from urtwin.control.dimension import run_one

out = []
for ts in [2.0, 3.0, 4.0, 5.0, 6.0]:
    for s in range(3):
        seed = 2000 + s
        r = run_one(0, seed, 0.30, 0.10, scales=(ts, ts))
        out.append({"traffic_scale": ts, **r})
        print(f"ts={ts} seed{s}: viol={r['violations']} "
              f"(e:{r['viol_embb']} u:{r['viol_urllc']}) util={r['prb_util']:.3f}",
              flush=True)

Path("data/calibration_fr3.json").write_text(json.dumps(out, indent=1))
print("\nSummary:")
for ts in [2.0, 3.0, 4.0, 5.0, 6.0]:
    v = [r["violations"] for r in out if r["traffic_scale"] == ts]
    u = [r["viol_urllc"] for r in out if r["traffic_scale"] == ts]
    print(f"ts={ts}: viol mean {statistics.mean(v):.1f}, urllc mean {statistics.mean(u):.2f}")
