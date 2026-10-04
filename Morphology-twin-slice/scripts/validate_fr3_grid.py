"""Validation grid on the new radio config (FR3 + per-carrier coupling + UMa/UMi).

Subset of the dimensioning sweep: 9 share structures x 1 layout x 2 seeds.
Does NOT touch data/stage_e_dimensioning.json (FR1 pilot backing the paper).
"""
import json
from pathlib import Path

from urtwin.control.dimension import run_one, GRID

results = []
for s in range(2):
    seed = 1000 + s
    for embb_min, urllc_min in GRID:
        r = run_one(0, seed, embb_min, urllc_min)
        results.append(r)
        print(f"seed{s} e={embb_min:.2f}/u={urllc_min:.2f}: "
              f"served={r['served_kb']:.0f} viol={r['violations']} "
              f"(e:{r['viol_embb']} u:{r['viol_urllc']} m:{r['viol_miot']}) "
              f"util={r['prb_util']:.3f}", flush=True)

out = Path("data/stage_e_validation_fr3.json")
out.write_text(json.dumps(results, indent=1))
print(f"-> {out} ({len(results)} runs)")
