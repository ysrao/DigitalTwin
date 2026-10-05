"""Revision r4 experiments (addresses review high-severity items).
long : 10,000-slot (5 s) urban runs at the 30%/10% and 30%/30% structures,
       2 layouts x 10 seeds, so the 1000 ms mIoT deadline is observable.
hour : stationary hour-snapshot runs (1000 slots) at trough/median/peak hourly
       multipliers of the synthetic 24 h profile, replacing the compressed ramp.
Usage: python scripts/run_revision_r4.py <long|hour> <worker> <n_workers>"""
import copy, json, sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, "src")
from urtwin.learning.corpus import _variants
from urtwin.scenarios.loader import from_dict
from urtwin.sim.baseline import BaselineSimulator

SCALES = (5.0, 6.0)
HOURS = {"trough_h00": 0.596, "median": 0.926, "peak_h19": 1.638}

def run(layout, seed, e, u, n_slots=1000, mult=1.0):
    base = yaml.safe_load(Path("configs/scenario_schema_v1.yaml").read_text())
    for i, cfg, d in _variants(base, SCALES):
        if i == layout: break
    sc = from_dict(copy.deepcopy(d))
    for s in sc.slices:
        s.min_share = {"eMBB": e, "URLLC": u, "mIoT": 0.05}[s.name]
    sim = BaselineSimulator(sc, n_ues=cfg["n_ues"], n_slots=n_slots, seed=seed,
                            budget_policy="reactive", borrowing=True)
    sim.traffic.rates = {k: v / (cfg["traffic_scale"] * mult) for k, v in sim.traffic.rates.items()}
    rep = sim.run()
    md = np.array(sim.delays["mIoT"]) * sim.slot_ms
    q = [p for qq in sim.ue_queues for p in qq if p["slice"] == "mIoT"]
    age = [(n_slots - p["arrival_slot"]) * sim.slot_ms for p in q]
    out = dict(layout=layout, seed=seed, embb_min=e, urllc_min=u, n_slots=n_slots, mult=mult,
               n_ues=cfg["n_ues"], handovers=rep["handovers"], prach=rep["prach"])
    for sl in ("eMBB", "URLLC", "mIoT"):
        r = rep[sl]
        out[sl] = {k: r[k] for k in ("violations", "packets", "reliability", "served_kb",
                                     "delay_ms_p50", "delay_ms_p95", "prb_used")}
    out["mIoT"]["delay_ms_max"] = float(md.max()) if md.size else 0.0
    out["mIoT"]["delay_ms_p99"] = float(np.percentile(md, 99)) if md.size else 0.0
    out["mIoT"]["queued_end"] = len(q)
    out["mIoT"]["queued_end_max_age_ms"] = max(age) if age else 0.0
    out["prb_util"] = sum(rep[s]["prb_used"] for s in ("eMBB", "URLLC", "mIoT")) / float(sum(sim.cell_total_prbs) * n_slots)
    return out

def jobs(mode):
    if mode == "long":
        return [(L, 3000 + 100 * L + s, 0.30, u, 10000, 1.0) for L in (0, 1) for s in range(10) for u in (0.10, 0.30)]
    return [(L, 7000 + 100 * L + s, 0.30, u, 1000, m) for m in HOURS.values() for L in (0, 1) for s in range(10) for u in (0.10, 0.30)]

if __name__ == "__main__":
    mode, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    outp = Path(f"data/r4_{mode}_w{w}.json")
    res = json.loads(outp.read_text()) if outp.exists() else []
    J = jobs(mode)[w::nw]
    for j in J[len(res):]:
        res.append(run(*j)); outp.write_text(json.dumps(res, indent=1))
        print(mode, w, len(res), "/", len(J), flush=True)
