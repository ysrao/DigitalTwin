"""Summaries for r4 experiments: means, 95% bootstrap CIs, paired Wilcoxon."""
import json, sys, glob
import numpy as np
from scipy.stats import wilcoxon
rng = np.random.default_rng(0)
def ci(x, B=10000):
    x = np.asarray(x, float); b = rng.choice(x, (B, len(x))).mean(1)
    return [round(float(x.mean()), 5), round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)]
def pw(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    if np.all(d == 0): return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
def load(mode):
    R = []
    for f in sorted(glob.glob(f"data/r4_{mode}_w*.json")): R += json.load(open(f))
    return R
def pair(R, key_fn):
    lo = {(r["layout"], r["seed"], r["mult"]): r for r in R if r["urllc_min"] == 0.10}
    hi = {(r["layout"], r["seed"], r["mult"]): r for r in R if r["urllc_min"] == 0.30}
    ks = sorted(set(lo) & set(hi)); return ks, lo, hi
out = {}
for mode in ("long", "hour"):
    R = load(mode)
    if not R: continue
    for m in sorted(set(r["mult"] for r in R)):
        S = [r for r in R if r["mult"] == m]
        ks, lo, hi = pair(S, None)
        g = lambda D, f: [f(D[k]) for k in ks]
        uL, uH = g(lo, lambda r: r["URLLC"]["violations"]), g(hi, lambda r: r["URLLC"]["violations"])
        eL, eH = g(lo, lambda r: r["eMBB"]["violations"]), g(hi, lambda r: r["eMBB"]["violations"])
        dE = np.mean(eH) - np.mean(eL); dU = np.mean(uL) - np.mean(uH)
        res = dict(n_pairs=len(ks), urllc_10=ci(uL), urllc_30=ci(uH), p_urllc=pw(uL, uH),
                   embb_10=ci(eL), embb_30=ci(eH), p_embb=pw(eL, eH),
                   exchange=round(dE / dU, 1) if dU > 0 else None,
                   urllc_rel_10=ci(g(lo, lambda r: 1 - r["URLLC"]["violations"] / max(1, r["URLLC"]["packets"]))),
                   embb_rel_10=ci(g(lo, lambda r: 1 - r["eMBB"]["violations"] / max(1, r["eMBB"]["packets"]))),
                   urllc_rel_30=ci(g(hi, lambda r: 1 - r["URLLC"]["violations"] / max(1, r["URLLC"]["packets"]))),
                   embb_rel_30=ci(g(hi, lambda r: 1 - r["eMBB"]["violations"] / max(1, r["eMBB"]["packets"]))),
                   prb_util_10=ci(g(lo, lambda r: r["prb_util"])))
        allr = S
        res["miot"] = dict(violations=int(sum(r["mIoT"]["violations"] for r in allr)),
                           packets=int(sum(r["mIoT"]["packets"] for r in allr)),
                           delay_max_ms=max(r["mIoT"]["delay_ms_max"] for r in allr),
                           delay_p99_ms_mean=round(float(np.mean([r["mIoT"]["delay_ms_p99"] for r in allr])), 1),
                           queued_end_max=max(r["mIoT"]["queued_end"] for r in allr),
                           queued_end_max_age_ms=max(r["mIoT"]["queued_end_max_age_ms"] for r in allr))
        att = sum(r["prach"]["attempts"] for r in allr); col = sum(r["prach"]["collisions"] for r in allr)
        res["prach_collision_pct"] = round(100 * col / max(1, att), 2)
        res["handovers_per_run"] = round(float(np.mean([r["handovers"] for r in allr])), 1)
        out[f"{mode}_x{m}"] = res
json.dump(out, open("data/r4_summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
