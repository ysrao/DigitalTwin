"""Train action-conditioned predictor v2 + temperature recalibration.

Recalibration: grid-search a sigma multiplier on the 'cal' layouts so the
nominal 90% prediction interval actually covers ~90% (fixes the Stage C
overconfidence before MPC trusts the margins).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from urtwin.learning.dataset_v2 import ActionDataset, F
from urtwin.learning.models_v2 import ActionPredictor, gaussian_nll


def evaluate(model, ds, sigma_scale=1.0, batch=2048):
    model.eval()
    maes = {"served": [], "viol": []}
    covs = {"served": [], "viol": []}
    with torch.no_grad():
        for xh, xa, yb in DataLoader(ds, batch_size=batch):
            mu, lv = model(xh, xa)
            sd = torch.exp(0.5 * lv) * sigma_scale
            mu_r = mu * ds.y_sd + ds.y_mu
            y_r = yb * ds.y_sd + ds.y_mu
            err = (mu_r - y_r).abs()
            maes["served"].append(err[:, :, 0].mean().item())
            maes["viol"].append(err[:, :, 1].mean().item())
            inside = ((y_r >= mu_r - 1.645 * sd) &
                      (y_r <= mu_r + 1.645 * sd)).float()
            covs["served"].append(inside[:, :, 0].mean().item())
            covs["viol"].append(inside[:, :, 1].mean().item())
    return ({k: float(np.mean(v)) for k, v in maes.items()},
            {k: float(np.mean(v)) for k, v in covs.items()})


def main(corpus_dir="data/corpus_v2", epochs=25, batch=512,
         out="checkpoints/predictor_v2.pt", model_kind="v2"):
    torch.manual_seed(0)
    train_ds = ActionDataset(corpus_dir, "train")
    cal_ds = ActionDataset(corpus_dir, "cal", stats=train_ds.stats)
    test_ds = ActionDataset(corpus_dir, "test", stats=train_ds.stats)
    print(f"train {len(train_ds)} / cal {len(cal_ds)} / test {len(test_ds)} "
          f"samples; hist_feat {train_ds.xh.shape[2]}, "
          f"act {train_ds.xa.shape[1]}", flush=True)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True)

    if model_kind == "film":
        from urtwin.learning.models_v3 import FiLMPredictor
        model = FiLMPredictor(train_ds.xh.shape[2], train_ds.stats["n_slices"])
    else:
        model = ActionPredictor(train_ds.xh.shape[2], train_ds.stats["n_slices"])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xh, xa, yb in loader:
            mu, lv = model(xh, xa)
            loss = gaussian_nll(mu, lv, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xh)
            n += len(xh)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1}/{epochs} nll={tot/n:.4f}", flush=True)

    # ---- temperature scaling on cal layouts ----
    best_c, best_gap = 1.0, 9.0
    for c in np.arange(0.5, 4.01, 0.1):
        _, cov = evaluate(model, cal_ds, sigma_scale=c)
        gap = abs(cov["served"] - 0.90)
        if gap < best_gap:
            best_gap, best_c = gap, c
    print(f"recalibration: sigma_scale={best_c:.2f} "
          f"(cal served coverage gap {best_gap:.3f})", flush=True)

    mae, cov_raw = evaluate(model, test_ds, sigma_scale=1.0)
    _, cov_cal = evaluate(model, test_ds, sigma_scale=best_c)
    res = {"mae_log1p": mae,
           "pi90_coverage_raw": cov_raw,
           "pi90_coverage_calibrated": cov_cal,
           "sigma_scale": float(best_c)}
    print(json.dumps(res, indent=1))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(),
                "hist_dim": train_ds.xh.shape[2],
                "n_slices": train_ds.stats["n_slices"],
                "model_kind": model_kind,
                "stats": train_ds.stats,
                "sigma_scale": float(best_c),
                "results": res}, out)
    print(f"checkpoint -> {out}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_v2")
    ap.add_argument("--out", default="checkpoints/predictor_v2.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--model", default="v2", choices=["v2", "film"])
    a = ap.parse_args()
    main(corpus_dir=a.corpus, epochs=a.epochs, out=a.out, model_kind=a.model)
