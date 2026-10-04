"""Train + evaluate the KPI predictor.

Gate (brief §4): held-out LAYOUTS (never trained on) — prediction MAE and
uncertainty calibration must beat persistence / moving-average baselines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from urtwin.learning.dataset import KpiDataset, H, F
from urtwin.learning.models import KpiPredictor, gaussian_nll


def to_log1p(y_std, stats):
    return y_std * stats["y_sd"] + stats["y_mu"]


def baseline_mae(ds: KpiDataset, mode: str) -> dict:
    """Persistence: repeat last history target; MA: mean of history targets."""
    maes = {"served": [], "viol": []}
    tgt_idx = [0, 1]  # served, viol are DYN_KEYS[0], DYN_KEYS[1]
    for i in range(len(ds)):
        x_raw = ds.x[i]  # unstandardized
        y_raw = ds.y[i]
        hist = x_raw[:, tgt_idx]  # (H, 2) in log1p units
        pred = (np.repeat(hist[-1][None, :], F, axis=0) if mode == "persist"
                else np.repeat(hist.mean(axis=0)[None, :], F, axis=0))
        err = np.abs(pred - y_raw)
        maes["served"].append(err[:, 0].mean())
        maes["viol"].append(err[:, 1].mean())
    return {k: float(np.mean(v)) for k, v in maes.items()}


def main(corpus_dir="data/corpus_v1_full", epochs=25, batch=512,
         out="checkpoints/predictor_v1.pt"):
    torch.manual_seed(0)
    train_ds = KpiDataset(corpus_dir, "train")
    test_ds = KpiDataset(corpus_dir, "test", stats=train_ds.stats)
    print(f"train {len(train_ds)} samples, test {len(test_ds)} samples, "
          f"feat {train_ds.x.shape[2]}", flush=True)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True)

    model = KpiPredictor(train_ds.x.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xb, yb in loader:
            mu, lv = model(xb)
            loss = gaussian_nll(mu, lv, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1}/{epochs} nll={tot/n:.4f}", flush=True)

    # ---- evaluation on held-out layouts ----
    model.eval()
    tloader = DataLoader(test_ds, batch_size=2048)
    maes = {"served": [], "viol": []}
    cov = {"served": [], "viol": []}
    with torch.no_grad():
        for xb, yb in tloader:
            mu, lv = model(xb)
            mu_r = to_log1p(mu, test_ds.stats)
            y_r = to_log1p(yb, test_ds.stats)
            err = (mu_r - y_r).abs()
            maes["served"].append(err[:, :, 0].mean().item())
            maes["viol"].append(err[:, :, 1].mean().item())
            sd = torch.exp(0.5 * lv)
            inside = ((y_r >= mu_r - 1.645 * sd)
                      & (y_r <= mu_r + 1.645 * sd)).float()
            cov["served"].append(inside[:, :, 0].mean().item())
            cov["viol"].append(inside[:, :, 1].mean().item())
    res = {
        "gru_mae_log1p": {k: float(np.mean(v)) for k, v in maes.items()},
        "gru_pi90_coverage": {k: float(np.mean(v)) for k, v in cov.items()},
        "persist_mae_log1p": baseline_mae(test_ds, "persist"),
        "ma_mae_log1p": baseline_mae(test_ds, "ma"),
    }
    print(json.dumps(res, indent=1))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(),
                "feat_dim": train_ds.x.shape[2],
                "stats": {k: v for k, v in train_ds.stats.items()},
                "results": res}, out)
    print(f"checkpoint -> {out}")
    return res


if __name__ == "__main__":
    main()
