"""Train the gray-box structured predictor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from urtwin.learning.dataset_v2 import ActionDataset, F
from urtwin.learning.models_structured import StructuredPredictor


def main(corpus_dir="data/corpus_v4", epochs=25, batch=512,
         out="checkpoints/predictor_v5.pt"):
    torch.manual_seed(0)
    train_ds = ActionDataset(corpus_dir, "train", aux=True)
    cal_ds = ActionDataset(corpus_dir, "cal", stats=train_ds.stats, aux=True)
    test_ds = ActionDataset(corpus_dir, "test", stats=train_ds.stats, aux=True)
    print(f"train {len(train_ds)} / cal {len(cal_ds)} / test {len(test_ds)}",
          flush=True)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    y_mu = torch.from_numpy(train_ds.y_mu).float()
    y_sd = torch.from_numpy(train_ds.y_sd).float()

    model = StructuredPredictor(train_ds.xh.shape[2],
                                train_ds.stats["n_slices"])
    model.set_action_stats(train_ds.a_mu, train_ds.a_sd)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xh, xa_s, yb, se_t, dem_t, dem_m in loader:
            xa_raw = xa_s * model.a_sd + model.a_mu
            served, viol, _, se, dem = model(xh, xa_raw)
            yr = yb * y_sd + y_mu
            loss = ((served - yr[:, :, 0]) ** 2).mean() + \
                   ((viol - yr[:, :, 1]) ** 2).mean()
            se_mask = (se_t > 0).float()
            if se_mask.sum() > 0:
                loss = loss + 0.5 * (((se - se_t) ** 2 * se_mask).sum() /
                                     se_mask.sum())
            if dem_m.sum() > 0:
                loss = loss + 0.5 * (((dem - dem_t) ** 2 * dem_m).sum() /
                                     dem_m.sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xh)
            n += len(xh)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1}/{epochs} mse={tot/n:.4f}", flush=True)

    # eval + fallback threshold (p90 of predicted demand-std on cal)
    model.eval()
    maes = {"served": [], "viol": []}
    uncs = []
    with torch.no_grad():
        for xh, xa_s, yb, se_t, dem_t, dem_m in DataLoader(cal_ds,
                                                           batch_size=2048):
            xa_raw = xa_s * model.a_sd + model.a_mu
            served, viol, dlv, _, _ = model(xh, xa_raw)
            uncs.append((torch.exp(0.5 * dlv) * y_sd[0]).mean(-1).numpy())
        for xh, xa_s, yb, se_t, dem_t, dem_m in DataLoader(test_ds,
                                                           batch_size=2048):
            xa_raw = xa_s * model.a_sd + model.a_mu
            served, viol, _, _, _ = model(xh, xa_raw)
            yr = yb * y_sd + y_mu
            maes["served"].append((served - yr[:, :, 0]).abs().mean().item())
            maes["viol"].append((viol - yr[:, :, 1]).abs().mean().item())
    uncs = np.concatenate(uncs)
    thr = float(np.percentile(uncs, 90))
    res = {"mae_log1p": {k: float(np.mean(v)) for k, v in maes.items()},
           "fallback_unc_p90": thr}
    print(json.dumps(res, indent=1))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(),
                "hist_dim": train_ds.xh.shape[2],
                "n_slices": train_ds.stats["n_slices"],
                "model_kind": "structured",
                "stats": train_ds.stats,
                "fallback_thr": thr,
                "results": res}, out)
    print(f"checkpoint -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_v4")
    ap.add_argument("--out", default="checkpoints/predictor_v5.pt")
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    main(corpus_dir=a.corpus, epochs=a.epochs, out=a.out)
