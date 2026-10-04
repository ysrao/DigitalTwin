"""Action-conditioned dataset v2.

Sample unit: (cell, slice, decision time t).
- history: H epochs of KPI outcomes (epochs t-H..t-1)
- action: planned budget vectors for the cell over the next F epochs
- target: KPI outcomes over the next F epochs

Snapshot alignment: snapshot(e).kpis = outcomes of epoch e-1;
snapshot(e).budgets = budgets applied during epoch e.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

H, F = 6, 6
KPI_KEYS = ["served", "viol", "offered", "backlog", "delay_mean", "prb_used"]


def _log1p(a: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(a, 0))


class ActionDataset(Dataset):
    def __init__(self, corpus_dir: str | Path, split: str,
                 stats: dict | None = None, aux: bool = False):
        self.aux = aux
        self.corpus = Path(corpus_dir)
        manifest = json.loads((self.corpus / "manifest.json").read_text())
        train_ids = manifest["train_config_ids"]
        if split == "train":
            want = train_ids[2:]          # reserve 2 layouts for calibration
        elif split == "cal":
            want = train_ids[:2]
        else:
            want = manifest["held_out_config_ids"]
        runs = [r for r in manifest["runs"] if r["config_id"] in want]
        if not runs:
            raise ValueError(f"no runs for split {split}")

        xh, xa, ys = [], [], []
        se_t, dem_t, dem_m = [], [], []  # aux targets
        for r in runs:
            d = np.load(self.corpus / r["file"])
            E, C, S = d["served"].shape
            static = d["static"].astype(np.float32)          # (C, Fs)
            Fs = static.shape[1]
            kpi = np.stack([_log1p(d[k]) for k in KPI_KEYS],
                           axis=-1)                          # (E,C,S,K)
            kpi_raw = np.stack([d[k] for k in KPI_KEYS], -1)  # raw
            bud = d["budgets"].astype(np.float32)            # (E,C,S) PRBs
            tgt = np.stack([_log1p(d[k]) for k in ("served", "viol")],
                           axis=-1)                          # (E,C,S,2)
            for t in range(H, E - F):
                # hist: snapshots[t-H+1 .. t].kpis -> outcomes of t-H..t-1
                hist = kpi[t - H + 1:t + 1]                  # (H,C,S,K)
                act = bud[t:t + F]                           # (F,C,S)
                fut = tgt[t + 1:t + F + 1]                    # (F,C,S,2)
                if aux:
                    served_f = kpi_raw[t + 1:t + F + 1, :, :, 0]  # (F,C,S)
                    prb_f = kpi_raw[t + 1:t + F + 1, :, :, 5]
                    se = np.log1p(served_f / np.maximum(prb_f, 1e-9))
                    se_m = (prb_f > 1e-9).astype(np.float32)
                    util = prb_f / np.maximum(act, 1e-9)
                    dm = (util < 0.5).astype(np.float32) * se_m
                    to_cs = lambda a: a.transpose(1, 2, 0).reshape(C * S, F)
                    se_t.append(to_cs(se * se_m))
                    dem_t.append(to_cs(fut[..., 0]))
                    dem_m.append(to_cs(dm))
                st = np.broadcast_to(static[:, None, :], (C, S, Fs))
                so = np.zeros((C, S, S), np.float32)
                so[:, np.arange(S), np.arange(S)] = 1.0
                feat = np.concatenate(
                    [np.broadcast_to(hist, (H, C, S, len(KPI_KEYS))),
                     np.broadcast_to(st[None], (H, C, S, Fs)),
                     np.broadcast_to(so[None], (H, C, S, S))], axis=-1)
                xh.append(feat.reshape(H, C * S, -1).transpose(1, 0, 2))
                # action: full cell budget plan over horizon -> (C*S, F*S)
                a = act.transpose(1, 0, 2)          # (C, F, S)
                a = np.broadcast_to(a[:, None, :, :], (C, S, F, S))
                xa.append(a.reshape(C * S, F * S))
                ys.append(fut.reshape(F, C * S, 2).transpose(1, 0, 2))
        self.xh = np.concatenate(xh, 0).astype(np.float32)   # (N,H,Ft)
        self.xa = np.concatenate(xa, 0).astype(np.float32)   # (N,F*S)
        self.y = np.concatenate(ys, 0).astype(np.float32)    # (N,F,2)
        if aux:
            self.se_tgt = np.concatenate(se_t, 0).astype(np.float32)
            self.dem_tgt = np.concatenate(dem_t, 0).astype(np.float32)
            self.dem_mask = np.concatenate(dem_m, 0).astype(np.float32)

        if stats is None:
            self.h_mu = self.xh.mean(axis=(0, 1))
            self.h_sd = self.xh.std(axis=(0, 1)) + 1e-6
            self.h_sd[-S:] = 1.0
            self.h_mu[-S:] = 0.0
            self.a_mu = self.xa.mean(axis=0)
            self.a_sd = self.xa.std(axis=0) + 1e-6
            self.y_mu = self.y.mean(axis=(0, 1))
            self.y_sd = self.y.std(axis=(0, 1)) + 1e-6
        else:
            self.h_mu, self.h_sd = stats["h_mu"], stats["h_sd"]
            self.a_mu, self.a_sd = stats["a_mu"], stats["a_sd"]
            self.y_mu, self.y_sd = stats["y_mu"], stats["y_sd"]
        self.stats = {"h_mu": self.h_mu, "h_sd": self.h_sd,
                      "a_mu": self.a_mu, "a_sd": self.a_sd,
                      "y_mu": self.y_mu, "y_sd": self.y_sd,
                      "n_slices": S}

    def __len__(self):
        return len(self.xh)

    def __getitem__(self, i):
        xh = (self.xh[i] - self.h_mu) / self.h_sd
        xa = (self.xa[i] - self.a_mu) / self.a_sd
        y = (self.y[i] - self.y_mu) / self.y_sd
        if self.aux:
            return (torch.from_numpy(xh).float(), torch.from_numpy(xa).float(),
                    torch.from_numpy(y).float(),
                    torch.from_numpy(self.se_tgt[i]).float(),
                    torch.from_numpy(self.dem_tgt[i]).float(),
                    torch.from_numpy(self.dem_mask[i]).float())
        return (torch.from_numpy(xh).float(), torch.from_numpy(xa).float(),
                torch.from_numpy(y).float())
