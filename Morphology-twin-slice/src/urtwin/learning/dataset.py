"""Supervised dataset for the KPI predictor.

Sample unit: (cell, slice, time). Input: H=6 epochs of history
(dynamic KPIs + static cell features + slice one-hot). Target: next F=6
epochs of served traffic and violation counts. Splits are by layout
(config_id) — held-out layouts never appear in training (no area leakage).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

H, F = 6, 6
DYN_KEYS = ["served", "viol", "offered", "backlog", "budgets",
            "delay_mean", "prb_used"]
TARGET_KEYS = ["served", "viol"]


def _log1p_norm(a: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(a, 0))


class KpiDataset(Dataset):
    def __init__(self, corpus_dir: str | Path, split: str,
                 stats: dict | None = None):
        self.corpus = Path(corpus_dir)
        manifest = json.loads((self.corpus / "manifest.json").read_text())
        want = (manifest["train_config_ids"] if split == "train"
                else manifest["held_out_config_ids"])
        runs = [r for r in manifest["runs"] if r["config_id"] in want]
        if not runs:
            raise ValueError(f"no runs for split {split}")

        xs, ys = [], []
        for r in runs:
            d = np.load(self.corpus / r["file"])
            E, C, S = d["served"].shape
            static = d["static"]  # (C, Fs)
            Fs = static.shape[1]
            dyn = np.stack([_log1p_norm(d[k]) for k in DYN_KEYS],
                           axis=-1)  # (E,C,S,D)
            tgt = np.stack([_log1p_norm(d[k]) for k in TARGET_KEYS],
                           axis=-1)  # (E,C,S,2)
            for t in range(H, E - F + 1):
                hist = dyn[t - H:t]                      # (H,C,S,D)
                fut = tgt[t:t + F]                       # (F,C,S,2)
                st = np.broadcast_to(static[:, None, :],
                                     (C, S, Fs))          # (C,S,Fs)
                so = np.zeros((C, S, S))                  # slice one-hot
                so[:, np.arange(S), np.arange(S)] = 1.0
                feat = np.concatenate(
                    [np.broadcast_to(hist, (H, C, S, len(DYN_KEYS))),
                     np.broadcast_to(st[None], (H, C, S, Fs)),
                     np.broadcast_to(so[None], (H, C, S, S))], axis=-1)
                # -> (H,C,S,Ftot); samples are (C,S) units
                xs.append(feat.reshape(H, C * S, -1).transpose(1, 0, 2))
                ys.append(fut.reshape(F, C * S, 2).transpose(1, 0, 2))
        self.x = np.concatenate(xs, axis=0).astype(np.float32)  # (N,H,Ft)
        self.y = np.concatenate(ys, axis=0).astype(np.float32)  # (N,F,2)

        if stats is None:  # fit on train
            self.x_mu = self.x.mean(axis=(0, 1))
            self.x_sd = self.x.std(axis=(0, 1)) + 1e-6
            self.y_mu = self.y.mean(axis=(0, 1))
            self.y_sd = self.y.std(axis=(0, 1)) + 1e-6
            # don't standardize one-hot slice dims (last S of features)
            self.x_sd[-S:] = 1.0
            self.x_mu[-S:] = 0.0
        else:
            self.x_mu, self.x_sd = stats["x_mu"], stats["x_sd"]
            self.y_mu, self.y_sd = stats["y_mu"], stats["y_sd"]
        self.stats = {"x_mu": self.x_mu, "x_sd": self.x_sd,
                      "y_mu": self.y_mu, "y_sd": self.y_sd}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = (self.x[i] - self.x_mu) / self.x_sd
        y = (self.y[i] - self.y_mu) / self.y_sd
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float())
