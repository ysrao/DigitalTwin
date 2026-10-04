"""KPI predictor: compact shared-weight GRU with uncertainty.

Input: H=6 epochs of (cell, slice) history. Output: F=6-step forecasts of
served traffic and violation counts, as Gaussian mean + log-variance
(explicit uncertainty, per the brief). One model, shared across all
cells/slices — topology enters through the static cell features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .dataset import H, F


class KpiPredictor(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64, layers: int = 2,
                 n_targets: int = 2):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=layers,
                          batch_first=True)
        self.mu_head = nn.Linear(hidden, F * n_targets)
        self.lv_head = nn.Linear(hidden, F * n_targets)
        self.n_targets = n_targets

    def forward(self, x):
        _, h = self.gru(x)          # h: (layers, B, hidden)
        h = h[-1]                   # (B, hidden)
        mu = self.mu_head(h).view(-1, F, self.n_targets)
        lv = self.lv_head(h).view(-1, F, self.n_targets).clamp(-6, 6)
        return mu, lv


def gaussian_nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 / torch.exp(lv) + 0.6931).mean()


@torch.no_grad()
def predict_intervals(mu, lv, z: float = 1.645):
    sd = torch.exp(0.5 * lv)
    return mu - z * sd, mu + z * sd
