"""Action-conditioned KPI predictor v2.

GRU encodes the KPI history; a small MLP encodes the planned future budget
vector for the cell; concatenated, they feed Gaussian mean + log-variance
heads forecasting F steps of served traffic and violations.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .dataset_v2 import H, F


class ActionPredictor(nn.Module):
    def __init__(self, hist_dim: int, n_slices: int, hidden: int = 64,
                 layers: int = 2, act_dim: int = 32, n_targets: int = 2):
        super().__init__()
        self.gru = nn.GRU(hist_dim, hidden, num_layers=layers,
                          batch_first=True)
        self.act_mlp = nn.Sequential(
            nn.Linear(F * n_slices, act_dim), nn.ReLU(),
            nn.Linear(act_dim, act_dim), nn.ReLU())
        self.mu_head = nn.Linear(hidden + act_dim, F * n_targets)
        self.lv_head = nn.Linear(hidden + act_dim, F * n_targets)
        self.n_targets = n_targets

    def forward(self, xh, xa):
        _, h = self.gru(xh)
        h = h[-1]
        a = self.act_mlp(xa)
        z = torch.cat([h, a], dim=-1)
        mu = self.mu_head(z).view(-1, F, self.n_targets)
        lv = self.lv_head(z).view(-1, F, self.n_targets).clamp(-6, 6)
        return mu, lv


def gaussian_nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 / torch.exp(lv) + 0.6931).mean()
