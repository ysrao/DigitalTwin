"""Predictor v3: FiLM-conditioned forecaster.

The v2 model learned to zero out its late-fusion action path. Here the
planned budget action *modulates* the GRU hidden state directly
(Feature-wise Linear Modulation): the network cannot produce its forecast
without passing through action-dependent scaling, which forces the
action path to carry gradient.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .dataset_v2 import H, F


class FiLMPredictor(nn.Module):
    def __init__(self, hist_dim: int, n_slices: int, hidden: int = 64,
                 layers: int = 2, n_targets: int = 2):
        super().__init__()
        self.gru = nn.GRU(hist_dim, hidden, num_layers=layers,
                          batch_first=True)
        self.film = nn.Sequential(
            nn.Linear(F * n_slices, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * hidden))
        self.mu_head = nn.Linear(hidden, F * n_targets)
        self.lv_head = nn.Linear(hidden, F * n_targets)
        self.n_targets = n_targets

    def forward(self, xh, xa):
        _, h = self.gru(xh)
        h = h[-1]
        gamma, beta = self.film(xa).chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(gamma)) + beta
        mu = self.mu_head(h).view(-1, F, self.n_targets)
        lv = self.lv_head(h).view(-1, F, self.n_targets).clamp(-6, 6)
        return mu, lv


def gaussian_nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 / torch.exp(lv) + 0.6931).mean()
