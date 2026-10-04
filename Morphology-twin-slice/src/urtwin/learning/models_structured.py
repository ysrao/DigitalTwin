"""Gray-box (structured) predictor.

The black-box v2/v3 models learned to zero out their action path: nothing
in their architecture *requires* the budget to matter. Here the physics is
structural — per (cell, slice, epoch):

    served = min(demand, budget_prbs * spectral_efficiency)

The GRU predicts (demand, spectral efficiency, violations) from history;
the planned budget enters analytically through the min(). The network
cannot ignore the action because its output is a direct function of it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .dataset_v2 import H, F


class StructuredPredictor(nn.Module):
    def __init__(self, hist_dim: int, n_slices: int, hidden: int = 64,
                 layers: int = 2):
        super().__init__()
        self.n_slices = n_slices
        self.gru = nn.GRU(hist_dim, hidden, num_layers=layers,
                          batch_first=True)
        self.demand_head = nn.Linear(hidden, F)   # log1p bits wanted
        self.se_head = nn.Linear(hidden, F)       # log1p bits per PRB
        self.viol_head = nn.Linear(hidden, F)     # log1p violations
        self.demand_lv = nn.Linear(hidden, F)     # uncertainty (fallback)
        # action standardization buffers (model takes raw PRB budgets)
        self.register_buffer("a_mu", torch.zeros(F * n_slices))
        self.register_buffer("a_sd", torch.ones(F * n_slices))

    def set_action_stats(self, mu, sd):
        self.a_mu.copy_(torch.from_numpy(mu).float())
        self.a_sd.copy_(torch.from_numpy(sd).float())

    def forward(self, xh, xa_raw):
        """xa_raw: (B, F*S) planned budgets in real PRBs."""
        _, h = self.gru(xh)
        h = h[-1]
        s = xh[:, 0, -self.n_slices:].argmax(dim=-1)  # slice idx (B,)
        bud = xa_raw.view(-1, F, self.n_slices)
        bud_s = bud[torch.arange(bud.size(0)), :, s]  # (B, F) this slice
        dem = self.demand_head(h).clamp(0, 15)
        se = self.se_head(h).clamp(0, 12)
        dem_r = torch.expm1(dem)
        se_r = torch.expm1(se)
        served_r = torch.minimum(dem_r, bud_s * se_r)
        served = torch.log1p(served_r)
        viol = self.viol_head(h)
        dlv = self.demand_lv(h).clamp(-6, 6)
        return served, viol, dlv, se, dem


def structured_loss(served, viol, y):
    # y: (B, F, 2) standardized log1p targets — unstandardize outside
    return ((served - y[:, :, 0]) ** 2).mean() + \
           ((viol - y[:, :, 1]) ** 2).mean()
