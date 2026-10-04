"""Sionna coupling-loss provider for arbitrary HetNet geometry.

Wraps the Stage-A-verified chain (UMa channel -> get_pathloss ->
coupling_loss_db) but with *explicit* topology tensors, so macro sectors AND
small cells share one code path. Sector antenna pattern (TR 38.901 Table
7.3-1, 65 deg 3dB) is applied analytically on top — Sionna's PanelArray here
uses single-element panels, so sectorization gain is our documented extension.

All UTs are marked outdoor (in_state=0): see Stage A report on the indoor
zero-gain edge case.
"""
from __future__ import annotations

import numpy as np
import torch

from sionna.phy.channel.tr38901 import PanelArray, UMa, UMi
from sionna.sys import get_pathloss, coupling_loss_db

_CLIP_DB = 150.0

# 3GPP 38.901 scenario per cell tier (outdoor deployments):
# macro (25 m BS) -> UMa; pico/femto (10 m BS) -> UMi-Street Canyon.
_TIER_SCENARIO = {"macro": "UMa", "pico": "UMi", "femto": "UMi"}
_SCENARIO_CLS = {"UMa": UMa, "UMi": UMi}


class CouplingProvider:
    def __init__(self, carrier_freq_hz: float = 3.5e9, device: str = "cpu",
                 seed: int = 0, scenario_kind: str = "UMa"):
        self.device = device
        self.seed = seed
        self.scenario_kind = scenario_kind
        self._n_calls = 0
        self.device = device
        bs_array = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=1,
            polarization="dual", polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_freq_hz, device=device)
        ut_array = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=1,
            polarization="single", polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_freq_hz, device=device)
        model_cls = _SCENARIO_CLS[scenario_kind]
        self._model = model_cls(
            carrier_frequency=carrier_freq_hz, o2i_model="low",
            ut_array=ut_array, bs_array=bs_array,
            direction="downlink", device=device)

    def compute(self, ut_xy: np.ndarray, cells: list) -> np.ndarray:
        """Coupling loss [dB], shape (n_ut, n_cell). Lower = better link."""
        # Seed Sionna's RNGs so fading is deterministic per (seed, call):
        # matched-seed policy comparisons need identical channel realizations.
        from sionna.phy import config as _sionna_config
        _sionna_config.seed = (self.seed * 100003 + self._n_calls) % 2**31
        self._n_calls += 1
        n_ut = len(ut_xy)
        n_cell = len(cells)
        dev = self.device

        ut_loc = torch.zeros(1, n_ut, 3, device=dev)
        ut_loc[0, :, 0] = torch.as_tensor(ut_xy[:, 0])
        ut_loc[0, :, 1] = torch.as_tensor(ut_xy[:, 1])
        ut_loc[0, :, 2] = 1.5

        bs_loc = torch.zeros(1, n_cell, 3, device=dev)
        for i, c in enumerate(cells):
            bs_loc[0, i, 0] = c["x"]
            bs_loc[0, i, 1] = c["y"]
            bs_loc[0, i, 2] = c["height_m"]

        z3u = torch.zeros(1, n_ut, 3, device=dev)
        z3c = torch.zeros(1, n_cell, 3, device=dev)
        in_state = torch.zeros(1, n_ut, dtype=torch.bool, device=dev)  # all outdoor
        bs_site_ids = torch.arange(n_cell, device=dev)
        # [batch, n_ut]: accepted by both the legacy (<6 GHz) and the new
        # (>=6 GHz, FR3) O2I indoor-distance paths. All UTs are outdoor, so
        # these zeros never enter the pathloss.
        d2d_in = torch.zeros(1, n_ut, device=dev)

        self._model.set_topology(
            ut_loc, bs_loc, z3u, z3c, z3u, in_state, None,
            None, bs_site_ids, None, d2d_in)
        h, _ = self._model(num_time_samples=1, sampling_frequency=1e6)
        pl_lin, _ = get_pathloss(h)  # [1, U, C, S], linear
        pl_lin = torch.clamp(pl_lin, max=10.0 ** (_CLIP_DB / 10.0))
        pl_db = 10.0 * torch.log10(pl_lin).mean(dim=-1)[0]  # [U, C]
        coupling = coupling_loss_db(-pl_db).numpy()  # [U, C]

        # analytic sector pattern for macro cells (omni small cells: 0 dB)
        for i, c in enumerate(cells):
            az = c["azimuth_deg"]
            if az is None:
                continue
            dx = ut_xy[:, 0] - c["x"]
            dy = ut_xy[:, 1] - c["y"]
            ang = (np.degrees(np.arctan2(dy, dx)) - az + 180.0) % 360.0 - 180.0
            pattern_db = -np.minimum(12.0 * (ang / 65.0) ** 2, 30.0)
            coupling[:, i] -= pattern_db  # pattern loss adds to coupling loss
        return coupling
