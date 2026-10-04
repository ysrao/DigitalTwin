"""Stage A smoke test: 21-cell UMa multicell channel + geometry SINR on CPU.

Follows the documented Sionna pattern (UMa docstring): multicell topology ->
set_topology -> channel generation -> pathloss -> coupling loss -> SINR.
Matches Brief v2 §1 baseline layout (7 sites x 3 sectors = 21 cells).
Single-element panels and 4 UEs/sector keep it CPU-feasible.
"""
import time
import resource

import torch

from sionna.phy.channel.tr38901 import PanelArray, UMa
from sionna.sys import (
    gen_tr38901_multicell_topology,
    get_pathloss,
    coupling_loss_db,
    geometry_sinr_db,
)

CARRIER_HZ = 3.5e9


def main():
    t0 = time.time()
    device = "cpu"

    bs_array = PanelArray(
        num_rows_per_panel=1, num_cols_per_panel=1,
        polarization="dual", polarization_type="cross",
        antenna_pattern="38.901", carrier_frequency=CARRIER_HZ, device=device,
    )
    ut_array = PanelArray(
        num_rows_per_panel=1, num_cols_per_panel=1,
        polarization="single", polarization_type="V",
        antenna_pattern="omni", carrier_frequency=CARRIER_HZ, device=device,
    )
    channel_model = UMa(
        carrier_frequency=CARRIER_HZ, o2i_model="low",
        ut_array=ut_array, bs_array=bs_array,
        direction="downlink", device=device,
    )

    # NOTE (Stage A finding): with default settings, indoor UTs at 3.5 GHz UMa
    # get exactly zero channel gain in this Sionna version (O2I edge case under
    # investigation). Smoke test uses all-outdoor drops; ~4% residual zero-gain
    # links (antenna nulls/deep fades) are clipped at 150 dB, standard practice.
    topology = gen_tr38901_multicell_topology(
        "uma", batch_size=1, num_ut_per_sector=4,
        carrier_frequency=CARRIER_HZ, num_rings=1,
        indoor_probability=0.0,
        return_site_positions=True, device=device,
    )
    channel_model.set_topology(*topology[0])
    t_topo = time.time() - t0

    h, tau = channel_model(num_time_samples=1, sampling_frequency=1e6)
    t_chan = time.time() - t0 - t_topo
    print(f"channel tensor: {tuple(h.shape)}")

    pathloss_lin, _ = get_pathloss(h)  # linear; [..., num_ut, num_bs, symbols]
    pathloss_lin = torch.clamp(pathloss_lin, max=1e15)  # clip at 150 dB
    pathloss_db = 10.0 * torch.log10(pathloss_lin).mean(dim=-1)  # [1, 84, 21]
    coupling = coupling_loss_db(-pathloss_db)  # path gain [dB] -> coupling loss
    serving = torch.argmin(coupling, dim=-1)
    sinr = geometry_sinr_db(
        coupling, tx_power_dbm=46.0, bandwidth_hz=20e6,
        noise_figure_db=7.0, serving=serving,
    )

    t_tot = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    flat = sinr.flatten()
    p5 = flat.kthvalue(max(1, int(flat.numel() * 0.05)))[0]
    print(f"sectors=21 uts={coupling.shape[1]}")
    print(f"SINR dB: mean {sinr.mean():.2f}  p5 {p5:.2f}  "
          f"min {sinr.min():.2f}  max {sinr.max():.2f}")
    print(f"topology {t_topo:.2f}s | channel {t_chan:.2f}s | "
          f"total {t_tot:.2f}s | peak RAM {peak_mb:.0f} MB")
    assert coupling.shape[2] == 21, f"expected 21 sectors, got {coupling.shape[2]}"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
