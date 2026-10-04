"""NR numerology helpers.

N_PRB from 3GPP TS 38.104 Table 5.3.2-1 (FR1). TDD DL fraction parsed from
the slot pattern (D=1.0, S=0.5, U=0.0 of the slot usable for DL).
"""
from ..scenarios.loader import Carrier

# (scs_khz, bw_mhz) -> N_PRB  — FR1, TS 38.104 Table 5.3.2-1
_N_PRB = {
    (15, 5): 25, (15, 10): 52, (15, 15): 79, (15, 20): 106, (15, 25): 133,
    (15, 30): 160, (15, 35): 188, (15, 40): 216, (15, 45): 242, (15, 50): 270,
    (30, 5): 11, (30, 10): 24, (30, 15): 38, (30, 20): 51, (30, 25): 65,
    (30, 30): 78, (30, 35): 92, (30, 40): 106, (30, 45): 120, (30, 50): 133,
    (30, 60): 162, (30, 70): 189, (30, 80): 217, (30, 90): 245, (30, 100): 273,
    (60, 10): 11, (60, 15): 18, (60, 20): 24, (60, 25): 31, (60, 30): 38,
    (60, 35): 44, (60, 40): 51, (60, 45): 58, (60, 50): 65, (60, 60): 79,
    (60, 70): 93, (60, 80): 107, (60, 90): 121, (60, 100): 135,
    # FR3 / 5G-Advanced study band (3GPP Rel-18/19, 7 GHz upper mid-band):
    # 200 MHz @ 60 kHz SCS. Not a published 38.104 entry — value follows the
    # 38.104 guard-band ratio (200e6 / (12*60e3) = 277.7 -> 273, same as
    # 100 MHz @ 30 kHz -> 273).
    (60, 200): 273,
}

_SCS = {0: 15, 1: 30, 2: 60, 3: 120, 4: 240}


def scs_khz(numerology: int) -> int:
    return _SCS[numerology]


def slot_duration_ms(numerology: int) -> float:
    return 1.0 / (2 ** numerology)


def n_prb(carrier: Carrier) -> int:
    key = (scs_khz(carrier.numerology), carrier.bw_mhz)
    if key not in _N_PRB:
        raise ValueError(f"no 38.104 FR1 entry for SCS {key[0]} kHz, BW {key[1]} MHz")
    return _N_PRB[key]


def tdd_dl_fraction(pattern: str) -> float:
    """Fraction of slots usable for DL from a pattern like 'DDDSU'."""
    w = {"D": 1.0, "S": 0.5, "U": 0.0}
    slots = [s for s in pattern.upper() if s in w]
    if not slots:
        raise ValueError(f"bad TDD pattern '{pattern}'")
    return sum(w[s] for s in slots) / len(slots)


def usable_dl_prbs(carrier: Carrier) -> float:
    n = n_prb(carrier)
    if carrier.duplex == "FDD":
        return float(n)
    return n * tdd_dl_fraction(carrier.tdd_pattern)


def dl_prb_units_per_grid_slot(carrier: Carrier, grid_mu: int) -> float:
    """DL PRB budget of `carrier` per simulator grid slot.

    The simulator runs on the grid of carriers[0] (numerology grid_mu,
    slot 1/2**grid_mu ms). A carrier with finer numerology mu_c packs
    2**(mu_c - grid_mu) of its own slots into one grid slot, each with
    n_prb PRBs — the pooled scheduler counts these as uniform PRB units
    (bits per unit per grid slot = SE * 180 for any numerology, since
    12 * SCS * slot_duration is numerology-invariant over a fixed window).
    Requires carrier.numerology >= grid_mu.
    """
    mu_c = carrier.numerology
    if mu_c < grid_mu:
        raise ValueError(
            f"carrier {carrier.cc_id}: numerology {mu_c} finer than grid "
            f"{grid_mu} required (carriers[0] sets the coarsest grid)")
    return n_prb(carrier) * (2 ** (mu_c - grid_mu)) * (
        1.0 if carrier.duplex == "FDD"
        else tdd_dl_fraction(carrier.tdd_pattern))
