"""Mutual response of each of the six pairs among layers 2-5 per network (pair_values), with the loader and crash filter of the
full-order table. Used by comparisons_strength_free_table.py.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import step_clock_all_networks  # noqa: E402,F401  (first: all 512 networks, clock = training step)
from comparisons_step_clock_pushes import fo  # noqa: E402

warnings.simplefilter("ignore")
NAMES = ["2-3", "3-4", "4-5", "2-4", "3-5", "2-5"]          # order of fo.PAIRS


def pair_values(D, delays, kr, ko, lo, hi):
    """Mutual response of the six pairs for each network, as in fo.orders: {network: (6,) array in the order NAMES}."""
    out = {}
    for n in np.unique(D[1]["net"]):
        M = []
        for L in range(1, 6):
            s = (D[L]["net"] == n) & (D[L]["age"] >= lo) & (D[L]["age"] < hi)
            if not s.any():
                break
            M.append(np.nanmean(D[L]["d"][s], 0))
        if len(M) < 5:
            continue
        M = np.stack(M)
        f = M[:, kr, :] / np.stack([M[i, ko, i] for i in range(5)])[:, None]
        v = np.array([np.sqrt(abs(f[i, j] * f[j, i])) for i, j in fo.PAIRS])
        if np.isfinite(v).all():
            out[int(n)] = v
    return out
