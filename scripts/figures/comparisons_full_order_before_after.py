"""Per-network order of the mutual response by distance among layers 2-5 (orders) and the share with a 95% bootstrap interval (ci).
Used by the push and table scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

PAIRS = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)]


def orders(D, delays, k_resp, k_own, lo, hi):
    out = {}
    for n in np.unique(D[1]["net"]):
        M = []
        for L in range(1, 6):
            s = (D[L]["net"] == n) & (D[L]["age"] >= lo) & (D[L]["age"] < hi)
            if not s.any():
                break
            with np.errstate(all="ignore"):
                M.append(np.nanmean(D[L]["d"][s], 0))
        if len(M) < 5:
            continue
        M = np.stack(M)
        f = M[:, k_resp, :] / np.stack([M[i, k_own, i] for i in range(5)])[:, None]
        v = np.array([np.sqrt(abs(f[i, j] * f[j, i])) for i, j in PAIRS])
        if np.isfinite(v).all():
            out[int(n)] = (bool(v[:3].mean() > v[3:5].mean() > v[5]), bool(v[:3].mean() > v[5]))
    return out


def ci(x, rng):
    b = rng.choice(x, (10000, len(x))).mean(1)
    return [float(100 * x.mean()), float(100 * np.percentile(b, 2.5)), float(100 * np.percentile(b, 97.5))]
