"""Push results at the strengths 0.1, 0.03 and 0.01 for AdamW and SGLD, on the same push steps at every strength, with the
crash filter (training accuracy below 0.5 in either run) applied jointly to the three strengths; table_ci gives the per-network
order counts with 95% bootstrap intervals. Used by comparisons_step_clock_pushes.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adamw_fullbatch_512_kicks as ak  # noqa: E402
import sgld_512_kick_response as sk  # noqa: E402
import comparisons_full_order_before_after as fo  # noqa: E402

H = ["0.1", "0.03", "0.01"]
# AdamW kick times run at all three strengths
ADAMW_KICKS = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 40000, 80000, 145000]


def joint(per_h):
    """Keep a (push, network, delay) only where every strength has it (NaN elsewhere)."""
    for L in range(1, 6):
        for h in H:
            if not (np.array_equal(per_h[h][L]["net"], per_h[H[0]][L]["net"])
                    and np.array_equal(per_h[h][L]["kstep"], per_h[H[0]][L]["kstep"])):
                raise SystemExit(f"h={h} L{L}: samples not in the same order as h={H[0]}")
    for L in range(1, 6):
        bad = np.any([np.isnan(per_h[h][L]["d"]).any(-1) for h in H], 0)
        for h in H:
            per_h[h][L]["d"] = np.where(bad[..., None], np.nan, per_h[h][L]["d"])
    return per_h


def adamw():
    out = {}
    for h in H:
        A, delays, chunks, _ = ak.load_adamw(h)
        if len(chunks) != 4:
            raise SystemExit(f"AdamW h={h}: only chunks {chunks}")
        out[h] = {L: {k: v[np.isin(A[L]["kstep"], ADAMW_KICKS)] for k, v in A[L].items()} for L in A}
    return joint(out), delays


def sgld():
    a, taus, fa = sk.load("", filtered=True)                 # 1e-4, 1e-3, 1e-2 (13 kick times)
    b, _, fb = sk.load("_h0.03-0.1", filtered=True)          # 0.03, 0.1 (29 kick times)
    shared = np.unique(a[1]["kstep"])
    out = {}
    for h in H:
        src, fields = (a, fa) if float(h) < 0.03 else (b, fb)
        fi = int(np.argmin(np.abs(np.asarray(fields) - float(h))))
        if not np.isclose(fields[fi], float(h)):
            raise SystemExit(f"SGLD h={h} not found")
        out[h] = {}
        for L in range(1, 6):
            s = np.isin(src[L]["kstep"], shared)
            o = np.lexsort((src[L]["net"][s], src[L]["kstep"][s]))     # same order in both sets of runs
            out[h][L] = dict(d=src[L]["d"][s][o][:, fi], age=src[L]["age"][s][o], net=src[L]["net"][s][o],
                             kstep=src[L]["kstep"][s][o])
    return joint(out), [int(t) for t in taus]


def table_ci(D, delays, win, tau=512, tau_own=32):
    """Shares of networks with the full order and with neighbours > 3 apart, with 95% ranges from resampling."""
    o = fo.orders(D, delays, delays.index(tau), delays.index(tau_own), *win)
    rng = np.random.default_rng(0)
    return dict(networks=len(o), full_order=fo.ci(np.array([v[0] for v in o.values()], float), rng),
                neighbours_gt_3apart=fo.ci(np.array([v[1] for v in o.values()], float), rng))
