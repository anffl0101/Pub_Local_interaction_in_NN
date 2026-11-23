"""Loads the SGLD push results: change of every layer's stable rank after one layer is pushed. Without step_clock_all_networks,
the 4 networks of sgld_512_grokked.not_grokked() are left out and age = push step minus the grokking onset; with it, all 512 networks
enter and age = push step. Reads kick_M128_chunk<c>_L<i>*.npz and burnin_M128_chunk<c>.npz from data/sgld_512/.
Used through load() by the figure scripts; filtered=True applies the crash filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sgld_512_grokked import not_grokked   # noqa: E402
from grokking_onset import sgld_onsets   # noqa: E402

D = ROOT / "data" / "sgld_512"
CHUNKS = (0, 1, 2, 3)
LAYERS = (1, 2, 3, 4, 5)
LA = {"train_loss": 0, "train_acc": 1, "test_loss": 2, "test_acc": 3}
# crash filter: drop a pair once training accuracy falls below 0.5 in either run 
CRASH_ACC = 0.5
DENSE_SUFFIX = "_h0.1_d20"                 # h = 0.1 copies recorded every 20 steps; other strengths only at the 15 delays


def crash_keep(f, fields, c: int, L: int) -> np.ndarray:
    """(Q, F, K, M): True where neither run crashed at any recorded moment up to delay K."""
    bad = (f["la_kick"][:, :, :, LA["train_acc"], :] < CRASH_ACC) \
        | (f["la_main"][:, None, :, LA["train_acc"], :] < CRASH_ACC)
    taus = np.asarray(f["kick_taus"])
    for fi in np.where(np.isclose(fields, 0.1))[0]:
        g = np.load(D / f"kick_M128_chunk{c}_L{L}{DENSE_SUFFIX}.npz")
        if not (np.array_equal(g["kick_steps"], f["kick_steps"])
                and np.array_equal(g["la_kick"][:, 0], f["la_kick"][:, fi])):
            raise RuntimeError(f"dense file chunk{c}_L{L} is not the same h = 0.1 copy")
        dt = int(g["dense_every"]); n = g["la_kick_dense"].shape[2]
        idx = np.searchsorted(f["log_step"], f["kick_steps"][:, None] + dt * np.arange(1, n + 1))
        orig = np.transpose(f["log_train_acc"][:, idx], (1, 2, 0))           # (Q, n, M)
        dense = (g["la_kick_dense"][:, 0, :, LA["train_acc"], :] < CRASH_ACC) | (orig < CRASH_ACC)
        cum = np.logical_or.accumulate(dense, axis=1)
        j = taus // dt - 1                                                     # last dense point <= tau
        bad[:, fi] |= np.where((j >= 0)[None, :, None], cum[:, np.maximum(j, 0)], False)
    return ~np.logical_or.accumulate(bad, axis=2)


def load(suffix: str = "", filtered: bool = False):
    onset = sgld_onsets()
    data = {}
    taus = fields = None
    for L in LAYERS:
        age, d, net, kstep, keep = [], [], [], [], []
        for c in CHUNKS:
            f = np.load(D / f"kick_M128_chunk{c}_L{L}{suffix}.npz")
            b = np.load(D / f"burnin_M128_chunk{c}.npz")
            # check: the untouched network must equal the training log bit for bit
            for k in ["log_step", "log_test_acc"] + [f"log_stable_rank_L{j}" for j in LAYERS]:
                if not np.array_equal(f[k], b[k]):
                    raise RuntimeError(f"CHECK FAILED: chunk {c} L{L} {k} differs from the training log")
            if int(f["field_layer"]) != L - 1:
                raise RuntimeError(f"file chunk{c}_L{L} kicked layer index {f['field_layer']} != {L-1}")
            taus, fields = f["kick_taus"], f["kick_fields"]
            ks = f["kick_steps"]                                   # (Q,)
            o = np.array([onset[(c, m)] for m in range(128)])
            dd = f["sr_kick"] - f["sr_main"][:, None]              # (Q,F,K,M,5)
            Q, M = len(ks), 128
            age.append((ks[:, None] - o[None, :]).ravel())
            d.append(np.transpose(dd, (0, 3, 1, 2, 4)).reshape(Q * M, len(fields), len(taus), 5))
            net.append(np.broadcast_to(c * 128 + np.arange(M), (Q, M)).ravel())
            kstep.append(np.broadcast_to(ks[:, None], (Q, M)).ravel())
            if filtered:
                keep.append(np.transpose(crash_keep(f, fields, c, L), (0, 3, 1, 2)).reshape(Q * M, len(fields), len(taus)))
        data[L] = dict(age=np.concatenate(age), d=np.concatenate(d), net=np.concatenate(net), kstep=np.concatenate(kstep))
        if filtered:
            k = np.concatenate(keep)
            data[L]["keep"] = k
            data[L]["d"][~k] = np.nan
        out = np.isin(data[L]["net"], [c * 128 + m for c, m in not_grokked()])   # the networks that did not grok (none with step_clock_all_networks)
        data[L] = {k: v[~out] for k, v in data[L].items()}
    print(f"check passed: {5 * len(CHUNKS)} untouched runs = training logs "
          f"(files kick_*{suffix}.npz)")
    return data, taus, fields
