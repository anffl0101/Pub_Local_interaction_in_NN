"""Loader and statistics of the push results of the 512 full-batch AdamW networks (used by the figure and table scripts).
Reads kicks/kick_M128_chunk<c>_L<layer>_h<h>.npz, adamw_M128_chunk<c>.npz and grokking_onset_table.json in data/adamw_fullbatch_512;
stops unless the untouched networks equal the training logs bit for bit. load_adamw(every=True) keeps all 512 networks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import data_dir  # noqa: E402
import sgld_512_kick_response as sk  # noqa: E402,F401  (used by other scripts as K.sk)

EDGES = GROUPS = None          # bins of the push step and groups of layer pairs, set by the calling script
MIN_N, N_BOOT, CUT, LAST, CRASH_ACC = 50, 500, 0.8, 10_000, 0.5
D_ = data_dir("adamw_fullbatch_512")


def load_adamw(h="0.1", every=False):
    """Load the AdamW kicks at push strength h; return per-layer data, delays, chunks and left-out networks.
    Per kicked layer: dict(d (n, K, 5) copy - original, age, net, kstep)."""
    on = {r["init_seed"]: r["onset"] for r in json.loads((D_ / "grokking_onset_table.json").read_text())["networks"]}
    chunks = [c for c in range(4) if all((D_ / "kicks" / f"kick_M128_chunk{c}_L{L}_h{h}.npz").exists() for L in range(1, 6))]
    data, delays, dropped, matched = {}, None, [], []
    for L in range(1, 6):
        parts = []
        for c in chunks:
            f = np.load(D_ / "kicks" / f"kick_M128_chunk{c}_L{L}_h{h}.npz")
            log = np.load(D_ / f"adamw_M128_chunk{c}.npz")
            last = log["log_step"] > log["log_step"][-1] - LAST
            end = np.median(log["log_test_acc"][:, last], 1)
            seeds = 128 * c + np.arange(128)
            ok = np.array([on[s] is not None and end[m] >= CUT for m, s in enumerate(seeds)])
            keep = np.ones(128, bool) if every else ok      # every=True: also the networks that do not grok (age is NaN without an onset)
            if L == 1:
                dropped += [int(s) for s in seeds[~ok]]
            delays, ks = [int(t) for t in f["delays"]], f["kicks"]
            # hard stop: the untouched networks must equal the training log, bit for bit, wherever both exist
            step = {int(t): s for s, t in enumerate(log["log_step"])}
            hit = [(q, j, step[int(t + dd)]) for q, t in enumerate(ks) for j, dd in enumerate(delays) if int(t + dd) in step]
            logsr = np.stack([log[f"log_stable_rank_L{k}"] for k in range(1, 6)], -1)
            if not all(np.array_equal(f["sr_main"][q, j], logsr[:, s]) and np.array_equal(f["la_main"][q, j, 1], log["log_train_acc"][:, s])
                       and np.array_equal(f["la_main"][q, j, 3], log["log_test_acc"][:, s]) for q, j, s in hit):
                raise SystemExit(f"chunk {c} L{L}: the untouched networks differ from the training log")
            matched.append(len(hit))
            o = np.array([on[s] if on[s] is not None else np.nan for s in seeds])
            d = np.transpose(f["sr_kick"] - f["sr_main"], (0, 2, 1, 3))           # (Q, M, K, 5)
            lak = np.transpose(f["la_kick"], (0, 3, 1, 2))
            lam = np.transpose(f["la_main"], (0, 3, 1, 2))
            bad = np.logical_or.accumulate((lak[..., 1] < CRASH_ACC) | (lam[..., 1] < CRASH_ACC), axis=2)   # (Q, M, K) crash filter: from the first delay with training accuracy < 0.5 in either run
            d = np.where(bad[..., None], np.nan, d)
            Q = len(ks)
            sel = np.broadcast_to(keep, (Q, 128))
            parts.append(dict(d=d[sel], age=(ks[:, None] - o[None, :])[sel], net=np.broadcast_to(seeds, (Q, 128))[sel],
                              kstep=np.broadcast_to(ks[:, None], (Q, 128))[sel]))
        data[L] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    print(f"check passed: untouched networks = training log (stable ranks, training and test accuracy, bit for bit) "
          f"at {min(matched)}-{max(matched)} readouts in each of {len(matched)} files", flush=True)
    return data, delays, chunks, dropped


def bin_means(d, age, net, b, rng_seed):
    """Mean over the samples of step bin b of d (n, K, 5), and N_BOOT network-resampled means; NaN-aware."""
    s = (age >= EDGES[b]) & (age < EDGES[b + 1])
    if s.sum() < MIN_N:
        return None, None, int(s.sum()), 0
    x, nets = d[s], net[s]
    ids, inv = np.unique(nets, return_inverse=True)
    ok = np.isfinite(x)
    sums = np.zeros((len(ids),) + x.shape[1:]); np.add.at(sums, inv, np.where(ok, x, 0.0))
    cnt = np.zeros((len(ids),) + x.shape[1:]); np.add.at(cnt, inv, ok)
    draws = np.random.default_rng(rng_seed).integers(0, len(ids), size=(N_BOOT, len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums.sum(0) / cnt.sum(0), sums[draws].sum(1) / cnt[draws].sum(1), int(s.sum()), len(ids)


def stats(data, delays):
    """Per bin: mutual response of each group of pairs (per own change at the same delay, and at 32 steps), with 95% bands."""
    k32 = delays.index(32)
    nb, K = len(EDGES) - 1, len(delays)
    out = dict(nets=np.zeros(nb, int),
               group=np.full((len(GROUPS), nb, K, 3), np.nan), group32=np.full((len(GROUPS), nb, K, 3), np.nan))
    for b in range(nb):
        M, BM = [], []
        for L in range(1, 6):
            m, bm, n, nn = bin_means(data[L]["d"], data[L]["age"], data[L]["net"], b, 9_000_000 + b)
            M.append(m); BM.append(bm)
            out["nets"][b] = max(out["nets"][b], nn)
        if any(m is None for m in M):
            continue
        M, BM = np.stack(M), np.stack(BM, 1)                     # (5 kicked, K, 5), (B, 5, K, 5)
        own = np.stack([M[i, :, i] for i in range(5)])            # (5, K)
        ownb = np.stack([BM[:, i, :, i] for i in range(5)], 1)    # (B, 5, K)
        for norm in (None, k32):
            f = M / (own if norm is None else own[:, norm:norm + 1])[:, :, None]       # (5, K, 5) f[i, k, j]
            fb = BM / (ownb if norm is None else ownb[:, :, norm:norm + 1])[..., None]
            for g, pairs in enumerate(GROUPS.values()):
                gv = np.mean([np.sqrt(np.abs(f[i, :, j] * f[j, :, i])) for i, j in pairs], 0)
                gb = np.mean([np.sqrt(np.abs(fb[:, i, :, j] * fb[:, j, :, i])) for i, j in pairs], 0)
                out["group" + ("" if norm is None else "32")][g, b] = np.stack([gv, *np.nanpercentile(gb, (2.5, 97.5), axis=0)], -1)
    return out
