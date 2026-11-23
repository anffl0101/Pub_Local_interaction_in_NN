"""Loaders of the two terms of the one-step response (T1, T2) as whole matrices, per saved state and pair, for AdamW (adamw, all
512 networks) and SGLD (sgld, states with training accuracy below 0.5 and the networks of not_grokked() left out). Reads data/<set>/onestep_terms_frobenius_chunk<c>.npz
(scripts/runs/onestep_terms_frobenius.py). Used by comparisons_step_clock_text_numbers.py.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from src.utils import data_dir  # noqa: E402
from sgld_512_grokked import not_grokked  # noqa: E402

IN25 = np.abs(np.subtract.outer(range(5), range(5))) > 0
IN25[0], IN25[:, 0] = False, False


def load(name):
    Z = [np.load(data_dir(name) / f"onestep_terms_frobenius_chunk{c}.npz") for c in range(4)]
    cat = lambda k: np.concatenate([z[k] for z in Z])
    return np.concatenate([128 * c + z["member"] for c, z in enumerate(Z)]), cat("step"), cat("size")


def adamw():
    net, step, size = load("adamw_fullbatch_512")
    return net, step, np.ones(len(net), bool), size


def sgld():
    D = data_dir("sgld_512")
    net, step, size = load("sgld_512")
    acc = np.empty(len(net))
    for c in range(4):
        b = np.load(D / f"burnin_M128_chunk{c}.npz"); st = list(b["log_step"])
        for r in np.where(net // 128 == c)[0]:
            acc[r] = b["log_train_acc"][net[r] % 128, st.index(step[r])]
    keep = ~np.isin(net, [128 * c + m for c, m in not_grokked()]) & (acc >= 0.5)
    return net, step, keep, size
