"""Modular-addition dataset (a + b) mod p with a seeded train/test split.

Each input is the one-hot vector of a joined to the one-hot vector of b (length 2p).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DataConfig:
    p: int = 113  # modulus
    train_fraction: float = 0.4  # fraction of all p*p pairs used for training
    seed: int = 0  # controls the train/test split


def make_modular_addition(cfg: DataConfig):
    """Return a dict with the train, test and full (x_all, y_all) tensors, n_train, n_total and p.

    Inputs x_* have shape (n, 2p) float32; targets y_* have shape (n,) int64.
    """
    p = cfg.p
    a = np.repeat(np.arange(p), p)  # 0,0,...,1,1,... each repeated p times
    b = np.tile(np.arange(p), p)  # 0,1,...,p-1,0,1,...
    y = (a + b) % p

    eye = np.eye(p, dtype=np.float32)
    x = np.concatenate([eye[a], eye[b]], axis=1)

    rng = np.random.default_rng(cfg.seed)
    n_total = p * p
    perm = rng.permutation(n_total)
    n_train = int(round(cfg.train_fraction * n_total))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y.astype(np.int64))
    return {
        "x_train": x_t[train_idx],
        "y_train": y_t[train_idx],
        "x_test": x_t[test_idx],
        "y_test": y_t[test_idx],
        "x_all": x_t,
        "y_all": y_t,
        "n_train": n_train,
        "n_total": n_total,
        "p": p,  # the model reads its output width from this
    }

if __name__ == "__main__": 
    pass