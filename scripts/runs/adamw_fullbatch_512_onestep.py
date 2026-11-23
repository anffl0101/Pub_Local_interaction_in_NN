"""One-step calculation at a saved AdamW state (`state_numbers`, float64): after an infinitely small push of W_i along the
gradient of its stable rank, the change of every layer's stable rank in the next AdamW update (Hessian-vector products),
the likeness of the pushes' effects on the scores, the line-up of neighbouring layers and AdamW's gain per layer.
`check`: one real AdamW update with a tiny push must reproduce the computed responses to 2% of the row maximum.
Used by adamw_fullbatch_512_saved_onestep.py and onestep_terms_frobenius.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import math

import torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vjp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
ETA_PLAIN = 0.1                                     # step size of the plain gradient step (SGLD's eta)
WK = [f"linears.{j}.weight" for j in range(5)]


def sr_grad(W: torch.Tensor) -> torch.Tensor:
    W = W.detach().requires_grad_(True)
    sr = (W ** 2).sum() / torch.linalg.matrix_norm(W, ord=2) ** 2
    (g,) = torch.autograd.grad(sr, W)
    return g


def state_numbers(p, m, v, t, T, x, y):
    """Push likeness (5,5), AdamW response total/first term (2,5,5) in %, line-up top-1/top-5 (4,), plain-step response (5,5) in %, gain (5,3).
    Gain per layer: median and 90th percentile of sqrt(v_hat)/eps, and the median of A / ((1-b1)/bc1/eps)."""
    loss = lambda q: F.cross_entropy(functional_call(T, q, (x,)), y)
    logits = lambda q: functional_call(T, q, (x,))
    g = grad(loss)(p)
    bc1, bc2 = 1 - B1 ** (t + 1), 1 - B2 ** (t + 1)
    A, gain = {}, np.zeros((5, 3))
    for j, k in enumerate(WK):
        mp, vp = B1 * m[k] + (1 - B1) * g[k], B2 * v[k] + (1 - B2) * g[k] ** 2
        den = vp.sqrt() / math.sqrt(bc2) + EPS
        A[k] = ((1 - B1) / bc1) / den - (mp / bc1) / den ** 2 * (1 - B2) * g[k] / (vp.sqrt().clamp_min(1e-30) * math.sqrt(bc2))
        r = (vp.sqrt() / math.sqrt(bc2) / EPS).flatten()
        gain[j] = [float(r.median()), float(r.quantile(0.9)), float((A[k] / ((1 - B1) / bc1 / EPS)).median())]
    d = [sr_grad(p[k]) for k in WK]
    f0, vjp_fn = vjp(logits, p)
    prob = torch.softmax(f0, 1).detach()
    E, C, Pl = [], np.zeros((2, 5, 5)), np.zeros((5, 5))
    for i in range(5):
        dv = {k: torch.zeros_like(tt) for k, tt in p.items()}
        dv[WK[i]] = d[i]
        _, Hv = jvp(grad(loss), (p,), (dv,))
        _, df = jvp(logits, (p,), (dv,))
        E.append(df)
        dp = prob * (df - (prob * df).sum(1, keepdim=True)) / len(y)
        (Jt,) = vjp_fn(dp)
        n2 = float((d[i] ** 2).sum())
        for j in range(5):
            C[0, i, j] = -LR * float((d[j] * A[WK[j]] * Hv[WK[j]]).sum()) / n2 * 100
            C[1, i, j] = -LR * float((d[j] * A[WK[j]] * Jt[WK[j]]).sum()) / n2 * 100
            Pl[i, j] = -ETA_PLAIN * float((d[j] * Hv[WK[j]]).sum()) / n2 * 100
    # G-inner products of the output changes: <a, b>_G = mean over examples of sum_k p a b - (p.a)(p.b)
    ip = lambda a, b: float(((prob * a * b).sum(1) - (prob * a).sum(1) * (prob * b).sum(1)).mean())
    Gm = np.array([[ip(E[i], E[j]) for j in range(5)] for i in range(5)])
    cos = Gm / np.sqrt(np.outer(np.diag(Gm), np.diag(Gm)))
    lu1, lu5 = np.zeros(4), np.zeros(4)
    for i in range(4):
        U = torch.linalg.svd(p[WK[i]], full_matrices=False)[0]
        Vh = torch.linalg.svd(p[WK[i + 1]], full_matrices=False)[2]
        lu1[i] = abs(float(U[:, 0] @ Vh[0]))
        lu5[i] = float(((U[:, :5].T @ Vh[:5].T) ** 2).sum()) / 5
    return cos, C, lu1, lu5, Pl, gain


def adamw_step(p, m, v, t, T, x, y):
    """One AdamW update from (p, m, v) at step t (weight decay 1.0), as src/batched_train._adamw_update."""
    g = grad(lambda q: F.cross_entropy(functional_call(T, q, (x,)), y))(p)
    bc1, bc2 = 1 - B1 ** (t + 1), math.sqrt(1 - B2 ** (t + 1))
    out = {}
    for k in p:
        mp, vp = B1 * m[k] + (1 - B1) * g[k], B2 * v[k] + (1 - B2) * g[k] ** 2
        out[k] = p[k] * (1 - LR * 1.0) - LR / bc1 * mp / (vp.sqrt() / bc2 + EPS)
    return out


def check(p, m, v, t, T, x, y, C, epss=(1e-7, 1e-8, 1e-9)):
    """Largest gap between a real AdamW update with push +-eps and C[0] over the 20 pairs, relative to the row maximum."""
    sr = lambda W: float((W ** 2).sum() / torch.linalg.matrix_norm(W, ord=2) ** 2)
    worst = 0.0
    for i in range(5):
        d = sr_grad(p[WK[i]]); n2 = float((d ** 2).sum())
        scale = max(abs(C[0, i, j]) for j in range(5) if j != i)
        best = np.inf
        for eps in epss:                          # best of three sizes: too large leaves the linear range, too small loses digits
            up = adamw_step({**p, WK[i]: p[WK[i]] + eps * d}, m, v, t, T, x, y)
            dn = adamw_step({**p, WK[i]: p[WK[i]] - eps * d}, m, v, t, T, x, y)
            diff = max(abs((sr(up[WK[j]]) - sr(dn[WK[j]])) / (2 * eps * n2) * 100 - C[0, i, j]) / scale
                       for j in range(5) if j != i)
            best = min(best, diff)
        worst = max(worst, best)
    return worst


CHECK_LIMIT = 0.02
