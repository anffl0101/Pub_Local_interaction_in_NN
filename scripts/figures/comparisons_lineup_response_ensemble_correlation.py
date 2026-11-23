"""Helpers for the correlation across the networks of a pair's line-up and its one-step mutual response: chi_ij of one state
(response), Pearson correlation (rho) and the Benjamini-Hochberg rule (bh). Used by comparisons_step_clock_lineup.py.
"""
import numpy as np

MIN_STATES = 40


def response(R, i, j, kr, k0):
    """chi_ij of one state, sqrt|chi_j<-i chi_i<-j|, each push divided by the own change of its pushed layer."""
    with np.errstate(all="ignore"):
        return np.sqrt(np.abs((R[:, i, kr, j] / R[:, i, k0, i]) * (R[:, j, kr, i] / R[:, j, k0, j])))


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def bh(p, q=0.05):
    """Benjamini-Hochberg: which of the p-values are significant at level q."""
    p = np.asarray(p, float); ok = np.isfinite(p); out = np.zeros(len(p), bool)
    order = np.argsort(p[ok]); m = ok.sum()
    passed = p[ok][order] <= q * (np.arange(m) + 1) / m
    if passed.any():
        out[np.where(ok)[0][order[: np.max(np.where(passed)[0]) + 1]]] = True
    return out
