"""Line-up |u_1(W_i).v_1(W_i+1)| and push response per state, and the helpers that cut each step bin into line-up groups.
Reads the push results, data/adamw_fullbatch_512/lineup_states.npz and data/sgld_512/lineup.npz.
Used through load(), groups(), chi(), band() and band_mean() by comparisons_step_clock_lineup.py.
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
with contextlib.redirect_stdout(io.StringIO()):
    import sgld_512_kick_response as sk
    import adamw_fullbatch_512_kicks as ak

NQ, NBOOT = 4, 500


def table(D, z, keep):
    """per state: net, age, line (4 pairs), R (pushed layer, delay, responder) raw changes."""
    age_of = {(int(n), int(s)): float(a) for L in D for n, s, a in zip(D[L]["net"], D[L]["kstep"], D[L]["age"])}
    keys = []
    for r, (c, m, s) in enumerate(zip(z["chunk"], z["member"], z["step"])):
        n, s = int(c) * 128 + int(m), int(s)
        if (n, s) in age_of and np.isfinite(age_of[(n, s)]) and keep[r]:
            keys.append((n, s, age_of[(n, s)], z["line"][r]))
    idx = {(k[0], k[1]): r for r, k in enumerate(keys)}
    K = D[1]["d"].shape[1]
    R = np.full((len(keys), 5, K, 5), np.nan)
    for L in D:
        for n, s, d in zip(D[L]["net"], D[L]["kstep"], D[L]["d"]):
            r = idx.get((int(n), int(s)))
            if r is not None:
                R[r, L - 1] = d
    return dict(net=np.array([k[0] for k in keys]), age=np.array([k[2] for k in keys]),
                line=np.array([k[3] for k in keys]), R=R)


def one_way(R, i, j, kr, ko):
    """mean change of layer j at delay kr / mean own change of pushed layer i at ko, each over its own pushes kept by the crash filter."""
    a, b = R[:, i, kr, j], R[:, i, ko, i]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return a.mean() / b.mean() if min(len(a), len(b)) >= 5 else np.nan


def chi(R, i, j, kr, ko):
    f, g = one_way(R, i, j, kr, ko), one_way(R, j, i, kr, ko)
    return 100 * np.sqrt(abs(f * g))


def band(R, net, fn, rng):
    ids, inv = np.unique(net, return_inverse=True); groups = [np.where(inv == k)[0] for k in range(len(ids))]
    v = [fn(R[np.concatenate([groups[k] for k in rng.integers(0, len(ids), len(ids))])]) for _ in range(NBOOT)]
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def band_mean(v, net, rng):
    """95% range of the mean of v from resampling the networks."""
    ids, inv = np.unique(net, return_inverse=True); rows = [np.where(inv == k)[0] for k in range(len(ids))]
    m = [v[np.concatenate([rows[k] for k in rng.integers(0, len(ids), len(ids))])].mean() for _ in range(NBOOT)]
    return [float(x) for x in np.percentile(m, [2.5, 97.5])]


def load():
    Dw, delays, _, _ = ak.load_adamw("0.1")
    zw = np.load(ROOT / "data/adamw_fullbatch_512/lineup_states.npz")
    data, taus, fields = sk.load("_h0.03-0.1", filtered=True)
    fi = int(np.where(np.isclose(fields, 0.1))[0][0])
    Ds = {L: dict(net=data[L]["net"], kstep=data[L]["kstep"], age=data[L]["age"], d=data[L]["d"][:, fi]) for L in data}
    zs = np.load(ROOT / "data/sgld_512/lineup.npz")
    return {"AdamW": (table(Dw, zw, np.ones(len(zw["step"]), bool)), list(delays)),
            "SGLD": (table(Ds, zs, zs["acc"] >= 0.5), list(taus))}


def groups(T, p, E):
    """line-up group of pair p inside each step bin; -1 outside the bins."""
    g = np.full(len(T["age"]), -1)
    for lo, hi in zip(E[:-1], E[1:]):
        s = np.where((T["age"] >= lo) & (T["age"] < hi))[0]
        if len(s) >= 2 * NQ:
            r = np.argsort(np.argsort(T["line"][s, p], kind="stable"), kind="stable")
            g[s] = (r * NQ) // len(s)
    return g
