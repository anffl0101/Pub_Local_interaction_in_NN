"""Line-up of neighbouring layers in the SGLD networks, from the saved burn-in states (every 20,000 steps, all 512 networks).
Per state: |u_1(W_i).v_1(W_i+1)| for the four neighbour pairs, with random and cross-network levels, the strengths of the strongest
patterns, and the inner products of the score changes made by the stable-rank pushes (key `gram`).
Writes data/sgld_512/lineup.npz (delete it to recompute). Run:  uv run python scripts/figures/sgld_512_lineup.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import DataConfig, make_modular_addition  # noqa: E402
from src.utils import build_config, load_config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grokking_onset import sgld_onsets  # noqa: E402

D = ROOT / "data" / "sgld_512"
CK = Path(os.environ.get("GROKKING_CKPT", ROOT / "checkpoints"))      # folder of the saved states
CHUNKS = (0, 1, 2, 3)
SEED = 0                                   # random unit vectors for the random level


def forward(p, x):
    """hs[k] = input of layer k; masks[k] = ReLU on/off after layer k (k < 4); logits."""
    hs, masks, h = [x], [], x
    for k in range(5):
        z = h @ p[f"linears.{k}.weight"].T + p[f"linears.{k}.bias"]
        if k == 4:
            return hs, masks, z
        masks.append(z > 0)
        h = z * masks[-1]
        hs.append(h)


def prop(dz, k, p, masks):
    """Change of the logits from a change dz of layer k's pre-activation, on/off pattern fixed."""
    for l in range(k, 4):
        dz = (dz * masks[l]) @ p[f"linears.{l + 1}.weight"].T
    return dz


def gram(E, prob):
    """<E_a, E_b>_G for logit changes E (K, N, C): G = cross-entropy curvature, mean over examples."""
    PE = (E * prob).sum(-1)
    return (torch.einsum("anc,bnc->ab", E, E * prob) - PE @ PE.T) / E.shape[1]


def state_numbers(p, x, y, gen):
    hs, masks, f = forward(p, x)
    prob = torch.softmax(f, 1)
    W = [p[f"linears.{k}.weight"] for k in range(5)]
    U, S, Vh = zip(*(torch.linalg.svd(w, full_matrices=False) for w in W))
    out = dict(acc=float((f.argmax(1) == y).float().mean()), sig=np.array([s[:6].numpy() for s in S]))
    E, dn2, mfrac = [], [], []
    for i in range(5):
        u, v, s1 = U[i][:, 0], Vh[i][0], S[i][0]
        sr = (W[i] ** 2).sum() / s1 ** 2
        d = 2 / s1 ** 2 * (W[i] - sr * s1 * torch.outer(u, v))          # d(stable rank)/dW
        m = (u @ d @ v) * torch.outer(u, v)
        E += [prop(hs[i] @ m.T, i, p, masks), prop(hs[i] @ (d - m).T, i, p, masks)]
        dn2.append(float((d ** 2).sum())); mfrac.append(float((m ** 2).sum() / (d ** 2).sum()))
    E = torch.stack(E)                                                   # (10, N, C): m0 r0 m1 r1 ...
    out.update(gram=gram(E, prob).numpy(), dn2=np.array(dn2), mfrac=np.array(mfrac))
    Efull = E[0::2] + E[1::2]
    line, top5, relu, rnd, rnd_relu = (np.zeros(4) for _ in range(5))
    path, path_check = np.zeros((4, 4)), np.zeros(4)
    for i in range(4):
        u, v, vn = U[i][:, 0], Vh[i][0], Vh[i + 1][0]
        g = torch.randn(256, generator=gen); g /= g.norm()
        line[i], rnd[i] = abs(float(u @ vn)), abs(float(u @ g))
        top5[i] = float(((U[i][:, :5].T @ Vh[i + 1][:5].T) ** 2).sum() / 5)
        Y = (hs[i] @ v)[:, None] * (masks[i] * u)                         # main-pattern push after the ReLU
        relu[i] = float(((Y @ vn) ** 2).sum() / (Y ** 2).sum()) ** 0.5
        rnd_relu[i] = float(((Y @ g) ** 2).sum() / (Y ** 2).sum()) ** 0.5
        # the full push of layer i, split by whether it enters layer i+1 along vn
        sr = (W[i] ** 2).sum() / S[i][0] ** 2
        d = 2 / S[i][0] ** 2 * (W[i] - sr * S[i][0] * torch.outer(u, v))
        da = (hs[i] @ d.T) * masks[i]
        through = prop(((da @ vn)[:, None] * vn) @ W[i + 1].T, i + 1, p, masks)
        path_check[i] = float((prop(da @ W[i + 1].T, i + 1, p, masks) - Efull[i]).norm() / Efull[i].norm())
        G = gram(torch.stack([through, Efull[i], Efull[i + 1]]), prob).numpy()
        path[i] = (G[0, 2], G[1, 2], G[0, 0], G[1, 1])                   # <thr,e_j>, <e_i,e_j>, |thr|^2, |e_i|^2
    out.update(line=line, top5=top5, relu=relu, rnd=rnd, rnd_relu=rnd_relu, path=path, path_check=path_check,
               _vec=(torch.stack([U[i][:, 0] for i in range(4)]), torch.stack([Vh[i + 1][0] for i in range(4)])))
    return out


def compute(x, y):
    gen = torch.Generator().manual_seed(SEED)
    onset = sgld_onsets()
    rows, recs, t0 = [], [], time.time()
    for c in CHUNKS:
        ck = torch.load(CK / f"aligned_burnin_M128_chunk{c}_ckpts.pt", map_location="cpu", weights_only=False)["checkpoints"]
        for step in sorted(ck):
            if step == 0:
                continue
            new = []
            for mbr in range(128):
                p = {k: t[mbr].float() for k, t in ck[step].items()}
                new.append(state_numbers(p, x, y, gen))
                rows.append((c, mbr, step, step - onset[(c, mbr)]))
            for mbr, r in enumerate(new):                                   # cross-network control
                vB = new[(mbr + 1) % 128]["_vec"][1]
                r["cross"] = np.abs((r["_vec"][0] * vB).sum(1).numpy())
            for r in new:
                del r["_vec"]
            recs += new
            print(f"  chunk {c} step {step}: {len(rows)} states, {time.time() - t0:.0f} s", flush=True)
    z = {k: np.stack([r[k] for r in recs]) for k in recs[0]}
    rows = np.array(rows, dtype=np.int64)
    np.savez_compressed(D / "lineup.npz", chunk=rows[:, 0], member=rows[:, 1], step=rows[:, 2], age=rows[:, 3], **z)


def main() -> int:
    torch.set_grad_enabled(False)
    data = make_modular_addition(build_config(DataConfig, load_config(ROOT / "configs" / "default.json").get("data")))
    x, y = data["x_train"].float(), data["y_train"]
    if not (D / "lineup.npz").exists():
        compute(x, y)
    print(f"{D / 'lineup.npz'} is in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
