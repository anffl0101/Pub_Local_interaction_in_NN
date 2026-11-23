"""One-step change of layer j's gradient after a push on layer i, split into T1 (through the changed error) and T2 (the rest).
Per saved state ($GROKKING_CKPT/phase6b_adamw_M128_chunk<c>_step<t>.pt or aligned_burnin_M128_chunk<c>_ckpts.pt) and pair (i, j), push along
the gradient of layer i's stable rank: |T1|^2, |T2|^2, <T1,T2>, |T1+T2|^2 per |push|^2; for AdamW also times its step factor. Float64, CPU.
Hard stops (AdamW): the stable-rank numbers must match data/adamw_fullbatch_512/saved_onestep_chunk<c>.npz, and
one real update with a tiny push must reproduce the responses to 2% of the row maximum. Writes data/<set>/onestep_terms_frobenius_chunk<c>.npz.
Usage: python scripts/runs/onestep_terms_frobenius.py --method {adamw,sgld} --chunk C
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vjp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from src.data import DataConfig, make_modular_addition  # noqa: E402
from src.model import MLP, ModelConfig  # noqa: E402
from src.utils import build_config, data_dir, load_config  # noqa: E402
from adamw_fullbatch_512_onestep import B1, B2, CHECK_LIMIT, EPS, LR, WK, check, sr_grad, state_numbers  # noqa: E402

ADAMW_STEPS = range(10000, 150001, 10000)
CHECK_AT = (10000, 20000, 80000, 150000)


def terms(p, T, x, y, A=None):
    """Returns |d_i|^2, the projections <d_j, total>, <d_j, T1> and the sizes |T1|^2, |T2|^2, <T1,T2>, |T1+T2|^2, per |d_i|^2.
    Shapes (5,), (variants, 2, 5, 5), (variants, 4, 5, 5); variants: plain, then times A."""
    loss = lambda q: F.cross_entropy(functional_call(T, q, (x,)), y)
    logits = lambda q: functional_call(T, q, (x,))
    d = [sr_grad(p[k]) for k in WK]
    f0, vjp_fn = vjp(logits, p)
    prob = torch.softmax(f0, 1).detach()
    nv = 1 if A is None else 2
    d2, proj, size = np.zeros(5), np.zeros((nv, 2, 5, 5)), np.zeros((nv, 4, 5, 5))
    for i in range(5):
        dv = {k: torch.zeros_like(t) for k, t in p.items()}
        dv[WK[i]] = d[i]
        _, Hv = jvp(grad(loss), (p,), (dv,))                       # total change of every gradient
        _, df = jvp(logits, (p,), (dv,))                            # change of the scores
        dp = prob * (df - (prob * df).sum(1, keepdim=True)) / len(y)
        (Jt,) = vjp_fn(dp)                                          # first term
        d2[i] = n2 = float((d[i] ** 2).sum())
        for j in range(5):
            for v in range(nv):
                f = 1.0 if v == 0 else A[WK[j]]
                tot, t1 = f * Hv[WK[j]], f * Jt[WK[j]]
                t2 = tot - t1
                proj[v, :, i, j] = [float((d[j] * tot).sum()) / n2, float((d[j] * t1).sum()) / n2]
                size[v, :, i, j] = [float((t1 ** 2).sum()) / n2, float((t2 ** 2).sum()) / n2,
                                    float((t1 * t2).sum()) / n2, float((tot ** 2).sum()) / n2]
    return d2, proj, size


def adamw_factor(p, m, v, t, T, x, y):
    """AdamW's step per unit change of the gradient, for each weight (as state_numbers)."""
    g = grad(lambda q: F.cross_entropy(functional_call(T, q, (x,)), y))(p)
    bc1, bc2 = 1 - B1 ** (t + 1), 1 - B2 ** (t + 1)
    A = {}
    for k in WK:
        mp, vp = B1 * m[k] + (1 - B1) * g[k], B2 * v[k] + (1 - B2) * g[k] ** 2
        den = vp.sqrt() / math.sqrt(bc2) + EPS
        A[k] = ((1 - B1) / bc1) / den - (mp / bc1) / den ** 2 * (1 - B2) * g[k] / (vp.sqrt().clamp_min(1e-30) * math.sqrt(bc2))
    return A


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=("adamw", "sgld"), required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="short trial run: only the first N members of the first saved step")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    ck = Path(os.environ.get("GROKKING_CKPT", ROOT / "checkpoints"))
    data = make_modular_addition(build_config(DataConfig, load_config(ROOT / "configs" / "default.json").get("data")))
    x, y = data["x_train"].double(), data["y_train"]
    T = MLP.for_dataset(ModelConfig(n_layers=5, width=256, bias=True), data).double()
    out = {k: [] for k in ("step", "member", "d2", "proj", "size")}
    checks, worst = {}, 0.0
    members = range(args.limit or 128)

    if args.method == "adamw":
        name = "adamw_fullbatch_512"
        old = np.load(data_dir(name) / f"saved_onestep_chunk{args.chunk}.npz")
        steps = list(ADAMW_STEPS)[:1] if args.limit else ADAMW_STEPS
        for t in steps:
            st = torch.load(ck / f"phase6b_adamw_M128_chunk{args.chunk}_step{t:06d}.pt", map_location="cpu", weights_only=False)["state"]
            for mbr in members:
                pick = lambda dct: {k: v[mbr].double() for k, v in dct.items()}
                p, m, v = pick(st["params"]), pick(st["m"]), pick(st["v"])
                d2, proj, size = terms(p, T, x, y, adamw_factor(p, m, v, t, T, x, y))
                ref = old["coupling"][(old["step"] == t) & (old["member"] == mbr)][0]          # (2,5,5), in %
                worst = max(worst, float(np.abs(-LR * 100 * proj[1] - ref).max() / np.abs(ref).max()))
                if worst > 1e-6:
                    raise SystemExit(f"stable-rank numbers differ from the saved ones: {worst:.2e} at step {t} member {mbr}")
                if mbr == 0 and t in CHECK_AT:
                    checks[t] = check(p, m, v, t, T, x, y, state_numbers(p, m, v, t, T, x, y)[1])
                    print(f"known-answer check step {t}: {checks[t]:.4f} of the row maximum", flush=True)
                    if checks[t] > CHECK_LIMIT:
                        raise SystemExit(f"known-answer check failed at step {t}: {checks[t]:.4f} > {CHECK_LIMIT}")
                for k, val in zip(out, (t, mbr, d2, proj, size)):
                    out[k].append(val)
            print(f"adamw chunk {args.chunk} step {t} done, largest difference from the saved numbers {worst:.1e}", flush=True)
    else:
        name = "sgld_512"
        cks = torch.load(ck / f"aligned_burnin_M128_chunk{args.chunk}_ckpts.pt", map_location="cpu", weights_only=False)["checkpoints"]
        steps = [s for s in sorted(cks) if s != 0]
        for t in steps[:1] if args.limit else steps:
            for mbr in members:
                p = {k: v[mbr].double() for k, v in cks[t].items()}
                d2, proj, size = terms(p, T, x, y)
                for k, val in zip(out, (t, mbr, d2, proj, size)):
                    out[k].append(val)
            print(f"sgld chunk {args.chunk} step {t} done", flush=True)

    if args.limit:
        print(f"test run only ({len(out['step'])} states), nothing written; size[0] of the first state:\n{out['size'][0][0]}")
        return 0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cfg = dict(method=args.method, chunk=args.chunk, steps=[int(s) for s in steps], dtype="float64", push="gradient of the stable rank, unit size",
               variants=["plain"] + (["adamw"] if args.method == "adamw" else []), proj=["<d_j,total>", "<d_j,T1>"],
               size=["|T1|^2", "|T2|^2", "<T1,T2>", "|T1+T2|^2"], largest_difference_from_saved=worst, checks=checks,
               commit=commit, torch=torch.__version__, threads=torch.get_num_threads())
    np.savez_compressed(data_dir(name) / f"onestep_terms_frobenius_chunk{args.chunk}.npz", _config=np.array(json.dumps(cfg)),
                        **{k: np.array(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
