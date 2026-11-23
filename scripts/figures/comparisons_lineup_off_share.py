"""Off-ratio f_i of the strongest output patterns P of W_i (top 5, or --patterns k) and the line-up |u_1(W_i).v_1(W_i+1)| at the saved
states: f_i = sum_x |P^T((1 - m_i) * z_i)|^2 / sum_x |P^T z_i|^2, with m_i = 1 on the neurons that are on.
SGLD from the saved burn-in states from step 60,000 (training accuracy >= 0.5 enters later), AdamW from the states saved every 10,000 steps.
Writes data/comparisons/lineup_off_share_{adamw,sgld}[_top<k>].npz: rows (chunk, member, step), val (state, hidden layer, [f_i, line-up]), acc.
Usage: python scripts/figures/comparisons_lineup_off_share.py --set {adamw,sgld} --patterns 1
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from src.data import DataConfig, make_modular_addition  # noqa: E402
from src.utils import build_config, data_dir, load_config  # noqa: E402

CK = Path(os.environ.get("GROKKING_CKPT", ROOT / "checkpoints"))      # folder of the saved states
NPAT = 5        # number of strongest output patterns of W_i in P (--patterns; the paper's table uses 1)


def tag():
    return "" if NPAT == 5 else f"_top{NPAT}"


def state(p, x, y):
    """Per hidden layer (4, 2): off-ratio and line-up; and the training accuracy."""
    h, zs = x, []
    for j in range(5):
        z = h @ p[f"linears.{j}.weight"].T + p[f"linears.{j}.bias"]
        if j < 4:
            h = torch.relu(z); zs.append(z)
    acc, out = float((z.argmax(1) == y).double().mean()), np.zeros((4, 2))
    for j in range(4):
        W, Wn = p[f"linears.{j}.weight"], p[f"linears.{j + 1}.weight"]
        U = torch.linalg.svd(W, full_matrices=False)[0]
        P = U[:, :NPAT]
        z = zs[j]; on = (z > 0).double()
        v1 = torch.linalg.svd(Wn, full_matrices=False)[2][0]
        out[j] = (float(((((1 - on) * z) @ P) ** 2).sum() / ((z @ P) ** 2).sum()), float(abs(U[:, 0] @ v1)))
    return out, acc


def compute(which, x, y):
    rows, V, A = [], [], []
    if which == "sgld":
        for c in range(4):
            ck = torch.load(CK / f"aligned_burnin_M128_chunk{c}_ckpts.pt", map_location="cpu", weights_only=False)["checkpoints"]
            for step in sorted(ck):
                if step < 60000:
                    continue
                for m in range(128):
                    v, a = state({k: t[m].double() for k, t in ck[step].items()}, x, y)
                    rows.append((c, m, step)); V.append(v); A.append(a)
                print(f"SGLD chunk {c} step {step}: {len(rows)} states", flush=True)
            del ck
    else:
        ck = CK
        for c in range(4):
            for step in range(10000, 150001, 10000):
                Pm = torch.load(ck / f"phase6b_adamw_M128_chunk{c}_step{step:06d}.pt", map_location="cpu", weights_only=False)["state"]["params"]
                for m in range(128):
                    v, a = state({k: t[m].double() for k, t in Pm.items()}, x, y)
                    rows.append((c, m, step)); V.append(v); A.append(a)
                print(f"AdamW chunk {c} step {step}: {len(rows)} states", flush=True)
    np.savez_compressed(data_dir("comparisons") / f"lineup_off_share_{which}{tag()}.npz", rows=np.array(rows), val=np.array(V), acc=np.array(A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("adamw", "sgld"), required=True)
    ap.add_argument("--patterns", type=int, default=5, help="number of strongest output patterns of W_i (paper table: 1)")
    args = ap.parse_args()
    global NPAT
    NPAT = args.patterns
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())))
    data = make_modular_addition(build_config(DataConfig, load_config(ROOT / "configs" / "default.json").get("data")))
    compute(args.set, data["x_train"].double(), data["y_train"])


if __name__ == "__main__":
    main()
