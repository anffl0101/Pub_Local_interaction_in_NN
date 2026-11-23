"""Line-up of neighbouring layers in the 512 full-batch AdamW networks, at each state saved every 10,000 steps.
Per network: |u_1(W_i) . v_1(W_i+1)| for i = 1..4 (u_1 strongest output pattern of W_i, v_1 strongest input
pattern of W_i+1), from a float64 copy of the weights. Reads $GROKKING_CKPT/phase6b_adamw_M128_chunk<c>_step<t>.pt.
Writes data/adamw_fullbatch_512/lineup_states.npz (chunk, member, step, line[n, 4], plus stable ranks and checks).
Usage: GROKKING_CKPT=<folder of saved states> python scripts/runs/adamw_fullbatch_512_lineup.py
"""
import os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CK = Path(os.environ.get("GROKKING_CKPT", ROOT / "checkpoints"))
rows, lines, extra = [], [], []
for c in range(4):
    for t in range(0, 150001, 10000):
        f = CK / f"phase6b_adamw_M128_chunk{c}_step{t:06d}.pt"
        if not f.exists():
            print("missing", f, flush=True); continue
        P = torch.load(f, map_location="cpu", weights_only=False)["state"]["params"]
        W = [P[f"linears.{k}.weight"].double() for k in range(5)]
        sv = [torch.linalg.svd(w) for w in W]                          # (U, S, Vh) per layer, batched over members
        U = [sv[i][0][..., :, 0] for i in range(4)]                     # (M, out) strongest output pattern of W_1..W_4
        V = [sv[i][2][..., 0, :] for i in range(1, 5)]                  # (M, in)  strongest input pattern of W_2..W_5
        L = torch.stack([(U[i] * V[i]).sum(-1).abs() for i in range(4)], 1).numpy()
        S_ = [x[1] for x in sv]
        sr = torch.stack([(s ** 2).sum(-1) / s[..., 0] ** 2 for s in S_], 1).numpy()          # stable rank, to match the log
        gap = torch.stack([s[..., 0] / s[..., 1] for s in S_], 1).numpy()                     # sigma_1 / sigma_2 per layer
        top5 = torch.stack([((sv[i][0][..., :, :5].transpose(-1, -2) @ sv[i + 1][2][..., :5, :].transpose(-1, -2)) ** 2).sum((-1, -2)) / 5
                            for i in range(4)], 1).numpy()                                   # overlap of the 5 strongest patterns
        cosAB = []
        for i in range(4):
            Bm = W[i] @ W[i].transpose(-1, -2); Am = W[i + 1].transpose(-1, -2) @ W[i + 1]
            cosAB.append((Am * Bm).sum((-1, -2)) / (Am.norm(dim=(-1, -2)) * Bm.norm(dim=(-1, -2))))
        cosAB = torch.stack(cosAB, 1).numpy()                                                   # all patterns, weighted by strength
        for m in range(L.shape[0]):
            rows.append((c, m, t)); lines.append(L[m]); extra.append(np.concatenate([sr[m], gap[m], top5[m], cosAB[m]]))
        print(f"chunk {c} step {t}: median line-up {np.round(np.median(L, 0), 3)}", flush=True)
rows = np.array(rows)
out = ROOT / "data" / "adamw_fullbatch_512" / "lineup_states.npz"
np.savez_compressed(out, chunk=rows[:, 0], member=rows[:, 1], step=rows[:, 2], line=np.array(lines), stable_rank=np.array(extra)[:, :5], sigma_gap=np.array(extra)[:, 5:10],
                    top5=np.array(extra)[:, 10:14], cos_AB=np.array(extra)[:, 14:18])
print("wrote", out)
