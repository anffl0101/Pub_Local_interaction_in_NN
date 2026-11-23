"""Sharpness (top Hessian eigenvalue of the training loss) around SGLD collapses vs the limit 2/eta = 20 (Cohen et al. 2021).
Runs 5000 SGLD steps from step 100,000 of aligned_burnin_M128_chunk<c>_ckpts.pt (in checkpoints/ or $GROKKING_CKPT); logs every 20 steps.
On CUDA (A10, the training's GPU model, needed for bit-exact replay; all 128 members) the run is the original training:
loss and accuracy must equal data/sgld_512/burnin_M128_chunk<c>.npz bit for bit, or nothing is written. Collapse: training accuracy below 0.5.
Writes data/sgld_512/sharpness_chunk<c>.npz (CUDA) or data/tests/sgld_crash_sharpness/sharpness_chunk<c>.npz (CPU, new noise).
Usage: python scripts/runs/tests_sgld_crash_sharpness.py [--device cuda|cpu] [--chunk C] [--members M]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.batched_train import _make_grad_fn, _sgld_ensemble_step, batched_loss_acc, data_to_device, make_template  # noqa: E402
from src.data import DataConfig, make_modular_addition  # noqa: E402
from src.model import ModelConfig  # noqa: E402
from src.train import SGLDConfig  # noqa: E402
from src.utils import build_config, data_dir, disable_tf32, load_config, save_config  # noqa: E402

LOG_EVERY = 20
START, STEPS = 100_000, 5_000
CFG = SGLDConfig(eta=0.1, T=3e-5, weight_decay=1e-3, seed=42)
POWER_ITERS, POWER_ITERS_FIRST = 8, 30
OUT = data_dir("tests/sgld_crash_sharpness")


@torch.no_grad()
def readout(template, params, x, y):
    logits = vmap(lambda prm, xb: functional_call(template, prm, (xb,)), in_dims=(0, None))(params, x)
    idx = y.view(1, -1, 1).expand(logits.shape[0], -1, 1)
    loss = -F.log_softmax(logits, -1).gather(-1, idx).squeeze(-1).mean(1)
    acc = (logits.argmax(-1) == y.view(1, -1)).float().mean(1)
    return loss.cpu().numpy(), acc.cpu().numpy()


def sharpness(template, prm, x, y, v, iters):
    """Top Hessian eigenvalue of one network's training cross-entropy, by power iteration from v; returns (value, vector)."""
    keys = list(prm)
    leaves = [prm[k].detach().clone().requires_grad_(True) for k in keys]
    p = dict(zip(keys, leaves))
    loss = F.cross_entropy(functional_call(template, p, (x,)), y)
    grads = torch.autograd.grad(loss, leaves, create_graph=True)
    lam = float("nan")
    for _ in range(iters):
        gv = sum((g * vi).sum() for g, vi in zip(grads, v))
        hv = torch.autograd.grad(gv, leaves, retain_graph=True)
        hv = [h.detach() for h in hv]
        lam = float(sum((h * vi).sum() for h, vi in zip(hv, v)))
        norm = float(torch.sqrt(sum((h * h).sum() for h in hv)))
        v = [h / norm for h in hv]
    return lam, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--members", type=int, default=128)
    ap.add_argument("--require-gpu", default="NVIDIA A10", help="card of the 512 aligned networks")
    args = ap.parse_args()
    MEMBERS, chunk, dev = args.members, args.chunk, torch.device(args.device)
    main_setting = dev.type == "cuda"                    # the original training of the 512 networks, checked below
    if main_setting:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
        if name != args.require_gpu or MEMBERS != 128:      # equality: "NVIDIA A100" contains "NVIDIA A10"
            print(f"ERROR: the main setting needs {args.require_gpu} and all 128 members; got {name}, {MEMBERS}", file=sys.stderr)
            return 3
        ref = np.load(data_dir("sgld_512") / f"burnin_M128_chunk{chunk}.npz")
        ref_step = ref["log_step"]
    disable_tf32()
    torch.set_grad_enabled(True)
    data = data_to_device(make_modular_addition(build_config(DataConfig, load_config(ROOT / "configs" / "default.json").get("data"))), dev)
    template = make_template(ModelConfig(n_layers=5, width=256, bias=True), data, dev)
    xt, yt = data["x_train"], data["y_train"]
    source = f"aligned_burnin_M128_chunk{chunk}_ckpts.pt"
    ck = torch.load(Path(os.environ.get("GROKKING_CKPT", ROOT / "checkpoints")) / source, map_location="cpu", weights_only=False)["checkpoints"][START]
    params = {k: x[:MEMBERS].clone().float().to(dev) for k, x in ck.items()}
    del ck
    grad_fn = _make_grad_fn(template, loss_float64=False)
    gen = torch.Generator().manual_seed(0)
    vecs = [[torch.randn(p[m].shape, generator=gen).to(dev) for p in params.values()] for m in range(MEMBERS)]
    for m in range(MEMBERS):
        n = float(torch.sqrt(sum((v * v).sum() for v in vecs[m])))
        vecs[m] = [v / n for v in vecs[m]]
    steps, loss, acc, lam = [], [], [], []
    t0 = time.time()
    for t in range(START, START + STEPS):
        if t % LOG_EVERY == 0:
            lo, ac = readout(template, params, xt, yt)
            if main_setting:                             # hard stop: these must be the original networks
                bl, ba = (v.cpu().numpy() for v in batched_loss_acc(template, params, xt, yt))
                j = int(np.where(ref_step == t)[0][0])
                if not (np.array_equal(bl, ref["log_train_loss"][:, j]) and np.array_equal(ba, ref["log_train_acc"][:, j])):
                    print(f"IDENTITY CHECK FAILED at step {t}: max |loss diff| {np.abs(bl - ref['log_train_loss'][:, j]).max():.3e}, "
                          f"max |acc diff| {np.abs(ba - ref['log_train_acc'][:, j]).max():.3e}. Nothing written.", file=sys.stderr)
                    return 1
            row = []
            for m in range(MEMBERS):
                prm = {k: p[m] for k, p in params.items()}
                iters = POWER_ITERS_FIRST if not steps else POWER_ITERS
                lm, vecs[m] = sharpness(template, prm, xt, yt, vecs[m], iters)
                row.append(lm)
            steps.append(t); loss.append(lo); acc.append(ac); lam.append(row)
            if len(steps) % 25 == 1:
                print(f"step {t}: {time.time() - t0:.0f} s; acc {np.round(ac, 2).tolist()}; "
                      f"sharpness {np.round(row, 1).tolist()}", flush=True)
        _sgld_ensemble_step(params, grad_fn, xt, yt, CFG, t, chunk=chunk)
    steps, loss, acc, lam = np.array(steps), np.stack(loss, 1), np.stack(acc, 1), np.array(lam).T
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cfg = dict(sgld=vars(CFG), chunk=chunk, members=list(range(MEMBERS)), start_step=START, steps=STEPS,
               log_every=LOG_EVERY, power_iters=POWER_ITERS, power_iters_first=POWER_ITERS_FIRST,
               source=source, device=str(dev), gpu=torch.cuda.get_device_name(0) if main_setting else None,
               torch_threads=torch.get_num_threads(), stability_limit_2_over_eta=2 / CFG.eta, commit=commit,
               note=("the original training of the aligned networks: training loss and accuracy equal the burn-in log "
                     "bit for bit at every logged step" if main_setting else
                     "CPU noise stream, a new statistically equivalent stretch"))
    out = data_dir("sgld_512") if main_setting else OUT
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / f"sharpness_chunk{chunk}_config.json")
    np.savez_compressed(out / f"sharpness_chunk{chunk}.npz", _config=np.array(json.dumps(cfg)), step=steps,
                        train_loss=loss, train_acc=acc, sharpness=lam)
    if main_setting:
        print(f"IDENTITY CHECK PASSED at all {len(steps)} logged steps: these are the original networks of chunk {chunk}.")
    # summary: sharpness around collapses (first log point below 0.5 after a point at or above 0.5)
    print(f"\ndone in {time.time() - t0:.0f} s. Stability limit 2/eta = {2 / CFG.eta:g}.")
    print(f"sharpness over all logged points: median {np.nanmedian(lam):.1f}, 10-90% "
          f"{np.nanpercentile(lam, 10):.1f}-{np.nanpercentile(lam, 90):.1f}, max {np.nanmax(lam):.1f}")
    before, at, after, n = [], [], [], 0
    for m in range(MEMBERS):
        hits = np.where((acc[m, :-1] >= 0.5) & (acc[m, 1:] < 0.5))[0] + 1
        for k in hits:
            n += 1
            before.append(lam[m, max(0, k - 5):k].tolist()); at.append(lam[m, k]); after.append(lam[m, k + 1:k + 6].tolist())
    print(f"collapses: {n} in {MEMBERS} networks over {STEPS} steps")
    if n:
        b = np.array([np.nanmean(r) for r in before]); a = np.array([np.nanmean(r) for r in after])
        print(f"sharpness in the 100 steps before a collapse: median {np.nanmedian(b):.1f}; at the collapse point "
              f"{np.nanmedian(at):.1f}; in the 100 steps after: median {np.nanmedian(a):.1f}")
        quiet = lam[acc >= 0.99]
        print(f"sharpness at log points with training accuracy >= 0.99: median {np.nanmedian(quiet):.1f}, "
              f"share above 2/eta: {np.mean(quiet > 2 / CFG.eta):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
