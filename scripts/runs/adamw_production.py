"""Trains the AdamW networks: 5 layers x 256, orthogonal start, task from the data block of configs/default.json.
Full-batch AdamW: lr 1e-3, betas 0.9/0.999, eps 1e-8, weight decay 1.0, loss from float64 scores; seed of a network = members * chunk + member.
Before training it checks that a resumed run is bit-identical and that the fast float64 stable rank matches a full SVD.
Writes data/adamw_fullbatch_512/adamw_M<M>_chunk<c>.npz; states every --segment steps to $GROKKING_CKPT (default checkpoints/).
Usage: python scripts/runs/adamw_production.py --chunk 0 --members 128 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.batched_train import (adamw_ensemble_trajectory, batched_observables, fast_observables,  # noqa: E402
                               stacked_init)
from src.data import DataConfig, make_modular_addition  # noqa: E402
from src.model import ModelConfig  # noqa: E402
from src.train import AdamWConfig  # noqa: E402
from src.utils import build_config, data_dir, load_config, save_config  # noqa: E402

N_LAYERS, WIDTH = 5, 256
LR, WD = 1e-3, 1.0
SETTINGS_FOR_RESUME = ("phase", "optimizer", "adamw_lr", "adamw_weight_decay", "betas", "eps", "model",
                       "ortho_init", "data", "init_seeds", "log_every", "segment", "loss_float64",
                       "readout_float64", "readout_fast", "device_name", "torch", "torch_threads", "batch_size")


def train(model_cfg, data, seeds, device, steps, log_every, start_state=None, start_step=0, snapshot_every=None):
    cfg = AdamWConfig(steps=steps, lr=LR, weight_decay=WD, log_every=log_every, seed=0)
    return adamw_ensemble_trajectory(model_cfg, data, cfg, seeds, device=device, ortho_init=True,
                                     snapshot_every=snapshot_every, start_state=start_state, start_step=start_step,
                                     loss_float64=True, readout_float64=True, readout_fast=True)


def hard_stops(model_cfg, data, seeds, device) -> dict:
    a = train(model_cfg, data, seeds, device, 200, 100, snapshot_every=200)["snapshots"][200]
    half = train(model_cfg, data, seeds, device, 100, 100, snapshot_every=100)["snapshots"][100]
    b = train(model_cfg, data, seeds, device, 200, 100, start_state=half, start_step=100,
              snapshot_every=100)["snapshots"][200]
    resume_ok = all(torch.equal(a[n][k], b[n][k]) for n in a for k in a[n])
    p0 = stacked_init(model_cfg, data, seeds, device, ortho_init=True)
    fast = fast_observables(p0, N_LAYERS, float64=True)
    full = batched_observables({k: v.cpu() for k, v in p0.items()}, N_LAYERS, float64=True)
    readout_diff = max(float(np.max(np.abs(fast[k] - full[k]) / np.abs(full[k]))) for k in full
                       if k.startswith("stable_rank"))
    print(f"hard stop 1, resume 100 + 100 = 200 straight, bit for bit: {'PASSED' if resume_ok else 'FAILED'}", flush=True)
    print(f"hard stop 2, fast float64 readout vs exact float64 SVD (CPU) at the start: largest relative difference "
          f"{readout_diff:.1e} ({'PASSED' if readout_diff < 1e-4 else 'FAILED'})", flush=True)
    if not resume_ok or readout_diff >= 1e-4:
        raise SystemExit("hard stop failed; not training")
    return dict(resume_check="passed", readout_check_max_rel_diff=readout_diff)


def save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--members", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--segment", type=int, default=10_000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--require-gpu", default=None, help="stop unless the card name is exactly this")
    args = ap.parse_args()
    phase = "6b"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; refusing to fall back to the CPU")
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
    if args.require_gpu is not None and gpu != args.require_gpu:
        raise SystemExit(f"card is {gpu!r}, not {args.require_gpu!r}; one card model per dataset")
    assert args.steps % args.segment == 0, "steps must be a multiple of segment"

    data_cfg = build_config(DataConfig, load_config(ROOT / "configs" / "default.json").get("data"))
    data = make_modular_addition(data_cfg)
    model_cfg = ModelConfig(n_layers=N_LAYERS, width=WIDTH, bias=True)
    seeds = [args.members * args.chunk + m for m in range(args.members)]
    tag = f"adamw_M{args.members}_chunk{args.chunk}"
    out_dir = data_dir("adamw_fullbatch_512")
    ck_dir = Path(os.environ.get("GROKKING_CKPT", ROOT / "checkpoints"))
    ck_dir.mkdir(parents=True, exist_ok=True)
    final, part = out_dir / f"{tag}.npz", out_dir / f"{tag}.partial.npz"
    if final.exists():
        print(f"{final} exists; nothing to do", flush=True)
        return 0
    run = dict(phase=phase, optimizer="AdamW", batch_size=None, adamw_lr=LR, adamw_weight_decay=WD, betas=[0.9, 0.999], eps=1e-8,
               model=vars(model_cfg), ortho_init=True, data=vars(data_cfg), chunk=args.chunk, members=args.members,
               init_seeds=seeds, steps=args.steps, log_every=args.log_every, segment=args.segment,
               loss_float64=True, readout_float64=True, readout_fast=True, device=args.device, device_name=gpu,
               torch=torch.__version__, torch_threads=torch.get_num_threads(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    print(f"chunk {args.chunk}: {args.members} networks (seeds {seeds[0]}-{seeds[-1]}), {args.steps} steps on {gpu}, "
          f"torch threads {torch.get_num_threads()}", flush=True)

    logs, t0, state = {}, 0, None
    if part.exists():                                       # continue a killed run
        z = dict(np.load(part))
        old = json.loads(str(z["_config"]))
        diff = [k for k in SETTINGS_FOR_RESUME if old.get(k) != run.get(k)]
        if diff:
            raise SystemExit(f"partial run has different settings {diff}; not continuing")
        t0 = int(z["reached"])
        state = torch.load(ck_dir / f"phase{phase}_{tag}_step{t0:06d}.pt", map_location="cpu", weights_only=False)["state"]
        logs = {k[4:]: v for k, v in z.items() if k.startswith("log_")}
        run["resumed_from_step"] = t0
        run.update({k: old[k] for k in ("resume_check", "readout_check_max_rel_diff") if k in old})
        print(f"continuing after step {t0}", flush=True)
    else:
        run.update(hard_stops(model_cfg, data, seeds, args.device))

    t_start = time.time()
    for t1 in range(t0 + args.segment, args.steps + 1, args.segment):
        res = train(model_cfg, data, seeds, args.device, t1, args.log_every, start_state=state, start_step=t0,
                    snapshot_every=args.segment)
        if t0 == 0:
            torch.save({"state": res["snapshots"][0], "step": 0, "config": run}, ck_dir / f"phase{phase}_{tag}_step000000.pt")
        state = res["snapshots"][t1]
        torch.save({"state": state, "step": t1, "config": run}, ck_dir / f"phase{phase}_{tag}_step{t1:06d}.pt")
        for k, v in res["log"].items():
            logs[k] = v if k not in logs else np.concatenate([logs[k], v], axis=-1)
        save_npz(part, _config=np.array(json.dumps(run)), reached=np.array(t1), init_seeds=np.array(seeds),
                 **{f"log_{k}": v for k, v in logs.items()})
        done = (t1 - (run.get("resumed_from_step") or 0)) / (args.steps - (run.get("resumed_from_step") or 0))
        el = time.time() - t_start
        print(f"  step {t1}/{args.steps}: {el / 3600:.2f} h, about {el / done * (1 - done) / 3600:.2f} h left; "
              f"test accuracy median {np.median(res['log']['test_acc'][:, -1]):.3f}, "
              f"networks above 0.9: {int((res['log']['test_acc'][:, -1] >= 0.9).sum())}/{args.members}", flush=True)
        t0 = t1

    run["seconds"] = round(time.time() - t_start)
    save_config(run, out_dir / f"{tag}_config.json")
    save_npz(final, _config=np.array(json.dumps(run)), init_seeds=np.array(seeds),
             **{f"log_{k}": v for k, v in logs.items()})
    part.unlink()
    print(f"wrote {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
