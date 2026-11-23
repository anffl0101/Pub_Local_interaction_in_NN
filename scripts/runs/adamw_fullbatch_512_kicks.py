"""Push ("kick") one layer of a copy of the 512 full-batch AdamW networks; record how much reaches the other layers.
Re-runs the training from saved states ($GROKKING_CKPT/phase6b_adamw_M<M>_chunk<c>_step<t>.pt); this keeps the
CPU thread count out of the numbers. Copy and main path: stable ranks, loss and accuracy at delays 1 to 5000 steps.
Hard stops (bit for bit, same GPU model only): main path = every saved state; a copy without push = main path.
Writes data/adamw_fullbatch_512/kicks/kick_M<M>_chunk<c>_L<layer>_h<h>.npz, one file per strength (resumable).
Usage: python scripts/runs/adamw_fullbatch_512_kicks.py --chunk C [--layer 1-5] [--h 0.1,0.03] [--gate]
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.batched_train import (_adamw_update, _make_grad_fn, _stable_rank_grad, batched_loss_acc,  # noqa: E402
                               data_to_device, fast_stable_ranks, make_template)
from src.data import DataConfig, make_modular_addition  # noqa: E402
from src.model import ModelConfig  # noqa: E402
from src.utils import build_config, data_dir, disable_tf32, load_config  # noqa: E402

LR, WD, N_PUSH, N_LAYERS = 1e-3, 1.0, 32, 5
DELAYS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000)
KICKS = tuple(range(2000, 30001, 1000)) + tuple(range(35000, 145001, 5000))
SAVED_EVERY, TWIN_STEPS = 10_000, 1_000
SETTINGS = ("kicks", "layer", "h", "window", "delays", "n_push", "members", "chunk", "lr", "weight_decay")


def load_state(ck: Path, members: int, chunk: int, t: int):
    f = ck / f"phase6b_adamw_M{members}_chunk{chunk}_step{t:06d}.pt"
    return torch.load(f, map_location="cpu", weights_only=False) if f.exists() else None


def to_dev(state, device):
    return tuple({k: x.to(device).clone() for k, x in state[n].items()} for n in ("params", "m", "v"))


def same(P, Mm, V, saved) -> bool:
    return all(torch.equal(x[k].cpu(), saved[n][k]) for n, x in (("params", P), ("m", Mm), ("v", V)) for k in x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--layer", type=int, choices=(1, 2, 3, 4, 5), default=1)
    ap.add_argument("--h", default="0.1", help="push strength(s), comma-separated: one retraining, one copy per strength "
                    "at each kick time, one output file per strength")
    ap.add_argument("--members", type=int, default=128)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--kicks", default=",".join(map(str, KICKS)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate", action="store_true", help="only run the two checks below, write nothing")
    ap.add_argument("--gate-until", type=int, default=SAVED_EVERY, help="--gate: run the main path to this step")
    ap.add_argument("--saved-every", type=int, default=SAVED_EVERY, help="spacing of the saved states (short trial runs only)")
    ap.add_argument("--out-dir", default=None, help="output folder (short trial runs only; default data/adamw_fullbatch_512/kicks)")
    args = ap.parse_args()
    every = args.saved_every
    disable_tf32()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; refusing to fall back to the CPU")
    ck = Path(os.environ.get("GROKKING_CKPT", ROOT / "checkpoints"))
    first = load_state(ck, args.members, args.chunk, 0)
    if first is None:
        raise SystemExit(f"no saved step-0 state in {ck}")
    src = first["config"]
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
    if gpu != src["device_name"]:
        raise SystemExit(f"card is {gpu!r}; the networks were made on {src['device_name']!r} (one card model per dataset)")
    assert (src["adamw_lr"], src["adamw_weight_decay"], src["loss_float64"], src.get("batch_size")) == (LR, WD, True, None)

    data = data_to_device(make_modular_addition(build_config(DataConfig, load_config(ROOT / "configs" / "default.json").get("data"))),
                          torch.device(args.device))
    xt, yt, xv, yv = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    T = make_template(ModelConfig(n_layers=N_LAYERS, width=256, bias=True), data, args.device)
    gf = _make_grad_fn(T, loss_float64=True)
    key = f"linears.{args.layer - 1}.weight"
    dl = [d for d in DELAYS if d <= args.window]

    def step(P, Mm, V, t):
        g = gf(P, xt, yt)
        with torch.no_grad():
            _adamw_update(P, g, Mm, V, t, LR, WD)

    def readout(P):
        a, b = batched_loss_acc(T, P, xt, yt), batched_loss_acc(T, P, xv, yv)
        return fast_stable_ranks(P, N_LAYERS, float64=True), np.stack([x.cpu().numpy() for x in (*a, *b)])

    def copy_run(P0, M0, V0, t1, h, n, record):
        P, Mm, V = ({k: x.clone() for k, x in d.items()} for d in (P0, M0, V0))
        sr = np.full((len(dl), args.members, N_LAYERS), np.nan)
        la = np.full((len(dl), 4, args.members), np.nan)
        bad = 0
        for k in range(n):
            step(P, Mm, V, t1 + k)
            if h and k < N_PUSH:                      # h/32 x |W| along the steepest stable-rank rise; weights only
                W = P[key]
                d = _stable_rank_grad(W.double())
                nrm = d.flatten(1).norm(dim=1)[:, None, None]
                ok = torch.isfinite(nrm) & (nrm > 0)
                bad += int((~ok).sum())
                d = torch.where(ok, d / nrm, torch.zeros_like(d))
                with torch.no_grad():
                    W.add_(((h / N_PUSH) * W.double().flatten(1).norm(dim=1)[:, None, None] * d).to(W.dtype))
            if record and k + 1 in dl:
                j = dl.index(k + 1)
                sr[j], la[j] = readout(P)
        return (P, Mm, V), sr, la, bad

    if args.gate:                                     # hard stops only, no output
        P, Mm, V = to_dev(first["state"], args.device)
        t_twin, twin, t0 = 2000, None, time.time()
        for t in range(args.gate_until + 1):
            if t == t_twin:
                twin = copy_run(P, Mm, V, t, 0.0, TWIN_STEPS, False)[0]
            if t == t_twin + TWIN_STEPS:
                print(f"check: copy without push = main path after {TWIN_STEPS} steps, bit for bit): "
                      f"{'PASSED' if all(torch.equal(a[k], b[k]) for a, b in zip(twin, (P, Mm, V)) for k in a) else 'FAILED'}", flush=True)
            if t and t % every == 0:
                s = load_state(ck, args.members, args.chunk, t)
                print(f"check at step {t} (weights and AdamW memory = saved state, bit for bit): "
                      f"{'PASSED' if same(P, Mm, V, s['state']) else 'FAILED'}  ({time.time() - t0:.0f} s)", flush=True)
            if t < args.gate_until:
                step(P, Mm, V, t)
        return 0

    kicks = sorted(int(k) for k in args.kicks.split(","))
    hs = [float(x) for x in args.h.split(",")]
    if len(set(hs)) != len(hs) or any(h <= 0 for h in hs):
        raise SystemExit(f"--h must list distinct positive strengths, got {args.h}")
    t_end = kicks[-1] + args.window
    out_dir = Path(args.out_dir) if args.out_dir else data_dir("adamw_fullbatch_512", "kicks")
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    def cfg_for(h):
        return dict(kicks=kicks, layer=args.layer, h=h, window=args.window, delays=dl, n_push=N_PUSH,
                    members=args.members, chunk=args.chunk, lr=LR, weight_decay=WD, loss_float64=True,
                    push="direct on the weights", run_with_strengths=hs,
                    readout="fast_stable_ranks float64", source=f"phase6b_adamw_M{args.members}_chunk{args.chunk}_step*.pt",
                    init_seeds=src["init_seeds"], device_name=gpu, torch=torch.__version__, torch_threads=torch.get_num_threads(),
                    commit=commit, slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))

    def paths(h):
        tag = f"kick_M{args.members}_chunk{args.chunk}_L{args.layer}_h{h:g}"
        return out_dir / f"{tag}.npz", out_dir / f"{tag}.partial.npz"

    hs = [h for h in hs if not paths(h)[0].exists()]          # strengths already finished are skipped
    if not hs:
        print("all requested files exist; nothing to do", flush=True)
        return 0
    Q, D, M, H = len(kicks), len(dl), args.members, len(hs)
    SRm, LAm = np.full((Q, D, M, N_LAYERS), np.nan), np.full((Q, D, 4, M), np.nan)
    SRk, LAk = np.full((H, Q, D, M, N_LAYERS), np.nan), np.full((H, Q, D, 4, M), np.nan)
    BAD = np.zeros((H, Q), dtype=int)
    for a, h in enumerate(hs):                                # resume: every strength from its own partial file
        part = paths(h)[1]
        if part.exists():
            z = np.load(part)
            old, cur = json.loads(str(z["_config"])), cfg_for(h)
            if any(old[k] != cur[k] for k in SETTINGS):
                raise SystemExit(f"{part} holds a run with other settings; delete it deliberately")
            SRk[a], LAk[a], BAD[a] = z["sr_kick"], z["la_kick"], z["push_skipped"]
            for X, k in ((SRm, "sr_main"), (LAm, "la_main")):  # the main path is the same in every file
                np.copyto(X, z[k], where=np.isnan(X) & ~np.isnan(z[k]))
    done_main = ~np.isnan(SRm).any(axis=(1, 2, 3))
    done_kick = ~np.isnan(SRk).any(axis=(2, 3, 4))            # (H, Q)
    todo = [q for q in range(Q) if not (done_main[q] and done_kick[:, q].all())]
    start = None if not todo else (kicks[todo[0]] // every) * every
    need = {}
    for q in todo:
        for j, d in enumerate(dl):
            need.setdefault(kicks[q] + d, []).append((q, j))

    def save(path, a, h):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, _config=np.array(json.dumps(cfg_for(h))), kicks=np.array(kicks), delays=np.array(dl),
                                sr_main=SRm, sr_kick=SRk[a], la_main=LAm, la_kick=LAk[a], push_skipped=BAD[a],
                                la_names=np.array(["train_loss", "train_acc", "test_loss", "test_acc"]))
        os.replace(tmp, path)

    print(f"chunk {args.chunk}, layer L{args.layer}, h {', '.join(f'{h:g}' for h in hs)}: {Q} kicks ({len(todo)} to do) "
          f"on {gpu}; main path from step {start}", flush=True)
    if start is not None:
        s = load_state(ck, M, args.chunk, start)
        P, Mm, V = to_dev(s["state"], args.device)
        twin_at, twin, twin_done, t0 = None, None, False, time.time()
        for t in range(start, t_end + 1):
            if t % every == 0 and t > start:
                s = load_state(ck, M, args.chunk, t)
                if s is not None and not same(P, Mm, V, s["state"]):
                    raise SystemExit(f"CHECK FAILED at step {t}: the re-run is not the recorded training")
                if s is not None:
                    print(f"  identity at step {t}: passed", flush=True)
            if twin_at is not None and t == twin_at:
                if not all(torch.equal(a[k], b[k]) for a, b in zip(twin, (P, Mm, V)) for k in a):
                    raise SystemExit("CHECK FAILED: a copy without push differs from the untouched run")
                print(f"  same-noise check at step {t}: passed", flush=True)
                twin_at, twin, twin_done = None, None, True
            for q, j in need.get(t, ()):
                SRm[q, j], LAm[q, j] = readout(P)
            if t in kicks and not done_kick[:, kicks.index(t)].all():
                q = kicks.index(t)
                if not twin_done and twin is None and t + TWIN_STEPS <= t_end:
                    twin, twin_at = copy_run(P, Mm, V, t, 0.0, TWIN_STEPS, False)[0], t + TWIN_STEPS
                for a, h in enumerate(hs):
                    if done_kick[a, q]:
                        continue
                    _, SRk[a, q], LAk[a, q], BAD[a, q] = copy_run(P, Mm, V, t, h, args.window, True)
                    done_kick[a, q] = True
                    save(paths(h)[1], a, h)
                print(f"  kick {q + 1}/{Q} at step {t} done ({(time.time() - t0) / 3600:.2f} h)", flush=True)
            if t < t_end:
                step(P, Mm, V, t)
    if np.isnan(SRm).any() or np.isnan(SRk).any():
        raise SystemExit("some readouts are missing")
    for a, h in enumerate(hs):
        final, part = paths(h)
        save(final, a, h)
        part.unlink(missing_ok=True)
        print(f"wrote {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
