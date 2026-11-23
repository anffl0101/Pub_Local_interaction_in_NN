"""SGLD runs of the shared-start 512 networks, one chunk per call: burn-in (training) or kick (pushes).
Writes burnin_M<M>_chunk<c>.npz or kick_M<M>_chunk<c>_L<i>*.npz to data/sgld_512/; burn-in states go to $GROKKING_CKPT.
Each chunk index gives its members different noise, so every call needs its own --chunk.
Usage: python scripts/runs/sgld_512_runs.py --mode kick --device cuda --members 128 --chunk 0 --kick-layer 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sgld_production import

import torch  # noqa: E402

from src.data import DataConfig, make_modular_addition  # noqa: E402
from src.model import ModelConfig  # noqa: E402
from src.batched_train import (  # noqa: E402
    SIGMA_FALLBACKS,
    reset_sigma_fallbacks,
    sgld_ensemble_trajectory,
)
from src.utils import (  # noqa: E402
    build_config,
    load_config,
    data_dir,
    save_config,
    seed_everything,
)

# SGLD settings and the folder of saved states
from sgld_production import SGLDRecipe, ckpt_dir, _sgld_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "default.json"

# Kick mode: push one layer for KICK_LEN steps at each kick step, record every layer up to KICK_WINDOW later.
KICK_STEPS = tuple(range(10_000, 60_001, 5_000)) + (100_000, 150_000)   # 13 kick steps
KICK_LEN = 32                                      # the push acts for 32 steps, then stops
KICK_WINDOW = 5_000                                # length of each pushed copy (steps)
KICK_FIELDS = (1e-4, 1e-3, 1e-2)                   # strengths, x10 apart
KICK_TAUS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000)


@dataclass
class RunSettings:
    horizon: int = 200_000      # burn-in trajectory length (steps)
    ckpt_every: int = 20_000    # saved states, from which the pushed copies are re-run


def run_burnin(rec: SGLDRecipe, arec: RunSettings, members: int, chunk: int,
               device: str, data, model_cfg, out_dir: Path) -> str:
    """Burn-in to the horizon: logs every log_every steps plus checkpoints for replay."""
    tag = f"burnin_M{members}_chunk{chunk}"
    out_path = out_dir / f"{tag}.npz"
    cfg_json = json.dumps({"mode": "burnin", **asdict(rec), **asdict(arec),
                           "members": members, "chunk": chunk}, sort_keys=True)
    if out_path.exists():
        with np.load(out_path) as f:
            if str(f["_config"].item()) == cfg_json:
                return f"{tag}: already complete, skipped"
        raise RuntimeError(f"{out_path} holds a DIFFERENT run; delete deliberately.")
    ckpt_steps = tuple(range(0, arec.horizon, arec.ckpt_every))
    t0 = time.perf_counter()
    traj = sgld_ensemble_trajectory(
        model_cfg, data, _sgld_cfg(rec), n_steps=arec.horizon,
        log_every=rec.log_every, init_seeds=[rec.init_seed] * members,
        checkpoint_steps=ckpt_steps, device=device, chunk=chunk,
        ortho_init=rec.ortho_init,
    )
    print(f"  [timing] burn-in {arec.horizon} steps: {time.perf_counter()-t0:.1f}s",
          flush=True)
    payload = {f"log_{k}": v for k, v in traj["log"].items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, _config=np.array(cfg_json), chunk=chunk,
                        n_members=members, **payload)
    ck = ckpt_dir() / f"aligned_{tag}_ckpts.pt"
    torch.save({"chunk": chunk, "n_members": members,
                "ckpt_steps": ckpt_steps, "checkpoints": traj["checkpoints"]}, ck)
    return f"{tag}: done ({arec.horizon} steps, {len(ckpt_steps)} checkpoints)"


def run_kick(rec: SGLDRecipe, arec: RunSettings, members: int, chunk: int,
             kick_layer: int, device: str, data, model_cfg, out_dir: Path,
             kick_steps=KICK_STEPS, window: int = KICK_WINDOW,
             kick_fields=KICK_FIELDS, dense_every: int = 0) -> str:
    """Kick run: push layer kick_layer (1-based) along its stable-rank gradient, record all layers of the copy."""
    from src.batched_train import sgld_ensemble_kick_trajectory
    KICK_FIELDS_ = tuple(kick_fields)
    # non-default strengths add _h<list>; dense_every N (used by the crash filter) adds _d<N>
    suffix ="" if KICK_FIELDS_ == tuple(KICK_FIELDS) else "_h" + "-".join(f"{h:g}" for h in KICK_FIELDS_)
    suffix += f"_d{dense_every}" if dense_every else ""
    tag = f"kick_M{members}_chunk{chunk}_L{kick_layer}{suffix}"
    out_path = out_dir / f"{tag}.npz"
    cfg_json = json.dumps({"mode": "kick", **asdict(rec), **asdict(arec),
                           "members": members, "chunk": chunk,
                           "kick_layer": kick_layer, "kick_steps": list(kick_steps),
                           "kick_len": KICK_LEN, "window": window,
                           "kick_fields": KICK_FIELDS_, "kick_taus": KICK_TAUS,
                           **({"dense_every": dense_every} if dense_every else {})}, sort_keys=True)
    if out_path.exists():
        with np.load(out_path) as f:
            if str(f["_config"].item()) == cfg_json:
                return f"{tag}: already complete, skipped"
        raise RuntimeError(f"{out_path} holds a DIFFERENT run; delete deliberately.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"{tag}.partial.npz"

    def save(traj: dict, path: Path) -> None:
        k = traj["kick"]
        payload = {f"log_{n}": v for n, v in traj["log"].items()}
        payload.update({n: k[n] for n in ("dense_every", "sr_kick_dense", "la_kick_dense") if n in k})
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, _config=np.array(cfg_json), chunk=chunk,
                            n_members=members, steps_run=traj["steps_run"],
                            kick_layer=kick_layer, kick_len=KICK_LEN, window=window,
                            field_layer=kick_layer - 1,
                            kick_steps=k["steps"], kick_taus=k["taus"], kick_fields=k["fields"],
                            sr_main=k["sr_main"], sr_kick=k["sr_kick"],
                            la_main=k["la_main"], la_kick=k["la_kick"], **payload)
        tmp.replace(path)

    t0 = time.perf_counter()
    traj = sgld_ensemble_kick_trajectory(
        model_cfg, data, _sgld_cfg(rec), n_steps=arec.horizon,
        log_every=rec.log_every, init_seeds=[rec.init_seed] * members,
        kick_steps=list(kick_steps), kick_len=KICK_LEN, window=window,
        kick_fields=KICK_FIELDS_, kick_taus=KICK_TAUS,
        field_layer=kick_layer - 1, device=device, chunk=chunk,
        ortho_init=rec.ortho_init,
        on_progress=lambda tr: save(tr, partial_path),
        dense_every=dense_every,
    )
    k = traj["kick"]
    print(f"  [timing] kick run {arec.horizon} steps + {len(k['steps'])} kicks x "
          f"{len(KICK_FIELDS_)} strengths x {window} steps: {time.perf_counter()-t0:.1f}s",
          flush=True)
    save(traj, out_path)
    partial_path.unlink(missing_ok=True)
    return (f"{tag}: done ({len(k['steps'])} kicks on L{kick_layer}, strengths {KICK_FIELDS_}, "
            f"window {window}, taus {KICK_TAUS}; same-noise check passed)")


def main() -> int:
    ap = argparse.ArgumentParser(description="SGLD training (burnin) and push (kick) runs")
    ap.add_argument("--mode", required=True, choices=["burnin", "kick"])
    ap.add_argument("--kick-layer", type=int, default=None,
                    help="kick: layer to kick, 1-based (L1..L5)")
    ap.add_argument("--kick-steps", type=str, default=None,
                    help="kick: comma-separated kick steps replacing KICK_STEPS "
                         "(for tests or another set of push times)")
    ap.add_argument("--window", type=int, default=None,
                    help="kick: length of each pushed copy (short trial runs only)")
    ap.add_argument("--kick-fields", type=str, default=None,
                    help="kick: comma-separated push strengths replacing KICK_FIELDS "
                         "(output gets an _h<list> suffix)")
    ap.add_argument("--dense-every", type=int, default=0,
                    help="kick: also record every copy every N steps (readout only, "
                         "the dynamics are unchanged; output gets a _d<N> suffix)")
    ap.add_argument("--require-gpu", type=str, default=None,
                    help="abort unless torch reports exactly this device name "
                         "(e.g. 'NVIDIA A10')")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--members", type=int, required=True)
    ap.add_argument("--chunk", type=int, required=True,
                    help="must differ between runs (each chunk gets different noise)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="override the training length (short trial runs only)")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("ERROR: CUDA unavailable", file=sys.stderr)
        return 2
    if args.mode == "kick" and args.kick_layer not in (1, 2, 3, 4, 5):
        print("ERROR: kick mode needs --kick-layer in 1..5", file=sys.stderr)
        return 2
    if args.require_gpu is not None:      # noise replay is bit-exact only on the same GPU model
        name =torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        if name != args.require_gpu:
            print(f"ERROR: this task runs on '{name}', the run requires "
                  f"'{args.require_gpu}' (same GPU model as the burn-in)", file=sys.stderr)
            return 3

    rec = SGLDRecipe()
    arec = RunSettings()
    if args.horizon is not None:
        arec.horizon = args.horizon
        print(f"(short trial run) training length set to {arec.horizon}, not the paper's setting")
    seed_everything(rec.init_seed)
    run_cfg = load_config(CONFIG_PATH)
    data = make_modular_addition(build_config(DataConfig, run_cfg.get("data")))
    model_cfg = ModelConfig(n_layers=rec.n_layers, width=rec.width, bias=True)
    out_dir = data_dir() / "sgld_512"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_name = (torch.cuda.get_device_name(0)
                if args.device.startswith("cuda") else "cpu")
    print(f"[sgld_512 {args.mode}] depth {rec.n_layers} x {rec.width}, "
          f"eta={rec.eta}, wd={rec.weight_decay}, T={rec.T:g} on {args.device} "
          f"({dev_name}); members={args.members} chunk={args.chunk}")
    save_config({"recipe": asdict(rec), "aligned": asdict(arec)},
                out_dir / "phase5b_aligned_recipe.json")
    reset_sigma_fallbacks()
    if args.mode == "burnin":
        msg = run_burnin(rec, arec, args.members, args.chunk, args.device,
                         data, model_cfg, out_dir)
    else:
        ks = KICK_STEPS
        win = KICK_WINDOW
        if args.kick_steps is not None:
            ks = tuple(int(v) for v in args.kick_steps.split(","))
            print(f"(NON-DEFAULT) kick steps {len(ks)} values {ks[0]}..{ks[-1]} replace the default "
                  f"{len(KICK_STEPS)} values {KICK_STEPS[0]}..{KICK_STEPS[-1]}")
        if args.window is not None:
            win = args.window
            print(f"(short trial run) window {win} replaces {KICK_WINDOW}, not the paper's setting")
        kf = (tuple(float(v) for v in args.kick_fields.split(",")) if args.kick_fields
              else KICK_FIELDS)
        if kf != KICK_FIELDS:
            print(f"(NON-DEFAULT) kick strengths {kf} replace the default {KICK_FIELDS}")
        msg = run_kick(rec, arec, args.members, args.chunk, args.kick_layer,
                       args.device, data, model_cfg, out_dir, kick_steps=ks, window=win,
                       kick_fields=kf, dense_every=args.dense_every)
    print(f"  {msg}")
    print(f"  [readout] subspace sigma_max exact-SVD fallbacks: {SIGMA_FALLBACKS[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
