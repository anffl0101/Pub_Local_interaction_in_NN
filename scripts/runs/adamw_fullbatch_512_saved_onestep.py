"""One-step numbers (`state_numbers` of adamw_fullbatch_512_onestep.py, float64, CPU) of the 512 full-batch AdamW networks:
responses, push likeness, line-up and gain at every saved state ($GROKKING_CKPT/phase6b_adamw_M128_chunk<c>_step<t>.pt, every 10,000 steps).
Hard stop: for network 0 at steps 20,000, 80,000 and 150,000, one real AdamW update with a tiny push must reproduce the computed responses
to 2% of the row maximum (`check`). Writes data/adamw_fullbatch_512/saved_onestep_chunk<c>.npz.
Usage: python scripts/runs/adamw_fullbatch_512_saved_onestep.py --chunk C
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from src.data import DataConfig, make_modular_addition  # noqa: E402
from src.model import MLP, ModelConfig  # noqa: E402
from src.utils import build_config, data_dir, load_config  # noqa: E402
from adamw_fullbatch_512_onestep import check, state_numbers, CHECK_LIMIT  # noqa: E402

STEPS = range(10000, 150001, 10000)
CHECK_AT = (20000, 80000, 150000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, required=True)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    ck = Path(os.environ.get("GROKKING_CKPT", ROOT / "checkpoints"))
    data = make_modular_addition(build_config(DataConfig, load_config(ROOT / "configs" / "default.json").get("data")))
    x, y = data["x_train"].double(), data["y_train"]
    T = MLP.for_dataset(ModelConfig(n_layers=5, width=256, bias=True), data).double()
    out = {k: [] for k in ("step", "member", "cos", "coupling", "lineup1", "lineup5", "plain", "gain")}
    checks = {}
    for t in STEPS:
        st = torch.load(ck / f"phase6b_adamw_M128_chunk{args.chunk}_step{t:06d}.pt", map_location="cpu", weights_only=False)["state"]
        for mbr in range(128):
            pick = lambda d: {k: v[mbr].double() for k, v in d.items()}
            p, m, v = pick(st["params"]), pick(st["m"]), pick(st["v"])
            cos, C, lu1, lu5, Pl, gain = state_numbers(p, m, v, t, T, x, y)
            if mbr == 0 and t in CHECK_AT:
                worst = check(p, m, v, t, T, x, y, C)
                checks[t] = worst
                print(f"check step {t}: largest difference {worst:.4f} of the row maximum", flush=True)
                if worst > CHECK_LIMIT:
                    raise SystemExit(f"known-answer check failed at step {t}: {worst:.4f} > {CHECK_LIMIT}")
            for k, val in zip(("step", "member", "cos", "coupling", "lineup1", "lineup5", "plain", "gain"), (t, mbr, cos, C, lu1, lu5, Pl, gain)):
                out[k].append(val)
        print(f"chunk {args.chunk} step {t} done", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cfg = dict(chunk=args.chunk, steps=list(STEPS), calc="adamw_fullbatch_512_onestep.state_numbers, float64, CPU",
               checks=checks, commit=commit, torch=torch.__version__, threads=torch.get_num_threads())
    np.savez_compressed(data_dir("adamw_fullbatch_512") / f"saved_onestep_chunk{args.chunk}.npz", _config=np.array(json.dumps(cfg)),
                        **{k: np.array(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
