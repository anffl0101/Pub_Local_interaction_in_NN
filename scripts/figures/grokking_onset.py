"""Grokking onset per network: first step at which test accuracy, smoothed over about 220 steps, stays above 0.03 for 500 steps
(None if never). Reads data/sgld_512/burnin_M128_chunk<c>.npz and data/adamw_fullbatch_512/adamw_M128_chunk<c>.npz;
writes grokking_onset_table.json into both folders.
Run:  uv run python scripts/figures/grokking_onset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import data_dir  # noqa: E402

ONSET_LEVEL, PERSIST, SMOOTH = 0.03, 500, 220


def smooth(st, y, smooth_steps=SMOOTH) -> np.ndarray:
    k = max(int(round(smooth_steps / int(st[1] - st[0]))), 1) | 1
    return np.array([np.convolve(r, np.ones(k) / k, "same") for r in y])


def sustained(st, y, thr, persist_steps=PERSIST, smooth_steps=SMOOTH) -> np.ndarray:
    """First step at which the smoothed y stays above thr for persist_steps, per network."""
    p = max(int(round(persist_steps / int(st[1] - st[0]))), 1)
    out = []
    for s in smooth(st, y, smooth_steps):
        ok = s > thr
        run, t = 0, np.nan
        for j in range(len(st)):
            run = run + 1 if ok[j] else 0
            if run >= p:
                t = st[j - p + 1]
                break
        out.append(t)
    return np.array(out, float)


def sgld_onsets() -> dict:
    """(chunk, member) -> grokking onset of the 512 SGLD networks (None if none)."""
    tab = json.loads((data_dir("sgld_512") / "grokking_onset_table.json").read_text())["networks"]
    return {(n["chunk"], n["member"]): n["onset"] for n in tab}


def onset_table(z) -> dict:
    st = z["log_step"]
    on = sustained(st, z["log_test_acc"][:, :len(st)], ONSET_LEVEL)
    return dict(rule=f"onset: smoothed test accuracy (about {SMOOTH} steps) above {ONSET_LEVEL} for {PERSIST} steps",
                networks=[dict(onset=None if np.isnan(t) else float(t)) for t in on])


def datasets():
    for folder, name in (("sgld_512", "burnin_M128_chunk{c}.npz"), ("adamw_fullbatch_512", "adamw_M128_chunk{c}.npz")):
        zs = [np.load(data_dir(folder) / name.format(c=c)) for c in range(4)]
        assert all(np.array_equal(z["log_step"], zs[0]["log_step"]) for z in zs)
        out = dict(log_step=zs[0]["log_step"], log_test_acc=np.concatenate([z["log_test_acc"] for z in zs]))
        if folder == "adamw_fullbatch_512":
            out["init_seeds"] = np.concatenate([z["init_seeds"] for z in zs])
        yield folder, out


def main() -> int:
    for folder, z in datasets():
        T = onset_table(z)
        for i, n in enumerate(T["networks"]):
            if folder == "sgld_512":
                n.update(chunk=i // 128, member=i % 128)
            else:
                n["init_seed"] = int(z["init_seeds"][i])
        print(folder, sum(n["onset"] is not None for n in T["networks"]), "of", len(T["networks"]), "networks with an onset")
        (data_dir(folder) / "grokking_onset_table.json").write_text(json.dumps(T, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
