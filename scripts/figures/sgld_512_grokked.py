"""The SGLD networks that did not grok: median test accuracy over the last 10,000 steps below CUT.
No network ends between 0.563 and 0.850, so any cut in that gap selects the same 4 networks.
Reads data/sgld_512/burnin_M128_chunk0-3.npz; writes data/sgld_512/grokked.json, read back by not_grokked().
Run:  uv run python scripts/figures/sgld_512_grokked.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import data_dir  # noqa: E402

D = data_dir("sgld_512")
CUT = 0.8
LAST = 10000


def not_grokked() -> set[tuple[int, int]]:
    """(chunk, member) of the SGLD networks that did not grok (final test accuracy below 0.8)."""
    return {tuple(x) for x in json.loads((D / "grokked.json").read_text())["not_grokked"]}


def main() -> int:
    end, best = [], []
    for c in range(4):
        b = np.load(D / f"burnin_M128_chunk{c}.npz")
        last = b["log_step"] > b["log_step"][-1] - LAST
        end.append(np.median(b["log_test_acc"][:, last], 1))
        best.append(b["log_test_acc"].max(1))
    end, best = np.stack(end), np.stack(best)
    out = [[c, m] for c in range(4) for m in range(128) if end[c, m] < CUT]
    lo, hi = float(end[end < CUT].max()), float(end[end >= CUT].min())
    S = dict(rule=f"median test accuracy over the last {LAST} steps below {CUT}", cut=CUT,
             gap=[lo, hi], n_networks=int(end.size), n_not_grokked=len(out), not_grokked=out,
             not_grokked_end_and_best=[[float(end[c, m]), float(best[c, m])] for c, m in out],
             n_below_0p9=int((end < 0.9).sum()),
             end_test_quantiles_0_1_5_50_100=[float(q) for q in np.percentile(end, [0, 1, 5, 50, 100])])
    (D / "grokked.json").write_text(json.dumps(S, indent=1))
    print(f"{len(out)} of {end.size} not grokked (end below {CUT}; no network ends between {lo:.3f} and {hi:.3f}): {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
