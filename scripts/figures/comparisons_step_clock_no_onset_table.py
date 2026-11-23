"""Full order per network and line-up of the AdamW networks without a grokking onset in 150,000 steps, against those with one.
Full order as comparisons_full_order_before_after.orders: early = pushes up to step 15,000, late = from step 60,000, tau = 512 and 2.
Also the line-up averaged over the three neighbour pairs at step 10,000 and from step 60,000, and the largest and final test accuracy.
AdamW only (every SGLD network has an onset). Pushes read through comparisons_step_clock_pushes.
Reads grokking_onset_table.json, adamw_M128_chunk<c>.npz, lineup_states.npz in data/adamw_fullbatch_512/; writes data/comparisons/step_clock_no_onset_table.json.
Usage: uv run python scripts/figures/comparisons_step_clock_no_onset_table.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import step_clock_all_networks  # noqa: E402,F401
import comparisons_step_clock_pushes as P  # noqa: E402
from comparisons_step_clock_pushes import fo  # noqa: E402
from src.utils import data_dir  # noqa: E402

warnings.simplefilter("ignore")
EARLY, LATE = (0, 15001), (60000, 150001)


def main() -> int:
    A, _, delays = P.load(); k = {t: delays.index(t) for t in delays}
    DA = data_dir("adamw_fullbatch_512")
    onset = {r["init_seed"]: r["onset"] for r in json.loads((DA / "grokking_onset_table.json").read_text())["networks"]}
    te = np.concatenate([np.load(DA / f"adamw_M128_chunk{c}.npz")["log_test_acc"] for c in range(4)])
    z = np.load(DA / "lineup_states.npz"); net = z["chunk"] * 128 + z["member"]; lm = z["line"][:, 1:4].mean(1)
    groups = {"no onset": [n for n, o in onset.items() if o is None], "with onset": [n for n, o in onset.items() if o is not None]}
    S = {}
    for g, ids in groups.items():
        row = dict(networks=len(ids), largest_test_accuracy=float(te[ids].max()), final_test_accuracy_mean=float(te[ids][:, -100:].mean()),
                   lineup_step_10000=float(lm[np.isin(net, ids) & (z["step"] == 10000)].mean()),
                   lineup_from_step_60000=float(lm[np.isin(net, ids) & (z["step"] >= 60000)].mean()))
        for name, (kr, ko) in (("tau 512", (k[512], k[32])), ("tau 2", (k[2], k[2]))):
            for wl, win in (("early", EARLY), ("late", LATE)):
                o = fo.orders(A, delays, kr, ko, *win); o = {n: v for n, v in o.items() if n in set(ids)}
                row[f"{name} {wl}"] = dict(networks=len(o), full_order=int(sum(v[0] for v in o.values())))
        S[g] = row
        print(f"{g}: {row['networks']} networks, largest test accuracy {row['largest_test_accuracy']:.3f}, line-up {row['lineup_step_10000']:.2f} -> {row['lineup_from_step_60000']:.2f}")
        for kx in ("tau 512 early", "tau 512 late", "tau 2 early", "tau 2 late"):
            r = row[kx]; print(f"    {kx:14s} full order in {r['full_order']:3d} of {r['networks']:3d}")
    (data_dir("comparisons") / "step_clock_no_onset_table.json").write_text(json.dumps(S, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
