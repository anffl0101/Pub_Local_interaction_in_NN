"""Ratios of the mutual response chi_ij with each layer's overall strength removed, per network, early and late.
C2 = sqrt(chi24 chi35 / (chi23 chi45)), C3 = chi25 chi34 / (chi23 chi45): every layer of 2-5 appears once above and once below, so its factor cancels.
Early = pushes up to step 15,000, late = from step 60,000; networks with pushes on every layer in both windows; a zero pair drops the network from that cell.
Gives medians of C2, C3 (95% bootstrap interval, 2000 draws), their 10th-90th percentiles, and the shares C2 < 1 and C2 < 1 with C3 < C2.
Reads the pushes through comparisons_step_clock_pushes; writes data/comparisons/strength_free_table.json.
Usage: uv run python scripts/figures/comparisons_strength_free_table.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import step_clock_all_networks  # noqa: E402,F401  (first: all 512 networks, clock = training step)
import comparisons_step_clock_pushes as P  # noqa: E402
from comparisons_step_clock_pushes import fo  # noqa: E402
from comparisons_distance_vs_layer import pair_values, NAMES  # noqa: E402
from src.utils import data_dir  # noqa: E402

warnings.simplefilter("ignore")
WINDOWS = {"early": (0, 15001), "late": (60000, 150001)}


def median_ci(x, rng):
    """Median of x and its 95% bootstrap interval (2000 draws)."""
    b = np.median(x[rng.integers(0, len(x), (2000, len(x)))], 1)
    return [float(np.median(x)), *map(float, np.percentile(b, [2.5, 97.5]))]


def main() -> int:
    rng = np.random.default_rng(0)
    A, G, delays = P.load(); k = {t: delays.index(t) for t in delays}; ix = {n: i for i, n in enumerate(NAMES)}
    out = {}
    for lab, D in (("AdamW", A), ("SGLD", G)):
        for name, (kr, ko) in (("tau 512", (k[512], k[32])), ("tau 2", (k[2], k[2]))):
            vals = {w: pair_values(D, delays, kr, ko, *win) for w, win in WINDOWS.items()}
            nets = sorted(set(vals["early"]) & set(vals["late"]))           # the same networks early and late
            for w in WINDOWS:
                V = np.array([vals[w][n] for n in nets]); c = lambda p: V[:, ix[p]]
                ok = (V > 0).all(1)
                C2 = np.sqrt(c("2-4") * c("3-5") / (c("2-3") * c("4-5")))[ok]; C3 = (c("2-5") * c("3-4") / (c("2-3") * c("4-5")))[ok]
                row = dict(networks_table1=len(nets), networks_with_a_zero_pair=int((~ok).sum()), networks=int(ok.sum()),
                           C2_median=median_ci(C2, rng), C3_median=median_ci(C3, rng),
                           pct_C2_below_1=fo.ci((C2 < 1).astype(float), rng), pct_C2_below_1_and_C3_below_C2=fo.ci(((C2 < 1) & (C3 < C2)).astype(float), rng))
                out[f"{lab} {name} {w}"] = row
                print(f"{lab} {name} {w}: {row['networks']} of {len(nets)} nets ({row['networks_with_a_zero_pair']} with a zero pair) | "
                      f"C2 {row['C2_median'][0]:.2f} [{row['C2_median'][1]:.2f},{row['C2_median'][2]:.2f}]  C3 {row['C3_median'][0]:.2f} [{row['C3_median'][1]:.2f},{row['C3_median'][2]:.2f}] | "
                      f"C2<1 {row['pct_C2_below_1'][0]:.1f}% [{row['pct_C2_below_1'][1]:.1f},{row['pct_C2_below_1'][2]:.1f}]  both {row['pct_C2_below_1_and_C3_below_C2'][0]:.1f}% [{row['pct_C2_below_1_and_C3_below_C2'][1]:.1f},{row['pct_C2_below_1_and_C3_below_C2'][2]:.1f}]")
    (data_dir("comparisons") / "strength_free_table.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
