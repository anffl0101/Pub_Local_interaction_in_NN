"""Largest Hessian eigenvalue of the training loss around the collapses of the grokked SGLD networks, against the limit 2/eta.
A collapse: training accuracy falls below 0.5 after 200 steps without a point below 0.5.
Reads data/sgld_512/sharpness_chunk0-3.npz; writes figures/sgld_512/Fig97_sharpness_collapses.pdf and data/sgld_512/sharpness_summary.json.
Run:  uv run python scripts/figures/sgld_512_sharpness_collapses.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from src.utils import data_dir, figures_dir  # noqa: E402
from paper_style import INK, METHOD, STYLE, TWO_COL, panel_label, save, step_number  # noqa: E402
from sgld_512_grokked import not_grokked  # noqa: E402

WIN = 10          # logged points (200 steps) on each side of a collapse
BEFORE = 3        # logged points (60 steps) before a collapse
YCUT = 30         # panel (b): the band is cut here (its top is 121 at the collapse)
YLOW = 10         # panel (b): lower end of the axis; the band never goes below 13


def load():
    bad, step, acc, lam, ids = not_grokked(), None, [], [], []
    for c in range(4):
        z = np.load(data_dir("sgld_512") / f"sharpness_chunk{c}.npz", allow_pickle=True)
        keep = [m for m in range(z["train_acc"].shape[0]) if (c, m) not in bad]
        step = z["step"]; acc.append(z["train_acc"][keep]); lam.append(z["sharpness"][keep]); ids += [(c, m) for m in keep]
        cfg = json.loads(str(z["_config"]))
    return step, np.concatenate(acc), np.concatenate(lam), ids, 2 / cfg["sgld"]["eta"]


def analyse(step, acc, lam, limit):
    falls = [(m, k) for m in range(acc.shape[0]) for k in np.where((acc[m, :-1] >= 0.5) & (acc[m, 1:] < 0.5))[0] + 1]
    events = [(m, k) for m, k in falls if k >= WIN and acc[m, k - WIN:k].min() >= 0.5]
    rel = np.arange(-WIN, WIN + 1)
    traces = np.full((len(events), len(rel)), np.nan)
    for n, (m, k) in enumerate(events):
        ok = (k + rel >= 0) & (k + rel < lam.shape[1])
        traces[n, ok] = lam[m, k + rel[ok]]
    before = np.nanmax(traces[:, WIN - BEFORE:WIN], 1)            # nan if the collapse is at the first logged point
    before = before[np.isfinite(before)]
    quiet = lam[acc >= 0.99]
    pc = lambda v: dict(median=float(np.nanmedian(v)), **{f"p{q}": float(np.nanpercentile(v, q)) for q in (10, 90)})
    per_net = np.bincount([m for m, _ in events], minlength=acc.shape[0])
    summary = dict(
        networks=int(acc.shape[0]), steps=int(step[-1] - step[0] + step[1] - step[0]), log_every=int(step[1] - step[0]),
        stability_limit=limit, collapses=len(events), networks_with_a_collapse=int((per_net > 0).sum()),
        sharpness_quiet=dict(median=float(np.nanmedian(quiet)), share_above_limit=float(np.mean(quiet > limit))),
        largest_in_60_steps_before_collapse=dict(collapses=int(before.size), above_limit=int((before > limit).sum()),
                                                 share_above_limit=float(np.mean(before > limit))),
        sharpness_relative_to_collapse={int(r) * int(step[1] - step[0]): pc(traces[:, q]) for q, r in enumerate(rel)},
    )
    return events, rel, traces, before, quiet, summary


def main() -> int:
    step, acc, lam, ids, limit = load()
    events, rel, traces, before, quiet, S = analyse(step, acc, lam, limit)
    (data_dir("sgld_512") / "sharpness_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k != "sharpness_relative_to_collapse"}, indent=1))
    print("median curvature against steps from the collapse:", {k: round(v["median"], 1) for k, v in S["sharpness_relative_to_collapse"].items()})

    plt.rcParams.update(STYLE)
    blue = METHOD["SGLD"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TWO_COL, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a.plot(step, lam[0], color=blue, lw=0.9)
    a.axhline(limit, color=INK, lw=0.8, ls="--")
    a.set_xticks([101000, 103000, 105000]); a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: step_number(v)))
    a.set_xlabel("Training step"); a.set_ylabel("Largest curvature")
    a2 = a.twinx(); a2.plot(step, acc[0], color="#888888", lw=0.8); a2.set_ylim(0, 1.05)
    a2.set_ylabel("Training accuracy")
    panel_label(a, "(a)")

    x = rel * S["log_every"]
    b.fill_between(x, np.nanpercentile(traces, 10, 0), np.nanpercentile(traces, 90, 0), color=blue, alpha=0.2, lw=0)
    b.plot(x, np.nanmedian(traces, 0), color=blue, marker="o", ms=2.5, lw=1.2)
    b.axhline(limit, color=INK, lw=0.8, ls="--"); b.axvline(0, color=INK, lw=0.6)
    b.set_ylim(top=45)                      # layout is made at this limit, so both panels keep their place
    b.set_xlabel("Steps from the collapse"); b.set_ylabel("Largest curvature")
    panel_label(b, "(b)")

    fig.tight_layout(w_pad=2.0)
    b.set_ylim(YLOW, YCUT); b.set_yticks(range(YLOW, YCUT + 1, 5))      # ticks every 5
    save(fig, figures_dir("sgld_512") / "Fig97_sharpness_collapses.pdf")
    print("wrote figures/sgld_512/Fig97_sharpness_collapses.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
