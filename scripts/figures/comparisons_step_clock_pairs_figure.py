"""Each neighbour pair against the training step: line-up, one-step response chi_ij(t,2,2), and their correlation across the networks.
Line-up: mean over the networks, 95% bootstrap interval, networks resampled. Correlation: open symbols not significant (q = 0.05).
All 512 networks per optimizer; SGLD states inside a collapse left out.
Reads data/comparisons/step_clock_pushes_pairs.json, step_clock_lineup_summary.json, data/adamw_fullbatch_512/lineup_states.npz, data/sgld_512/lineup.npz.
Writes figures/comparisons/Fig108_step_clock_pairs_response_and_correlation.pdf and data/comparisons/step_clock_pairs_lineup_mean.json.
Usage: uv run python scripts/figures/comparisons_step_clock_pairs_figure.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from src.utils import data_dir, figures_dir  # noqa: E402
from paper_style import INK, LAYER, STYLE, TWO_COL, full_steps, panel_label, save  # noqa: E402

D = data_dir("comparisons")
R = json.loads((D / "step_clock_pushes_pairs.json").read_text())
C = json.loads((D / "step_clock_lineup_summary.json").read_text())
raw = {"AdamW": np.load(ROOT / "data/adamw_fullbatch_512/lineup_states.npz"), "SGLD": np.load(ROOT / "data/sgld_512/lineup.npz")}
x = np.arange(0, 150001, 5000) / 1000          # bin centres (bins of comparisons_step_clock_pushes.py)
rng = np.random.default_rng(0)
LU = {}                                         # the line-up curves of (a), (b), written to step_clock_pairs_lineup_mean.json
plt.rcParams.update(STYLE)
fig, ax = plt.subplots(3, 2, figsize=(TWO_COL, 7.0))
for col, name in enumerate(("AdamW", "SGLD")):
    t0, a, b = ax[0, col], ax[1, col], ax[2, col]
    z = raw[name]; keep = (z["step"] > 0) & (z["step"] <= 150000) & ((z["acc"] >= 0.5) if "acc" in z.files else True)
    steps = np.unique(z["step"][keep])
    for p, lab in ((1, "2–3"), (2, "3–4"), (3, "4–5")):
        med, band = [], []
        for t in steps:
            v = z["line"][keep & (z["step"] == t), p]
            med.append(np.mean(v)); band.append(np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(500)], [2.5, 97.5]))   # mean over the networks, 95% bootstrap interval
        med, band = np.array(med), np.array(band)
        LU.setdefault(name, {})[lab] = dict(mean={int(t): float(m) for t, m in zip(steps, med)}, band_95={int(t): [float(q) for q in bb] for t, bb in zip(steps, band)})
        print(f"{name} layers {lab}: mean line-up " + " ".join(f"{int(t) // 1000}k:{m:.3f}" for t, m in zip(steps, med)) + f" | 95% range at most +-{np.max(band[:, 1] - band[:, 0]) / 2:.3f}")
        t0.fill_between(steps / 1000, band[:, 0], band[:, 1], color=LAYER[p], alpha=0.15, lw=0)
        t0.plot(steps / 1000, med, "-o", ms=2, lw=1.1, color=LAYER[p], label=f"Layers {lab}")
        y = np.array(R[name][r"$\chi_{ij}(t,2,2)$ (%) Layers " + lab], float); ok = np.isfinite(y[:, 0])
        a.fill_between(x[ok], y[ok, 1], y[ok, 2], color=LAYER[p], alpha=0.15, lw=0)
        a.plot(x[ok], y[ok, 0], "-o", color=LAYER[p], ms=2, lw=1.2)
        rows = sorted((int(t), d) for t, d in C[name]["correlation"][lab].items())
        bx = np.array([t for t, _ in rows]) / 1000; v = np.array([d["r"] for _, d in rows])
        bd = np.array([d["band"] for _, d in rows]); sg = np.array([d["significant"] for _, d in rows])
        b.fill_between(bx, bd[:, 0], bd[:, 1], color=LAYER[p], alpha=0.15, lw=0)
        b.plot(bx, v, "-", color=LAYER[p], lw=1.1)
        b.plot(bx[sg], v[sg], "o", color=LAYER[p], ms=3); b.plot(bx[~sg], v[~sg], "o", color=LAYER[p], ms=3, mfc="white", mew=0.8)
    t0.axhline(0.050, color=INK, lw=0.6, ls=":"); t0.set_ylim(0, 1.05)       # mean |u.v| of two random patterns of 256 entries
    a.set_ylim(bottom=0); b.axhline(0, color=INK, lw=0.6); b.set_ylim(-0.45, 1.0); b.set_xlabel("Training step")
    for row, q in enumerate((t0, a, b)):
        q.set_xlim(0, 150); q.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50)); full_steps(q.xaxis)
        q.text(0.03, 0.05 if row == 2 else 0.95, name, transform=q.transAxes, ha="left", va="bottom" if row == 2 else "top", fontsize=8)
        panel_label(q, "(" + "abcdef"[2 * row + col] + ")", dx=-34 if col == 0 else -22)
ax[0, 0].set_ylabel(r"$|u_1(W_i)\cdot v_1(W_{i+1})|$"); ax[1, 0].set_ylabel(r"$\chi_{ij}(t,2,2)$ (%)")
ax[2, 0].set_ylabel("Correlation of line-up\nand " + r"$\chi_{ij}(t,2,2)$")
ax[0, 0].legend(loc="lower right", handlelength=1.8, fontsize=6.5)
fig.tight_layout(h_pad=1.0, w_pad=1.5)
save(fig, figures_dir("comparisons") / "Fig108_step_clock_pairs_response_and_correlation.pdf")
(D / "step_clock_pairs_lineup_mean.json").write_text(json.dumps(LU, indent=1))
print("wrote Fig108")
