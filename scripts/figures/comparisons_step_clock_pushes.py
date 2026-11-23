"""Mutual response chi_ij against the training step of the push: by distance, by pair, full order per network, push strengths.
All 512 networks per optimizer, push 0.1 unless stated, AdamW pushes from step 4000; early = pushes up to step 15,000
(also 10,000), late = pushes from step 60,000 (also 40,000 and 80,000).
Parts: fig2 (Fig102, by distance), pairs (numbers only: chi_ij(t,2,2) of the neighbour pairs), sixpairs (numbers only: chi_ij(t,512,32) of the
six pairs of layers 2-5 from step 60,000), table1 (full order, also with other window limits), strengths (Fig104).
Reads the pushes through adamw_fullbatch_512_kicks and comparisons_kick_strengths; writes figures/comparisons/Fig102_step_clock_response_by_distance.pdf,
Fig104_step_clock_kick_strengths.pdf and data/comparisons/step_clock_pushes_<part>.json. Usage: uv run python scripts/figures/comparisons_step_clock_pushes.py {fig2|pairs|sixpairs|table1|strengths}
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import step_clock_all_networks  # noqa: E402,F401  (first: all 512 networks, clock = training step)
import adamw_fullbatch_512_kicks as K  # noqa: E402
import comparisons_full_order_before_after as fo  # noqa: E402
from src.utils import data_dir, figures_dir  # noqa: E402
from paper_style import DISTANCE, STYLE, TWO_COL, full_steps, panel_label, save  # noqa: E402

warnings.simplefilter("ignore")
EDGES = np.arange(-2500, 152501, 5000)      # bins centred on the multiples of 5000: an SGLD bin is one push step (10,000 ... 150,000)
DIST = {"Neighbors": [(1, 2), (2, 3), (3, 4)], "2 apart": [(1, 3), (2, 4)], "3 apart": [(1, 4)]}
PAIR = {"Layers 2–3": [(1, 2)], "Layers 3–4": [(2, 3)], "Layers 4–5": [(3, 4)]}
SIX = {"Layers 2–3": [(1, 2)], "Layers 3–4": [(2, 3)], "Layers 4–5": [(3, 4)], "Layers 2–4": [(1, 3)], "Layers 3–5": [(2, 4)], "Layers 2–5": [(1, 4)]}
SYM = {"AdamW": r"\alpha", "SGLD": "h"}
EARLY = {"up to 15,000": (0, 15001), "up to 10,000": (0, 10001)}
LATE = {"from 60,000": (60000, 150001), "from 40,000": (40000, 150001), "from 80,000": (80000, 150001)}


def step_clock(D):
    for L in D:
        D[L]["age"] = D[L]["kstep"].astype(float)
    return D


def load():
    A, delays, _, _ = K.load_adamw()
    A = step_clock({L: {k: v[A[L]["kstep"] >= 4000] for k, v in A[L].items()} for L in A})
    G, taus, fields = K.sk.load("_h0.03-0.1", filtered=True)
    hi = list(np.round(fields, 6)).index(0.1)
    G = step_clock({L: dict(d=g["d"][:, hi], net=g["net"], kstep=g["kstep"]) for L, g in G.items()})
    return A, G, delays


def panels(stats, k, groups, colors, fname, rows, styles=None, legend=None, legend_ax=(0, 0)):
    plt.rcParams.update(STYLE)
    x = 0.5 * (EDGES[:-1] + EDGES[1:]) / 1000
    fig, ax = plt.subplots(len(rows), 2, figsize=(TWO_COL, 2.4 * len(rows)), squeeze=False)
    out = {}
    for col, (lab, St) in enumerate(stats.items()):
        out[lab] = dict(networks_per_bin=St["nets"].tolist())
        for row, (key, kk, ylab) in enumerate(rows):
            a = ax[row, col]
            for gi, name in enumerate(groups):
                y = 100 * St[key][gi, :, k[kk]]; ok = np.isfinite(y[:, 0])
                a.fill_between(x[ok], y[ok, 1], y[ok, 2], color=colors[gi], alpha=0.15, lw=0)
                a.plot(x[ok], y[ok, 0], (styles[gi] if styles else "-o"), color=colors[gi], ms=2, lw=1.2, label=name)
                out[lab][f"{ylab} {name}"] = y.tolist()
            a.set_xlim(0, 150); a.set_ylim(bottom=0); a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50)); full_steps(a.xaxis)
            a.set_ylabel(ylab)
            if row == len(rows) - 1:
                a.set_xlabel("Training step")
            a.text(0.03, 0.95, lab, transform=a.transAxes, ha="left", va="top", fontsize=8)
            panel_label(a, f"({'abcd'[2 * row + col]})", dx=-34)
    ax[legend_ax].legend(**(legend or dict(loc="lower right", handlelength=1.8, fontsize=6.5)))
    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    save(fig, figures_dir("comparisons") / fname); plt.close(fig)
    print("wrote", fname)
    return out



def numbers(stats, k, groups, rows):
    """The numbers of panels() without drawing: mean and 95% interval per bin, per group and row."""
    out = {}
    for lab, St in stats.items():
        out[lab] = dict(networks_per_bin=St["nets"].tolist())
        for key, kk, ylab in rows:
            for gi, name in enumerate(groups):
                out[lab][f"{ylab} {name}"] = (100 * St[key][gi, :, k[kk]]).tolist()
    return out


def main(part) -> int:
    K.EDGES = EDGES
    S = {}
    if part in ("fig2", "pairs", "table1", "sixpairs"):
        A, G, delays = load(); k = {t: delays.index(t) for t in delays}
    if part == "fig2":
        K.GROUPS = DIST
        S = panels({"AdamW": K.stats(A, delays), "SGLD": K.stats(G, delays)}, k, DIST, [DISTANCE[1], DISTANCE[2], DISTANCE[3]],
                   "Fig102_step_clock_response_by_distance.pdf", (("group32", 512, r"$\chi_{ij}(t,512,32)$ (%)"), ("group", 2, r"$\chi_{ij}(t,2,2)$ (%)")))
    if part == "pairs":
        K.GROUPS = PAIR
        S = numbers({"AdamW": K.stats(A, delays), "SGLD": K.stats(G, delays)}, k, PAIR, (("group", 2, r"$\chi_{ij}(t,2,2)$ (%)"),))
    if part == "sixpairs":   # all six pairs of layers 2-5
        K.GROUPS = SIX
        K.EDGES = np.array([60000, 150001]); late = {}          # one bin, pushes from step 60,000
        for lab, D in (("AdamW", A), ("SGLD", G)):
            St = K.stats(D, delays)
            late[lab] = {name: (100 * St["group32"][gi, 0, k[512]]).tolist() for gi, name in enumerate(SIX)}
            print(lab, "tau 512 late:", {n: round(v[0], 2) for n, v in late[lab].items()})
        S["late_from_60000_tau512_pct_mean_ci95"] = late
        K.EDGES = EDGES
    if part == "table1":
        rng = np.random.default_rng(0)
        for lab, D in (("AdamW", A), ("SGLD", G)):
            S[lab] = {}
            for name, (kr, ko) in (("chi_512_32", (k[512], k[32])), ("chi_2_2", (k[2], k[2]))):
                S[lab][name] = {}
                for wl, win in {**{"early " + a: b for a, b in EARLY.items()}, **{"late " + a: b for a, b in LATE.items()}}.items():
                    o = fo.orders(D, delays, kr, ko, *win)
                    S[lab][name][wl] = dict(networks=len(o), full_order_pct=fo.ci(np.array([v[0] for v in o.values()], float), rng))
                    r = S[lab][name][wl]
                    print(f"{lab:5s} {name:10s} {wl:18s} n={r['networks']:3d}  full order {r['full_order_pct'][0]:5.1f}% [{r['full_order_pct'][1]:.1f}, {r['full_order_pct'][2]:.1f}]")
    if part == "strengths":
        import comparisons_kick_strengths as KS
        Da, delays = KS.adamw(); Ds, dls = KS.sgld()
        assert list(delays) == list(dls)
        k = {t: delays.index(t) for t in delays}; K.GROUPS = DIST; rng = np.random.default_rng(0)
        plt.rcParams.update(STYLE)
        fig, ax = plt.subplots(4, 3, figsize=(TWO_COL, 7.6), sharey="row", sharex=True)
        for r0, (lab, per_h) in enumerate((("AdamW", Da), ("SGLD", Ds))):
            S[lab] = {}
            for c, h in enumerate(KS.H[::-1] if float(KS.H[0]) < float(KS.H[-1]) else KS.H):
                D = step_clock(per_h[h]); steps = np.unique(D[1]["kstep"])
                K.EDGES = np.concatenate([[0], 0.5 * (steps[:-1] + steps[1:]), [150001]])     # one bin for each push step
                St = K.stats(D, delays); x = steps / 1000
                S[lab][h] = dict(push_steps=steps.tolist())
                for rr, (key, kk, ylab) in enumerate((("group32", 512, r"$\chi_{ij}(t,512,32)$ (%)"), ("group", 2, r"$\chi_{ij}(t,2,2)$ (%)"))):
                    a = ax[2 * rr + r0, c]
                    for gi, name in enumerate(DIST):
                        y = 100 * St[key][gi, :, k[kk]]; ok = np.isfinite(y[:, 0])
                        a.fill_between(x[ok], y[ok, 1], y[ok, 2], color=DISTANCE[gi + 1], alpha=0.15, lw=0)
                        a.plot(x[ok], y[ok, 0], "-o", color=DISTANCE[gi + 1], ms=1.6, lw=1.0, label=name)
                        S[lab][h][f"{ylab} {name}"] = y.tolist()
                    a.set_yscale("log"); a.set_xlim(0, 150); a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50)); full_steps(a.xaxis)
                    # ticks: long at powers of ten, short at 2-9, numbers at 2 and 5 in the AdamW rows
                    a.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1,), numticks=12))
                    a.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
                    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: np.format_float_positional(v, trim="-")))
                    a.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                        (lambda v, _: np.format_float_positional(v, trim="-") if round(v / 10 ** np.floor(np.log10(v))) in (2, 5) else "")
                        if lab == "AdamW" else (lambda v, _: "")))
                    a.text(1, 1.02, f"{lab}, ${SYM[lab]}={h}$", transform=a.transAxes, ha="right", va="bottom", fontsize=8)
                    if c == 0:
                        a.set_ylabel(ylab)
                    if 2 * rr + r0 == 3:
                        a.set_xlabel("Training step")
                    panel_label(a, f"({'abcdefghijkl'[3 * (2 * rr + r0) + c]})", dx=-30 if c == 0 else -14)
                wl = "from 60,000"
                S[lab][h]["late " + wl] = {n: KS.table_ci(D, delays, LATE[wl], t, to) for n, (t, to) in (("chi_512_32", (512, 32)), ("chi_2_2", (2, 2)))}
                r = S[lab][h]["late " + wl]["chi_512_32"]
                print(f"{lab} h={h} late {wl}: n={r['networks']} full order {r['full_order'][0]:.1f}%  nb>3ap {r['neighbours_gt_3apart'][0]:.1f}%  | one step: full {S[lab][h]['late ' + wl]['chi_2_2']['full_order'][0]:.1f}%")
        ax[0, 0].legend(loc="lower right", fontsize=6.5, handlelength=1.6)
        fig.tight_layout(h_pad=1.0, w_pad=0.6)
        save(fig, figures_dir("comparisons") / "Fig104_step_clock_kick_strengths.pdf"); print("wrote Fig104")
    (data_dir("comparisons") / f"step_clock_pushes_{part}.json").write_text(json.dumps(S, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
