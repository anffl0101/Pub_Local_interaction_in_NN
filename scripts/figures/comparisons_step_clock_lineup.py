"""Line-up |u_1(W_i).v_1(W_i+1)| of neighbouring layers against the training step, and its relation to the one-step response chi_ij(t,2,2).
All 512 networks per optimizer; 95% bootstrap intervals, networks resampled. Reads data/adamw_fullbatch_512/lineup_states.npz, data/sgld_512/lineup.npz, pushes via comparisons_lineup_quartiles.
--mean-pairs: mean of the three neighbour pairs, the neighbour response on a second axis (needs data/comparisons/step_clock_pushes_fig2.json from comparisons_step_clock_pushes.py fig2),
  response against line-up in ten groups at five steps; writes figures/comparisons/Fig111_step_clock_lineup_mean_pairs.pdf, data/comparisons/step_clock_lineup_mean_pairs_summary.json.
Without it: correlation across the networks of the line-up of each pair and its chi_ij(t,2,2), each state divided by its own change of the
  pushed layer (shuffle test, Benjamini-Hochberg q = 0.05); writes data/comparisons/step_clock_lineup_summary.json (no figure).
Usage: uv run python scripts/figures/comparisons_step_clock_lineup.py [--mean-pairs]
"""
import contextlib
import io
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
with contextlib.redirect_stdout(io.StringIO()):
    import comparisons_lineup_quartiles as Q
    import comparisons_lineup_response_ensemble_correlation as CR
from src.utils import data_dir, figures_dir  # noqa: E402
from paper_style import INK, LAYER, STYLE, TWO_COL, full_steps, panel_label, save  # noqa: E402

warnings.simplefilter("ignore")
PAIRS = ((1, "2–3"), (2, "3–4"), (3, "4–5"))
RESP = "#D55E00"                             # colour of the response curve on the second axis
MEAN_PAIRS = "--mean-pairs" in sys.argv      # average over the three neighbour pairs
CD_STEPS = (20000, 40000, 60000, 100000, 140000)      # steps of the response-against-line-up curves, the same for AdamW and SGLD
GROUP_STEPS = {"AdamW": (10000, 30000), "SGLD": (20000, 140000)}
NQ, NBOOT, NPERM = 10, 500, 2000
AVG = np.mean if MEAN_PAIRS else np.median   # mean over the networks (median for the single pairs)
UNRELATED = 0.050 if MEAN_PAIRS else 0.042   # |u.v| of two random patterns of 256 entries: mean 0.050, median 0.042


def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        data = Q.load()
    raw = {"AdamW": np.load(ROOT / "data/adamw_fullbatch_512/lineup_states.npz"), "SGLD": np.load(ROOT / "data/sgld_512/lineup.npz")}
    rng, rngx = np.random.default_rng(0), np.random.default_rng(1)
    S = {}
    plt.rcParams.update(STYLE)
    fig, axs = plt.subplots(2, 2, figsize=(TWO_COL, 4.9))
    Q.NQ = NQ
    for col, name in enumerate(("AdamW", "SGLD")):
        T, dl = data[name]; k2 = dl.index(2)
        T["step"] = np.round(T["age"]).astype(int)                  # the loaders give the push step as "age" (step_clock_all_networks.py)
        nets = np.unique(T["net"]); z = raw[name]; zn = z["chunk"] * 128 + z["member"]
        keep = np.isin(zn, nets) & (z["step"] > 0) & ((z["acc"] >= 0.5) if "acc" in z.files else True)
        S[name] = dict(networks=int(len(nets)), lineup={}, groups={}, correlation={})
        # (a), (b) line-up against the saved step
        ax = axs[0, col]; steps = np.unique(z["step"][keep]); steps = steps[steps <= 150000]
        for p, lab in ([(None, "mean of the three neighbor pairs")] if MEAN_PAIRS else PAIRS):
            med, band = [], []
            for t in steps:
                row = z["line"][keep & (z["step"] == t)]
                v = row[:, 1:4].mean(1) if p is None else row[:, p]      # mean over the pairs 2-3, 3-4, 4-5 of one network
                med.append(AVG(v)); band.append(np.percentile([AVG(rng.choice(v, len(v))) for _ in range(NBOOT)], [2.5, 97.5]))
            med, band = np.array(med), np.array(band)
            colour = "#0072B2" if p is None else LAYER[p]                # DISTANCE[1], the colour of 'Neighbors'
            ax.fill_between(steps / 1000, band[:, 0], band[:, 1], color=colour, alpha=0.15, lw=0)
            ax.plot(steps / 1000, med, "-o", ms=2, lw=1.1, color=colour, label="Neighbors" if p is None else f"Layers {lab}")
            S[name]["lineup"][lab] = {int(t): float(m) for t, m in zip(steps, med)}
        ax.axhline(UNRELATED, color=INK, lw=0.6, ls=":"); ax.set_ylim(0, 1.05); ax.set_xlim(0, 150)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50)); full_steps(ax.xaxis); ax.set_xlabel("Training step")
        ax.text(0.03, 0.95, name, transform=ax.transAxes, ha="left", va="top", fontsize=8); panel_label(ax, "(" + "ab"[col] + ")")
        if MEAN_PAIRS:          # one-step response of neighbours on a second axis, scaled to agree with the line-up from step 60,000 on
            F2 = json.loads((data_dir("comparisons") / "step_clock_pushes_fig2.json").read_text())
            xr = np.arange(0, 150001, 5000); yr = np.array([q[0] for q in F2[name][r"$\chi_{ij}(t,2,2)$ (%) Neighbors"]], float); okr = np.isfinite(yr)
            factor = float(np.nanmean(yr[okr & (xr >= 60000)]) / np.mean(med[steps >= 60000]))
            lo_, hi_ = (np.array([q[k_] for q in F2[name][r"$\chi_{ij}(t,2,2)$ (%) Neighbors"]], float) for k_ in (1, 2))    # its 95% bootstrap interval
            bx = ax.twinx(); bx.fill_between(xr[okr] / 1000, lo_[okr], hi_[okr], color=RESP, alpha=0.2, lw=0)
            bx.plot(xr[okr] / 1000, yr[okr], "--s", ms=1.8, lw=1.0, color=RESP)
            S[name]["lineup_band_largest_half_width"] = float(np.max(band[:, 1] - band[:, 0]) / 2)
            print(f"{name}: 95% range of the line-up: at most +-{S[name]['lineup_band_largest_half_width']:.3f}")
            bx.set_ylim(0, 1.05 * factor); bx.tick_params(axis="y", colors=RESP)
            if col == 1:
                bx.set_ylabel(r"$\chi_{ij}(t,2,2)$ (%)", color=RESP)
            S[name]["response_axis_factor"] = factor
            print(f"{name}: right axis = left axis x {factor:.4f} (the two curves agree from step 60,000 on)")
        # (c), (d) ten line-up groups formed at each saved step
        ax = axs[1, col]; lo, hi = GROUP_STEPS[name]; T["age"] = T["step"].astype(float)
        tsteps = np.unique(T["step"]); tsteps = tsteps[(tsteps >= lo) & (tsteps <= hi)]
        edges = [t - 1 for t in tsteps] + [tsteps[-1] + 1]
        if MEAN_PAIRS:      # (c), (d) mean of the three neighbour pairs, one curve for each of five steps
            # each group divided by its own change of the pushed layer, as in the definition of chi_ij
            # separate random generator rng3 for (c), (d)
            ml = T["line"][:, 1:4].mean(1); R = T["R"]; rng3, cmap, per, corr = np.random.default_rng(2), plt.get_cmap("viridis"), {}, {}

            def chi_c(num, den):
                return float(np.mean([100 * np.sqrt(abs((np.nanmean(R[num, p_, k2, p_ + 1]) / np.nanmean(R[den, p_, k2, p_]))
                                                        * (np.nanmean(R[num, p_ + 1, k2, p_]) / np.nanmean(R[den, p_ + 1, k2, p_ + 1])))) for p_, _ in PAIRS]))
            for n_, t in enumerate(CD_STEPS):
                rows_ = np.where(T["step"] == t)[0]; assert len(rows_) == len(np.unique(T["net"][rows_])), "one state for each network at a step"
                gq = (np.argsort(np.argsort(ml[rows_], kind="stable"), kind="stable") * NQ) // len(rows_)     # ten groups by the line-up at this step
                draws = [rng3.integers(0, len(rows_), len(rows_)) for _ in range(NBOOT)]                   # the networks of the step, resampled
                pts = []
                for q in range(NQ):
                    mine = rows_[gq == q]
                    pts.append(dict(networks=int(len(mine)), lineup=float(ml[mine].mean()), pair=chi_c(mine, mine),
                                    pair_band=[float(v) for v in np.nanpercentile([chi_c(rows_[d][gq[d] == q], rows_[d][gq[d] == q]) for d in draws], [2.5, 97.5])]))
                xs, ys, bb = (np.array([q_[k] for q_ in pts]) for k in ("lineup", "pair", "pair_band"))
                ax.errorbar(xs, ys, yerr=[ys - bb[:, 0], bb[:, 1] - ys], fmt="-o", ms=2.2, lw=1.0, capsize=1.2, elinewidth=0.6,
                            color=cmap(0.9 * n_ / (len(CD_STEPS) - 1)), label=f"Step {t:,}".replace(",", " "))
                y = np.nanmean([CR.response(R[rows_], p_, p_ + 1, k2, k2) for p_, _ in PAIRS], 0)   # Pearson r across the networks at this step
                corr[int(t)] = dict(r=CR.rho(ml[rows_], y), networks=int(np.isfinite(y).sum()))
                per[int(t)] = dict(networks=int(len(rows_)), groups=pts)
                print(f"{name} step {t}: {len(rows_)} networks, r = {corr[int(t)]['r']:+.2f}, groups " + " ".join(f"({q_['lineup']:.2f}, {q_['pair']:.3f})" for q_ in pts))
            ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.9), handlelength=1.6)
            S[name]["mean_pairs"] = dict(correlation=corr, steps=[int(t) for t in CD_STEPS], each_step=per)
        for p, lab in ([] if MEAN_PAIRS else PAIRS):      # line-up groups of each pair; the correlation bands below use the same random generator after them
            g = Q.groups(T, p, edges); fn = lambda X: Q.chi(X, p, p + 1, k2, k2); rows = []
            for q in range(NQ):
                s = g == q; R, net = T["R"][s], T["net"][s]
                rows.append(dict(states=int(s.sum()), lineup=float(T["line"][s, p].mean()), lineup_band=Q.band_mean(T["line"][s, p], net, rngx),
                                 pair=float(fn(R)), pair_band=Q.band(R, net, fn, rng)))
            S[name]["groups"][lab] = rows
        ax.axvline(UNRELATED, color=INK, lw=0.6, ls=":"); ax.axhline(0, color=INK, lw=0.4); ax.set_xlim(0, 0.6 if MEAN_PAIRS and name == "SGLD" else 1); ax.set_xlabel("Line-up of neighbors" if MEAN_PAIRS else "Line-up of the pair")
        ax.text(0.03, 0.95, name, transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        panel_label(ax, "(" + "cd"[col] + ")")
        # correlation across the networks at each saved step (in both modes: the line-up band of the next column uses the same random generator after it)
        rows = []
        for t in np.unique(T["step"]):
            s = np.where(T["step"] == t)[0]
            if len(s) < CR.MIN_STATES:
                continue
            R = T["R"][s]
            for p, lab in PAIRS:
                line = T["line"][s, p]; y = CR.response(R, p, p + 1, k2, k2)
                if np.isfinite(y).sum() < CR.MIN_STATES:
                    continue
                r0 = CR.rho(line, y)
                boot = []
                for _ in range(NBOOT):
                    d = rng.integers(0, len(s), len(s)); boot.append(CR.rho(line[d], CR.response(R[d], p, p + 1, k2, k2)))
                perm = np.array([CR.rho(rng.permutation(line), y) for _ in range(NPERM)])
                rows.append((p, lab, int(t), dict(networks=int(np.isfinite(y).sum()), r=r0, band=np.nanpercentile(boot, [2.5, 97.5]).tolist(),
                                                  p=float((np.sum(np.abs(perm) >= abs(r0)) + 1) / (NPERM + 1)),
                                                  lineup_median=float(np.median(line)), lineup_spread_10_90=float(np.subtract(*np.percentile(line, [90, 10]))))))
        sig = CR.bh([r[3]["p"] for r in rows])
        for p, lab in PAIRS:
            mine = [(t, d, sg) for (pp, _, t, d), sg in zip(rows, sig) if pp == p]
            S[name]["correlation"][lab] = {int(t): dict(d, significant=bool(x)) for t, d, x in mine}
            print(f"{name} layers {lab} r at each step: " + "  ".join(f"{t // 1000}k:{d['r']:+.2f}{'' if x else '?'}" for t, d, x in mine))
    axs[0, 0].set_ylabel(r"$|u_1(W_i)\cdot v_1(W_{i+1})|$"); axs[1, 0].set_ylabel(r"$\chi_{ij}(t,2,2)$ (%)")
    if MEAN_PAIRS:
        from matplotlib.lines import Line2D
        axs[0, 0].legend(handles=[Line2D([], [], color="#0072B2", marker="o", ms=2, lw=1.1, label="Line-up of neighbors (left axis)"),
                                  Line2D([], [], color=RESP, marker="s", ms=1.8, lw=1.0, ls="--", label="Response of neighbors (right axis)")],
                         fontsize=6.5, loc="lower right", handlelength=2.2)
    fig.tight_layout(w_pad=1.5, h_pad=1.2)
    if MEAN_PAIRS:
        save(fig, figures_dir("comparisons") / "Fig111_step_clock_lineup_mean_pairs.pdf")
        (data_dir("comparisons") / "step_clock_lineup_mean_pairs_summary.json").write_text(json.dumps({n_: dict(S[n_]["lineup"], response_axis_factor=S[n_]["response_axis_factor"], lineup_band_largest_half_width=S[n_]["lineup_band_largest_half_width"], **S[n_]["mean_pairs"]) for n_ in S}, indent=1))
        print("wrote Fig111"); return 0
    plt.close(fig)                                  # without --mean-pairs only the numbers are written
    (data_dir("comparisons") / "step_clock_lineup_summary.json").write_text(json.dumps(S, indent=1))
    print("wrote data/comparisons/step_clock_lineup_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
