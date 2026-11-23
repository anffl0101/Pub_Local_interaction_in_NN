"""Numbers for the text and tables on the training-step clock: full order per network (Table I), first bin of the full order,
term ratios (Table II), cosines (Table VI), off-ratio (Table VII), crash-filter share, push strengths and spread, share of the push
on the strongest pattern, correlation of line-up and response of each pair.
Early = pushes up to step 15,000, late = from step 60,000. Reads data/comparisons/step_clock_pushes_{fig2,strengths}.json,
step_clock_lineup_summary.json and run outputs in data/; writes data/comparisons/step_clock_text_numbers.json.
Usage: uv run python scripts/figures/comparisons_step_clock_text_numbers.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import step_clock_all_networks  # noqa: E402,F401  (first: all 512 networks, clock = training step)
import adamw_fullbatch_512_kicks as K  # noqa: E402
import comparisons_full_order_before_after as fo  # noqa: E402
import comparisons_onestep_terms_frobenius as F  # noqa: E402
from comparisons_step_clock_pushes import load  # noqa: E402
from sgld_512_grokked import not_grokked  # noqa: E402
from src.utils import data_dir  # noqa: E402

warnings.simplefilter("ignore")
EARLY, LATE = (0, 15001), (60000, 150001)
PAIRS = ((1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4))
OFF, LINE = 0, 1                 # columns of val in lineup_off_share_sgld_top1.npz: off-ratio, line-up
C = data_dir("comparisons")
S = {}
r1 = lambda v, n=1: [round(float(x), n) for x in np.atleast_1d(v)]

# full order in each network, early and late, the same networks in both columns
A, G, delays = load()
k = {t: delays.index(t) for t in delays}
rng = np.random.default_rng(0)
S["table1"] = {}
for name, D in (("AdamW", A), ("SGLD", G)):
    for lab, (kr, ko) in (("tau512", (k[512], k[32])), ("tau2", (k[2], k[2]))):
        e, l = fo.orders(D, delays, kr, ko, *EARLY), fo.orders(D, delays, kr, ko, *LATE)
        nets = sorted(set(e) & set(l))
        x0, x1 = np.array([e[n][0] for n in nets], float), np.array([l[n][0] for n in nets], float)
        S["table1"][f"{name} {lab}"] = dict(networks_paired=len(nets), early_full_pct=fo.ci(x0, rng), late_full_pct=fo.ci(x1, rng))
        print("table1", name, lab, S["table1"][f"{name} {lab}"])

# response by distance at tau = 512: from which bin on the full order holds
f2 = json.loads((C / "step_clock_pushes_fig2.json").read_text()); edges = np.arange(0, 150001, 5000)      # bin centres (bins of comparisons_step_clock_pushes.py)
S["fig2"] = {}
for name in ("AdamW", "SGLD"):
    y = np.array([[r[0] for r in f2[name][r"$\chi_{ij}(t,512,32)$ (%) " + g]] for g in ("Neighbors", "2 apart", "3 apart")], float)
    full = (y[0] > y[1]) & (y[1] > y[2])
    fin = np.isfinite(y).all(0)                                   # bins that hold pushes
    first = next(i for i in range(len(full)) if fin[i] and full[i:][fin[i:]].all())
    S["fig2"][f"{name} tau512"] = dict(full_order_in_every_bin_from_bin_centred_at=int(edges[first]))
    print("fig2", name, "full order from the bin centred at", edges[first])

# second term / first term with 95% bootstrap intervals, by saved step
S["table2"] = {}
for name, fn, v, rows in (("AdamW", F.adamw, 1, {"10000": (10000, 10001), "20000": (20000, 20001), "40000": (40000, 40001), "from 60000": (60000, 10 ** 9)}),
                          ("SGLD", F.sgld, 0, {"20000": (20000, 20001), "40000": (40000, 40001), "from 60000": (60000, 10 ** 9)})):
    net, step, keep, size = fn()
    for lab, (a, b) in rows.items():
        s = keep & (step >= a) & (step < b); ids = np.unique(net[s])
        Sz = size[s, v][:, :, F.IN25]
        per = np.array([[Sz[net[s] == n, q].sum() for q in range(2)] for n in ids])
        boot = per[np.random.default_rng(1).integers(0, len(ids), (10000, len(ids)))].sum(1)
        g = lambda q: 100 * np.sqrt(q[..., 1] / q[..., 0])
        S["table2"][f"{name} {lab}"] = dict(networks=int(len(ids)), states=int(s.sum()), pct=[float(g(per.sum(0))), *np.percentile(g(boot), [2.5, 97.5]).tolist()])
        print("table2", name, lab, len(ids), r1(S["table2"][f"{name} {lab}"]["pct"]))

# cosine by distance: mean with 95% bootstrap interval, by saved step
D = data_dir("adamw_fullbatch_512")
Z = [np.load(D / f"saved_onestep_chunk{c}.npz") for c in range(4)]
na = np.concatenate([128 * c + z["member"] for c, z in enumerate(Z)]); sa = np.concatenate([z["step"] for z in Z]); ca = np.concatenate([z["cos"] for z in Z])
grok_a = np.unique(A[1]["net"]); ka = np.isin(na, grok_a)
z = np.load(data_dir("sgld_512") / "lineup.npz"); bad = set(not_grokked()); ns = z["chunk"] * 128 + z["member"]
ks = np.array([(int(a), int(b)) not in bad for a, b in zip(z["chunk"], z["member"])]) & (z["acc"] >= 0.5)
Gm = z["gram"].reshape(-1, 5, 2, 5, 2).sum((2, 4)); dn = np.sqrt(np.einsum("sii->si", Gm)); cs = Gm / (dn[:, :, None] * dn[:, None, :])
S["table6"] = {}
for name, cos, step, keep, net, rows in (("AdamW", ca, sa, ka, na, {"10000": (10000, 10001), "20000": (20000, 20001), "30000": (30000, 30001), "40000": (40000, 40001), "from 60000": (60000, 10 ** 9)}),
                                         ("SGLD", cs, z["step"], ks, ns, {"20000": (20000, 20001), "40000": (40000, 40001), "from 60000": (60000, 10 ** 9)})):
    rg = np.random.default_rng(0)
    for lab, (a, b) in rows.items():
        s = np.where(keep & (step >= a) & (step < b))[0]; ids, inv = np.unique(net[s], return_inverse=True)
        byn = [s[inv == q] for q in range(len(ids))]
        draws = [np.concatenate([byn[q] for q in rg.integers(0, len(ids), len(ids))]) for _ in range(500)]
        row = dict(networks=int(len(ids)), states=int(len(s)))
        for ap in (1, 2, 3):
            m = np.mean([cos[:, i, j] for i, j in PAIRS if j - i == ap], 0)
            row[f"{ap} apart"] = [float(np.mean(m[s])), *np.percentile([np.mean(m[d]) for d in draws], [2.5, 97.5]).tolist()]   # mean; 95% bootstrap interval, networks resampled
        S["table6"][f"{name} {lab}"] = row
        print("table6", name, lab, row["networks"], {q: r1(row[f"{q} apart"], 2) for q in (1, 2, 3)})

# SGLD off-ratio and line-up of the strongest pattern, means over the states from step 60,000
o = np.load(C / "lineup_off_share_sgld_top1.npz"); r = o["rows"]
s = ~np.array([(int(c), int(m)) in bad for c, m in r[:, :2]]) & (o["acc"] >= 0.5) & (r[:, 2] >= 60000)
S["off_ratio"] = dict(states=int(s.sum()), off=[float(np.mean(o["val"][s, q, OFF])) for q in (1, 2, 3)], lineup=[float(np.mean(o["val"][s, q, LINE])) for q in (1, 2, 3)])
# 95% bootstrap interval, the networks resampled (500 draws)
sel = np.where(s)[0]; nets = r[sel, 0] * 128 + r[sel, 1]; ids, inv = np.unique(nets, return_inverse=True); byn = [sel[inv == q] for q in range(len(ids))]
rgo = np.random.default_rng(0); draws = [np.concatenate([byn[q] for q in rgo.integers(0, len(ids), len(ids))]) for _ in range(500)]
S["off_ratio"]["networks"] = int(len(ids))
for nm, col in (("off", OFF), ("lineup", LINE)):
    S["off_ratio"][nm + "_band_95"] = [np.percentile([np.mean(o["val"][d, q, col]) for d in draws], [2.5, 97.5]).tolist() for q in (1, 2, 3)]
print("off-ratio", r1(S["off_ratio"]["off"], 3), "line-up", r1(S["off_ratio"]["lineup"], 3))

# share of SGLD pushed copies (h = 0.1) kept by the crash filter of the loader
G2, taus2, fields2 = K.sk.load("_h0.03-0.1", filtered=True); hi2 = list(np.round(fields2, 6)).index(0.1)
S["kept_pct"] = {}
for lab, fsel in (("step: up to 15000", lambda g: g["kstep"] <= 15000), ("step: 15000-60000", lambda g: (g["kstep"] > 15000) & (g["kstep"] < 60000)),
                  ("step: from 60000", lambda g: g["kstep"] >= 60000)):
    for t in (512, 5000):
        q = list(taus2).index(t)
        S["kept_pct"][f"{lab} tau {t}"] = float(100 * np.concatenate([G2[L]["keep"][fsel(G2[L])][:, hi2, q] for L in G2]).mean())
print("kept (%)", {q: round(v, 2) for q, v in S["kept_pct"].items()})

# one-step neighbour response from step 60,000, and the table of push strengths
st = json.loads((C / "step_clock_pushes_strengths.json").read_text()); S["strengths"] = {}
for name in ("AdamW", "SGLD"):
    for h, d in st[name].items():
        steps = np.array(d["push_steps"]); late = steps >= 60000
        nb2 = np.array([q[0] for q in d[r"$\chi_{ij}(t,2,2)$ (%) Neighbors"]], float)
        t = d["late from 60,000"]
        S["strengths"][f"{name} {h}"] = dict(onestep_neighbour_pct_from_60000=float(np.nanmean(nb2[late])), table=t)
        print("strengths", name, h, "one-step neighbours %.4f" % np.nanmean(nb2[late]), "| 512: n", t["chi_512_32"]["networks"], "full", r1(t["chi_512_32"]["full_order"]), "nb>3", r1(t["chi_512_32"]["neighbours_gt_3apart"]),
              "| one step: n", t["chi_2_2"]["networks"], "full", r1(t["chi_2_2"]["full_order"]), "nb>3", r1(t["chi_2_2"]["neighbours_gt_3apart"]))

# SGLD: spread of the neighbour response one step after the push (tau = 2) from push to push, pushes from step 60,000
def spread(x):
    x = x[np.isfinite(x)]
    return dict(mean=float(x.mean()), spread=float(x.std()), spread_over_mean=float(x.std() / abs(x.mean())), pairs=int(x.size))
S["spread"] = {}
out = {}
for L in range(2, 6):
    for c in range(4):
        for suf in ("", "_h0.03-0.1"):
            f = np.load(ROOT / f"data/sgld_512/kick_M128_chunk{c}_L{L}{suf}.npz")
            keep = np.array([(c, m) not in bad for m in range(128)])
            s = (f["kick_steps"][:, None] >= 60000) & keep[None, :]
            ok = ~(np.logical_or.accumulate(f["la_kick"][:, :, :, 1, :] < 0.5, 2) | np.logical_or.accumulate(f["la_main"][:, :, 1, :] < 0.5, 1)[:, None])
            d = f["sr_kick"] - f["sr_main"][:, None]; taus = list(f["kick_taus"])
            q = taus.index(2)
            for fi, h in enumerate(f["kick_fields"]):
                if np.round(float(h), 6) in (0.01, 0.03, 0.1):
                    out.setdefault(float(h), []).extend(d[:, fi, q, :, j][s & ok[:, fi, q]] for j in range(1, 5) if abs(j - (L - 1)) == 1)
for h in sorted(out):
    S["spread"][f"SGLD {h:g}"] = spread(np.concatenate(out[h]))
print("spread / |mean|", {q: round(v["spread_over_mean"], 2) for q, v in S["spread"].items()})

# share (r - 1) / r of the push on the strongest pattern, from step 60,000
S["push_share"] = {}
for name, files, nets in (("AdamW", [D / f"adamw_M128_chunk{c}.npz" for c in range(4)], grok_a),
                          ("SGLD", [data_dir("sgld_512") / f"burnin_M128_chunk{c}.npz" for c in range(4)], np.unique(G[1]["net"]))):
    med = []
    for L in range(2, 6):
        v = []
        for c, p in enumerate(files):
            lg = np.load(p); m = [n - 128 * c for n in nets if n // 128 == c]; late = lg["log_step"] >= 60000
            sr = lg[f"log_stable_rank_L{L}"][m][:, late]; acc = lg["log_train_acc"][m][:, late]
            v.append(sr[acc >= 0.5])
        med.append(float(np.median(np.concatenate(v))))
    S["push_share"][name] = dict(stable_rank_median_layers_2_5=med, share=[(q - 1) / q for q in med])
    print("push share", name, r1(med), r1([(q - 1) / q for q in med], 3))

# correlation across the networks of the line-up of each pair and its response, at each step; spread of the line-up
lu = json.loads((C / "step_clock_lineup_summary.json").read_text()); S["lineup"] = {}
for name in ("AdamW", "SGLD"):
    for lab, d in lu[name]["correlation"].items():
        rr = {int(t): q["r"] for t, q in d.items()}
        S["lineup"][f"{name} {lab}"] = dict(r_by_step=rr, r_range_all_steps=[min(rr.values()), max(rr.values())],
                                           lineup_spread_10_90_by_step={int(t): q["lineup_spread_10_90"] for t, q in d.items()})
        print("line-up", name, lab, "r over all steps:", r1(S["lineup"][f"{name} {lab}"]["r_range_all_steps"], 2),
              "| spread 10-90:", {t // 1000: round(v, 3) for t, v in S["lineup"][f"{name} {lab}"]["lineup_spread_10_90_by_step"].items()})

(C / "step_clock_text_numbers.json").write_text(json.dumps(S, indent=1))
