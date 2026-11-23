"""Line-up of each SGLD neighbour pair against the off-ratio f_i of the strongest output pattern of W_i and the strength sigma_1, across the networks.
Per network: mean over its saved states from step 60,000 (states inside a collapse left out); all 512 SGLD networks.
Gives 10th-90th percentiles of f and line-up, Spearman and Pearson correlations (95% bootstrap interval, 2000 draws), and partial correlations.
Reads data/comparisons/lineup_off_share_sgld_top1.npz and data/sgld_512/lineup.npz; writes data/comparisons/offratio_within_pair_summary.json.
Usage: uv run python scripts/figures/comparisons_offratio_within_pair.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import data_dir  # noqa: E402

o = np.load(data_dir("comparisons") / "lineup_off_share_sgld_top1.npz")
r = o["rows"]; k = (o["acc"] >= 0.5) & (r[:, 2] >= 60000)
net = (r[:, 0] * 128 + r[:, 1])[k]; ids, inv = np.unique(net, return_inverse=True)
per = lambda x: np.array([x[inv == q].mean(0) for q in range(len(ids))])
OFF, LU = per(o["val"][k][:, :, 0]), per(o["val"][k][:, :, 1])
s = np.load(data_dir("sgld_512") / "lineup.npz"); ks = (s["step"] >= 60000) & (s["acc"] >= 0.5)
ns = (s["chunk"] * 128 + s["member"])[ks]; idn, invn = np.unique(ns, return_inverse=True)
S1 = np.array([s["sig"][ks][:, :, 0][invn == q].mean(0) for q in range(len(idn))])
S1 = S1[[dict(zip(idn, range(len(idn))))[i] for i in ids]]
out = {"networks": int(len(ids))}
for p, name in ((1, "2-3"), (2, "3-4"), (3, "4-5")):
    f, lu = OFF[:, p], LU[:, p]
    out[name] = dict(off_ratio_p10_p90=np.percentile(f, [10, 90]).round(3).tolist(), off_ratio_mean=round(float(f.mean()), 3),
                     lineup_p10_p90=np.percentile(lu, [10, 90]).round(3).tolist(), lineup_mean=round(float(lu.mean()), 3),
                     spearman_lineup_offratio=round(float(spearmanr(lu, f)[0]), 3),
                     spearman_lineup_sigma1_lower_layer=round(float(spearmanr(lu, S1[:, p])[0]), 3),
                     spearman_lineup_sigma1_upper_layer=round(float(spearmanr(lu, S1[:, p + 1])[0]), 3),
                     pearson_lineup_offratio=round(float(pearsonr(lu, f)[0]), 3),
                     pearson_lineup_sigma1_lower_layer=round(float(pearsonr(lu, S1[:, p])[0]), 3),
                     pearson_lineup_sigma1_upper_layer=round(float(pearsonr(lu, S1[:, p + 1])[0]), 3))
    def partial(y, x, z):                                  # Pearson r of y and x with z held fixed (least-squares residuals)
        A = np.column_stack([np.ones(len(z)), z]); res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        return round(float(np.corrcoef(res(y), res(x))[0, 1]), 3)
    out[name]["pearson_sigma1_lower_layer_offratio"] = round(float(pearsonr(S1[:, p], f)[0]), 3)
    out[name]["partial_lineup_sigma1_offratio_fixed"] = partial(lu, S1[:, p], f)
    out[name]["partial_lineup_offratio_sigma1_fixed"] = partial(lu, f, S1[:, p])
    rng = np.random.default_rng(0); B = rng.integers(0, len(lu), (2000, len(lu)))
    for key, x in (("pearson_lineup_offratio_ci95", f), ("pearson_lineup_sigma1_lower_layer_ci95", S1[:, p])):
        out[name][key] = np.percentile([np.corrcoef(lu[b], x[b])[0, 1] for b in B], [2.5, 97.5]).round(3).tolist()
    print(name, out[name])
(data_dir("comparisons") / "offratio_within_pair_summary.json").write_text(json.dumps(out, indent=1))
