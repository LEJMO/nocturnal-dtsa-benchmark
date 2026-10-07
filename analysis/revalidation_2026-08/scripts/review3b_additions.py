# -*- coding: utf-8 -*-
"""Round-3b additions (external re-review, 2026-09-08).

Appends 'review3b_additions' to results/paper_stats_v1.json:
  1. Cluster-aware conditioning of the albedo relation on mean nocturnal sky
     emissivity: partial r with city-cluster permutation p and cluster-
     bootstrap CI, in all / excl-Mpls variants.  [reviewer Major 2]
  2. Correction baselines, leave-one-city-out by cluster:  [reviewer Major 6]
       (a) intercept-only bias correction (training-cluster mean removed),
       (b) albedo-only observational climatology predicting dTsa directly,
     compared with the existing albedo-conditioned model correction, all in
     K-equivalent MAE.
  3. City-equal (cluster-mean) albedo regression variants. [reviewer suggestion]
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(20260908)

ROOT = Path(__file__).resolve().parents[3]
STATS = ROOT / "results" / "paper_stats_v1.json"
K_PER_WM2 = 0.2005

d = json.load(open(STATS, encoding="utf-8"))
P = d["per_site"]
sites = sorted(P)
eps_sky = d["review3_additions"]["forcing_covariates"]["covariates"][
    "sky_emissivity"]["per_site_mean"]

def arrs(ss):
    return (np.array([P[s]["dtsa_std"] for s in ss]),
            np.array([P[s]["albedo"] for s in ss]),
            np.array([eps_sky[s] for s in ss]),
            [P[s]["cluster"] for s in ss])

def partial_r(y, x, z):
    Z = np.column_stack([z, np.ones(len(z))])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearsonr(rx, ry)[0]

def cluster_perm_partial(y, x, z, cl, n=10000):
    """Permute albedo at cluster level; recompute partial r | z each time."""
    uniq = sorted(set(cl))
    xc = {c: np.mean([x[i] for i, cc in enumerate(cl) if cc == c]) for c in uniq}
    x_use = np.array([xc[c] for c in cl])
    r_obs = partial_r(y, x_use, z)
    vals = list(xc.values())
    cnt = 0
    for _ in range(n):
        pm = dict(zip(uniq, rng.permutation(vals)))
        xp = np.array([pm[c] for c in cl])
        if abs(partial_r(y, xp, z)) >= abs(r_obs):
            cnt += 1
    return float(r_obs), (cnt + 1) / (n + 1)

def cluster_boot_partial(y, x, z, cl, n=5000):
    uniq = sorted(set(cl))
    idx = {c: [i for i, cc in enumerate(cl) if cc == c] for c in uniq}
    rs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = [i for c in pick for i in idx[c]]
        if len(set(x[ii])) < 3:
            continue
        rs.append(partial_r(y[ii], x[ii], z[ii]))
    return [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]

out = {"meta": {"script": "analysis/revalidation_2026-08/scripts/review3b_additions.py",
                "date": "2026-09-08", "seed": 20260908}}

# ---- 1. cluster-aware sky-emissivity conditioning --------------------------
cond = {}
for name, ss in {
    "all19": sites,
    "excl_mpls17": [s for s in sites if not s.startswith("US-Minneapolis")],
}.items():
    y, x, z, cl = arrs(ss)
    r0 = partial_r(y, x, z)
    rc, pp = cluster_perm_partial(y, x, z, cl)
    ci = cluster_boot_partial(y, x, z, cl)
    cond[name] = {"n": len(ss), "partial_r_record": float(r0),
                  "partial_r_clustermean_albedo": rc,
                  "cluster_perm_p": pp, "cluster_boot_ci95": ci}
out["albedo_given_skyemissivity_clusteraware"] = cond

# ---- 2. correction baselines (LOCO by cluster) -----------------------------
def loco_mae(ss, predict):
    """predict(train_idx, test_idx, data...) -> preds for test; returns MAE."""
    cl = [P[s]["cluster"] for s in ss]
    uniq = sorted(set(cl))
    errs = []
    for c in uniq:
        tr = [i for i, cc in enumerate(cl) if cc != c]
        te = [i for i, cc in enumerate(cl) if cc == c]
        errs += list(predict(ss, tr, te))
    return float(np.mean(np.abs(errs)))

def bias_arr(ss, key):
    return np.array([P[s][key] for s in ss])

results = {}
for variant, ss in {"all19": sites,
                    "excl_mpls17": [s for s in sites
                                    if not s.startswith("US-Minneapolis")]}.items():
    alb = np.array([P[s]["albedo"] for s in ss])
    dts = np.array([P[s]["dtsa_std"] for s in ss])
    row = {}
    for scheme, key in [("TEB", "teb_lwup_bias"), ("CLMU5", "clmu_lwup_bias")]:
        b = bias_arr(ss, key)
        raw = float(np.mean(np.abs(b))) * K_PER_WM2
        def pred_int(ss_, tr, te, b=b):
            mu = b[tr].mean()
            return (b[te] - mu) * K_PER_WM2
        def pred_alb(ss_, tr, te, b=b, alb=alb):
            A = np.column_stack([alb[tr], np.ones(len(tr))])
            co = np.linalg.lstsq(A, b[tr], rcond=None)[0]
            return (b[te] - (alb[te] * co[0] + co[1])) * K_PER_WM2
        row[scheme] = {
            "raw_mae_K": raw,
            "intercept_only_loco_mae_K": loco_mae(ss, pred_int),
            "albedo_conditioned_loco_mae_K": loco_mae(ss, pred_alb),
        }
    # albedo-only observational climatology predicting dTsa itself
    def pred_obs(ss_, tr, te, dts=dts, alb=alb):
        A = np.column_stack([alb[tr], np.ones(len(tr))])
        co = np.linalg.lstsq(A, dts[tr], rcond=None)[0]
        return dts[te] - (alb[te] * co[0] + co[1])
    row["obs_albedo_climatology_loco_mae_K"] = loco_mae(ss, pred_obs)
    results[variant] = row
out["correction_baselines"] = {
    "note": ("MAE in K-equivalent (biases * 0.2005). obs_albedo_climatology "
             "predicts dTsa directly from albedo (no model), the natural "
             "no-model reference for the corrected-model residuals."),
    "variants": results,
}

# ---- 3. city-equal regression ---------------------------------------------
def city_equal(ss):
    cl = {}
    for s in ss:
        cl.setdefault(P[s]["cluster"], []).append(s)
    a = np.array([np.mean([P[s]["albedo"] for s in v]) for v in cl.values()])
    y = np.array([np.mean([P[s]["dtsa_std"] for s in v]) for v in cl.values()])
    r, p = pearsonr(a, y)
    slope = np.polyfit(a, y, 1)[0] / 10.0
    return {"n_cities": len(cl), "r": float(r), "p": float(p),
            "slope_K_per_0p1_albedo": float(slope)}

out["city_equal_regression"] = {
    "all": city_equal(sites),
    "excl_mpls": city_equal([s for s in sites
                             if not s.startswith("US-Minneapolis")]),
    "conservative_core": city_equal(
        [s for s in sites
         if not s.startswith("US-Minneapolis") and s != "PL-Lipowa"]),
}

d["review3b_additions"] = out
json.dump(d, open(STATS, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("=== cluster-aware conditioning (albedo | sky emissivity) ===")
for k, v in cond.items():
    print(f"  {k:12s} partial r={v['partial_r_record']:+.3f} "
          f"perm p={v['cluster_perm_p']:.4f} CI={v['cluster_boot_ci95']}")
print("=== correction baselines (MAE, K) ===")
print(json.dumps(results, indent=1))
print("=== city-equal ===")
print(json.dumps(out["city_equal_regression"], indent=1))
