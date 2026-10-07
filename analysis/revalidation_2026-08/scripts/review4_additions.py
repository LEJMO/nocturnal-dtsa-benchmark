# -*- coding: utf-8 -*-
"""Round-4 additions (external review of v5, 2026-09-09).

Appends 'review4_additions' to results/paper_stats_v1.json. Fixes the P0
fold/target inconsistency and supplies the demanded inference:

  1. UNIFIED correction analysis — single fold type (leave-one-city-cluster-
     out, LOCO) and single exact target (record-mean nocturnal dTsa error,
     model minus observation, in K; no W/m^2->K conversion anywhere):
       raw | intercept-only | albedo-conditioned, for TEB & CLMU5,
       plus the no-model albedo climatology (predicting obs dTsa),
       variants all19 / excl-Mpls17; R2_LOCO = 1 - SSres/SStot with
       SStot about the full-sample mean error; per-city paired values and
       cluster-bootstrap 95% CIs of the MAE improvement.
     Supersedes 'prescription' (record-LOO, W/m^2-converted) and the
     review3b baselines for manuscript use.
  2. Compression-slope inference — cluster-bootstrap 95% CI for the slope of
     model dTsa on observed dTsa (TEB, CLMU5), beta=1 exclusion, variants
     all / excl-Mpls / conservative core / city-equal.
  3. Freedman–Lane-style cluster permutation for the albedo association
     conditioned on mean nocturnal sky emissivity (residualize both dTsa and
     albedo on eps_sky, then permute the albedo residuals across clusters).
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(20260909)

ROOT = Path(__file__).resolve().parents[3]
STATS = ROOT / "results" / "paper_stats_v1.json"

d = json.load(open(STATS, encoding="utf-8"))
P = d["per_site"]
sites = sorted(P)
eps_sky = d["review3_additions"]["forcing_covariates"]["covariates"][
    "sky_emissivity"]["per_site_mean"]

out = {"meta": {"script": "analysis/revalidation_2026-08/scripts/review4_additions.py",
                "date": "2026-09-09", "seed": 20260909,
                "supersedes": ["prescription (record-LOO, W/m2-converted)",
                               "review3b_additions.correction_baselines"]}}

def city_folds(ss):
    cl = [P[s]["cluster"] for s in ss]
    return [[i for i, c in enumerate(cl) if c == u] for u in sorted(set(cl))], cl

def loco_errs(ss, y, x, mode):
    """LOCO residuals for y given x. mode: 'albedo' | 'intercept'."""
    folds, _ = city_folds(ss)
    e = np.zeros(len(ss))
    for f in folds:
        tr = [i for i in range(len(ss)) if i not in f]
        if mode == "albedo":
            A = np.column_stack([x[tr], np.ones(len(tr))])
            co = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            pred = x[np.array(f)] * co[0] + co[1]
        else:
            pred = np.full(len(f), y[tr].mean())
        e[np.array(f)] = y[np.array(f)] - pred
    return e

def cboot_mae_diff(ss, e_raw, e_corr, n=5000):
    """Cluster-bootstrap CI of MAE(raw)-MAE(corr)."""
    folds, cl = city_folds(ss)
    uniq = sorted(set(cl))
    idx = {u: [i for i, c in enumerate(cl) if c == u] for u in uniq}
    diffs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = [i for u in pick for i in idx[u]]
        diffs.append(np.mean(np.abs(e_raw[ii])) - np.mean(np.abs(e_corr[ii])))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

# ---- 1. unified correction analysis ---------------------------------------
uni = {}
for variant, ss in {"all19": sites,
                    "excl_mpls17": [s for s in sites
                                    if not s.startswith("US-Minneapolis")]}.items():
    alb = np.array([P[s]["albedo"] for s in ss])
    obs = np.array([P[s]["dtsa_std"] for s in ss])
    row = {}
    for scheme, key in [("TEB", "teb_dtsa"), ("CLMU5", "clmu_dtsa")]:
        e0 = np.array([P[s][key] for s in ss]) - obs      # exact K error
        e_int = loco_errs(ss, e0, alb, "intercept")
        e_alb = loco_errs(ss, e0, alb, "albedo")
        r2 = 1 - np.sum(e_alb ** 2) / np.sum((e0 - e0.mean()) ** 2)
        row[scheme] = {
            "raw_mae_K": float(np.mean(np.abs(e0))),
            "intercept_only_locomae_K": float(np.mean(np.abs(e_int))),
            "albedo_locomae_K": float(np.mean(np.abs(e_alb))),
            "r2_loco": float(r2),
            "mae_improvement_ci95_cluster": cboot_mae_diff(ss, e0, e_alb),
            "per_record_raw_err_K": dict(zip(ss, [float(v) for v in e0])),
            "per_record_alb_corrected_err_K": dict(zip(ss, [float(v) for v in e_alb])),
        }
    e_clim = loco_errs(ss, obs, alb, "albedo")            # no-model reference
    row["obs_albedo_climatology"] = {
        "locomae_K": float(np.mean(np.abs(e_clim))),
        "per_record_err_K": dict(zip(ss, [float(v) for v in e_clim]))}
    uni[variant] = row
out["unified_correction_LOCO_dtsa"] = {
    "target": ("record-mean nocturnal dTsa error (model - obs, K); "
               "R2_LOCO = 1 - SSres/SStot, SStot about the full-sample mean "
               "raw error; folds = city clusters"),
    "variants": uni}

# ---- 2. compression-slope inference ---------------------------------------
def slope_ci(ss, key, n=5000):
    obs = np.array([P[s]["dtsa_std"] for s in ss])
    mod = np.array([P[s][key] for s in ss])
    b = float(np.polyfit(obs, mod, 1)[0])
    folds, cl = city_folds(ss)
    uniq = sorted(set(cl))
    idx = {u: [i for i, c in enumerate(cl) if c == u] for u in uniq}
    bs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = [i for u in pick for i in idx[u]]
        if len(set(obs[ii])) < 3:
            continue
        bs.append(np.polyfit(obs[ii], mod[ii], 1)[0])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return {"n": len(ss), "slope": b, "ci95_cluster": ci,
            "excludes_unity": bool(ci[1] < 1.0)}

def city_equal_slope(ss, key):
    cl = {}
    for s in ss:
        cl.setdefault(P[s]["cluster"], []).append(s)
    o = np.array([np.mean([P[s]["dtsa_std"] for s in v]) for v in cl.values()])
    m = np.array([np.mean([P[s][key] for s in v]) for v in cl.values()])
    return float(np.polyfit(o, m, 1)[0])

sl = {}
VAR = {"all19": sites,
       "excl_mpls17": [s for s in sites if not s.startswith("US-Minneapolis")],
       "core16": [s for s in sites
                  if not s.startswith("US-Minneapolis") and s != "PL-Lipowa"]}
for scheme, key in [("TEB", "teb_dtsa"), ("CLMU5", "clmu_dtsa")]:
    sl[scheme] = {v: slope_ci(ss, key) for v, ss in VAR.items()}
    sl[scheme]["city_equal_slope_all"] = city_equal_slope(sites, key)
out["compression_slope_inference"] = sl

# ---- 3. Freedman–Lane cluster permutation ---------------------------------
def fl_perm(ss, n=10000):
    obs = np.array([P[s]["dtsa_std"] for s in ss])
    alb = np.array([P[s]["albedo"] for s in ss])
    eps = np.array([eps_sky[s] for s in ss])
    Z = np.column_stack([eps, np.ones(len(ss))])
    ra = alb - Z @ np.linalg.lstsq(Z, alb, rcond=None)[0]
    rd = obs - Z @ np.linalg.lstsq(Z, obs, rcond=None)[0]
    r_obs = pearsonr(ra, rd)[0]
    _, cl = city_folds(ss)
    uniq = sorted(set(cl))
    rac = {u: np.mean([ra[i] for i, c in enumerate(cl) if c == u]) for u in uniq}
    vals = list(rac.values())
    cnt = 0
    for _ in range(n):
        pm = dict(zip(uniq, rng.permutation(vals)))
        rap = np.array([pm[c] for c in cl])
        if abs(pearsonr(rap, rd)[0]) >= abs(r_obs):
            cnt += 1
    return {"n": len(ss), "partial_r": float(r_obs),
            "freedman_lane_cluster_p": (cnt + 1) / (n + 1)}

out["albedo_given_skyemissivity_freedman_lane"] = {
    "all19": fl_perm(sites),
    "excl_mpls17": fl_perm([s for s in sites
                            if not s.startswith("US-Minneapolis")])}

d["review4_additions"] = out
json.dump(d, open(STATS, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("=== unified LOCO (dtsa target, K) ===")
for v, row in uni.items():
    for sch in ("TEB", "CLMU5"):
        r = row[sch]
        print(f"  {v:12s} {sch:6s} raw={r['raw_mae_K']:.3f} "
              f"int={r['intercept_only_locomae_K']:.3f} "
              f"alb={r['albedo_locomae_K']:.3f} R2={r['r2_loco']:+.3f} "
              f"dMAE CI={np.round(r['mae_improvement_ci95_cluster'],3)}")
    print(f"  {v:12s} no-model albedo climatology = "
          f"{row['obs_albedo_climatology']['locomae_K']:.3f} K")
print("=== compression slopes ===")
for sch, rows in sl.items():
    for v in VAR:
        r = rows[v]
        print(f"  {sch:6s} {v:12s} slope={r['slope']:.3f} "
              f"CI={np.round(r['ci95_cluster'],3)} excl_unity={r['excludes_unity']}")
    print(f"  {sch:6s} city-equal slope (all) = {rows['city_equal_slope_all']:.3f}")
print("=== Freedman–Lane (albedo | eps_sky) ===")
print(json.dumps(out["albedo_given_skyemissivity_freedman_lane"], indent=1))
