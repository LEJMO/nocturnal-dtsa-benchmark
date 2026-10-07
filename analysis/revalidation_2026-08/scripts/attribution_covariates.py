# -*- coding: utf-8 -*-
"""Attribution stress test (peer-review action 5, 2026-09-05).

Computes rank/parametric correlations of the observed nocturnal offset
(dtsa_std, from results/paper_stats_v1.json per_site) against four candidate
morphological descriptors from the Urban-PLUMBER harmonised site metadata:

  - canyon_height_width_ratio          (SVF proxy; Oke 1981 direction test)
  - vegetated fraction = tree + grass  (tree_area_fraction + grass_area_fraction)
  - pervious fraction                  (per_site.pervious = 1 - impervious)
  - building_mean_height

and, for each, the partial correlation of the offset with observed midday
albedo conditioning on that descriptor. Appends the results as section
'attribution_covariates' to results/paper_stats_v1.json (idempotent).

Manuscript claims backed by this section (Sect. 2.5 / 3.1 / 5):
  - aspect ratio rho=+0.56 (p=0.014); pervious rho=-0.54 (p=0.016);
    vegetated rho=-0.43 (p=0.065); height rho=+0.43 (p=0.066)
  - partial r(dtsa, albedo | X) in [-0.77, -0.73], all parametric p<=4e-4
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
STATS = ROOT / "results" / "paper_stats_v1.json"
FULLCOL = ROOT / "data" / "urban-plumber" / "FullCollection"

d = json.load(open(STATS, encoding="utf-8"))
P = d["per_site"]
sites = sorted(P)


def sitedata(site: str, key: str) -> float:
    p = FULLCOL / site / f"{site}_sitedata_v1.csv"
    for row in csv.DictReader(p.open(encoding="utf-8")):
        if row["parameter"] == key:
            try:
                return float(row["value"])
            except ValueError:
                return float("nan")
    return float("nan")


dtsa = np.array([P[s]["dtsa_std"] for s in sites])
alb = np.array([P[s]["albedo"] for s in sites])
covs = {
    "canyon_height_width_ratio": np.array(
        [sitedata(s, "canyon_height_width_ratio") for s in sites]
    ),
    "vegetated_fraction_tree_plus_grass": np.array(
        [sitedata(s, "tree_area_fraction") + sitedata(s, "grass_area_fraction")
         for s in sites]
    ),
    "pervious_fraction": np.array([P[s]["pervious"] for s in sites]),
    "building_mean_height": np.array(
        [sitedata(s, "building_mean_height") for s in sites]
    ),
}

out = {
    "meta": {
        "script": "analysis/revalidation_2026-08/scripts/attribution_covariates.py",
        "date": "2026-09-05",
        "n_sites": len(sites),
        "note": ("Exploratory family, unadjusted p-values; albedo is the "
                 "pre-specified primary descriptor. Partial r conditions the "
                 "dtsa~albedo relation on each covariate via OLS residuals."),
    },
    "albedo_primary": {},
    "covariates": {},
}

r, p = pearsonr(alb, dtsa)
rs, ps = spearmanr(alb, dtsa)
out["albedo_primary"] = {
    "pearson_r": r, "pearson_p": p, "spearman_rho": rs, "spearman_p": ps,
}

for name, x in covs.items():
    ok = np.isfinite(x)
    r, p = pearsonr(x[ok], dtsa[ok])
    rs, ps = spearmanr(x[ok], dtsa[ok])
    # partial: residualize both albedo and dtsa on this covariate (+ intercept)
    X = np.column_stack([x[ok], np.ones(ok.sum())])
    res_alb = alb[ok] - X @ np.linalg.lstsq(X, alb[ok], rcond=None)[0]
    res_dts = dtsa[ok] - X @ np.linalg.lstsq(X, dtsa[ok], rcond=None)[0]
    pr, pp = pearsonr(res_alb, res_dts)
    out["covariates"][name] = {
        "n": int(ok.sum()),
        "pearson_r": r, "pearson_p": p,
        "spearman_rho": rs, "spearman_p": ps,
        "partial_r_dtsa_albedo_given_this": pr,
        "partial_p": pp,
    }

d["attribution_covariates"] = out
json.dump(d, open(STATS, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("appended attribution_covariates to", STATS.name)
for name, v in out["covariates"].items():
    print(f"  {name:38s} rho={v['spearman_rho']:+.3f} (p={v['spearman_p']:.4f})"
          f"  partial r={v['partial_r_dtsa_albedo_given_this']:+.3f}"
          f" (p={v['partial_p']:.4f})")
