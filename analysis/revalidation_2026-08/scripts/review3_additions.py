# -*- coding: utf-8 -*-
"""Round-3 (external review, path A) additions — 2026-09-08.

Appends section 'review3_additions' to results/paper_stats_v1.json:
  1. pooled-bias conventions, both defined explicitly (record-equal / n-weighted)
     — fixes the mixed +6.8/−2.8 reporting caught by the external review.
  2. nocturnal forcing covariates per site (sky emissivity, wind, Tair):
     Spearman vs dTsa + partial r(dTsa, albedo | X) — addresses "forcing/climate
     not separated" (external Major 1).
  3. no-burn-in sensitivity: dTsa with the full record (night mask only, spin-up
     exclusion lifted) — addresses "observation burn-in unjustified" (Major 3).
  4. envelope-vs-own-bias fractions at the probe sites — replaces the
     envelope-vs-inter-scheme-gap discriminant (Major 5).
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import pearsonr, spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]
STATS = ROOT / "results" / "paper_stats_v1.json"
SIG, EPS = 5.67e-8, 0.95

d = json.load(open(STATS, encoding="utf-8"))
P = d["per_site"]
sites = sorted(P)
dtsa = np.array([P[s]["dtsa_std"] for s in sites])
alb = np.array([P[s]["albedo"] for s in sites])

out = {"meta": {"script": "analysis/revalidation_2026-08/scripts/review3_additions.py",
                "date": "2026-09-08"}}

# ---- 1. pooled-bias conventions -------------------------------------------
teb = np.array([P[s]["teb_lwup_bias"] for s in sites])
clm = np.array([P[s]["clmu_lwup_bias"] for s in sites])
n = np.array([P[s]["n_night"] for s in sites], dtype=float)
out["bias_aggregates_Wm2"] = {
    "record_equal_mean": {"TEB": teb.mean(), "CLMU5": clm.mean()},
    "sample_weighted_mean": {"TEB": float((teb * n).sum() / n.sum()),
                             "CLMU5": float((clm * n).sum() / n.sum())},
    "note": ("Manuscript v3 mixed conventions (+6.8 sample-weighted TEB with "
             "-2.8 record-equal CLMU5); v4 reports record-equal means with the "
             "sample-weighted values in parentheses."),
}

# ---- 2 & 3. corpus-derived: forcing covariates + no-burn-in dTsa -----------
cov_vals = {"sky_emissivity": [], "wind": [], "tair": []}
dtsa_noburn = []
for s in sites:
    ds = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{s}.nc")
    night = ds["night_mask"].values.astype(bool)
    spin = ds["pre_spinup_flag"].values.astype(bool)
    m_std = night & ~spin          # analysis mask (as in the paper)
    m_all = night                  # burn-in lifted
    ta = ds["forcing_Tair"].values
    ld = ds["forcing_LWdown"].values
    lw = ds["obs_LWup"].values
    wn = np.hypot(ds["forcing_Wind_N"].values, ds["forcing_Wind_E"].values)
    ds.close()
    ok_std = m_std & np.isfinite(lw) & np.isfinite(ta) & np.isfinite(ld)
    ok_all = m_all & np.isfinite(lw) & np.isfinite(ta) & np.isfinite(ld)
    eps_sky = ld[ok_std] / (SIG * ta[ok_std] ** 4)
    cov_vals["sky_emissivity"].append(float(np.mean(eps_sky)))
    cov_vals["wind"].append(float(np.mean(wn[ok_std])))
    cov_vals["tair"].append(float(np.mean(ta[ok_std])))
    ts_all = ((lw[ok_all] - (1 - EPS) * ld[ok_all]) / (EPS * SIG)) ** 0.25
    dtsa_noburn.append(float(np.mean(ts_all - ta[ok_all])))

covs = {}
for name, xv in cov_vals.items():
    x = np.array(xv)
    r, p = pearsonr(x, dtsa)
    rs, ps = spearmanr(x, dtsa)
    X = np.column_stack([x, np.ones(len(x))])
    ra = alb - X @ np.linalg.lstsq(X, alb, rcond=None)[0]
    rd = dtsa - X @ np.linalg.lstsq(X, dtsa, rcond=None)[0]
    pr, pp = pearsonr(ra, rd)
    covs[name] = {"per_site_mean": dict(zip(sites, [float(v) for v in xv])),
                  "pearson_r": r, "pearson_p": p,
                  "spearman_rho": rs, "spearman_p": ps,
                  "partial_r_dtsa_albedo_given_this": pr, "partial_p": pp}
out["forcing_covariates"] = {
    "definition": ("Per-site means over the standard nocturnal analysis mask; "
                   "sky emissivity = LWdown/(sigma*Tair^4); wind = |(N,E)| "
                   "forcing wind speed."),
    "covariates": covs,
}

nb = np.array(dtsa_noburn)
r_nb, p_nb = pearsonr(alb, nb)
mask17 = np.array([not s.startswith("US-Minneapolis") for s in sites])
r_nb17, p_nb17 = pearsonr(alb[mask17], nb[mask17])
out["no_burnin_sensitivity"] = {
    "dtsa_no_burnin": dict(zip(sites, [float(v) for v in nb])),
    "max_abs_shift_K": float(np.max(np.abs(nb - dtsa))),
    "mean_abs_shift_K": float(np.mean(np.abs(nb - dtsa))),
    "r_albedo_all19": r_nb, "p_all19": p_nb,
    "r_albedo_excl_mpls17": r_nb17, "p_excl_mpls17": p_nb17,
    "note": ("Burn-in retained in the paper for identical obs/model sampling "
             "masks; this sensitivity lifts it for the observational statistics."),
}

# ---- 4. envelope vs own bias at probe sites --------------------------------
elim = d["elimination"]
rows = []
for scheme, key in [("TEB", "teb_material_envelope"),
                    ("CLMU5", "clmu_material_envelope")]:
    for s, v in elim[key]["per_site"].items():
        base = v["base"]
        best = min(abs(v["vmin"]), abs(v["vmax"]))  # valid: no sign reversal
        rows.append({"scheme": scheme, "site": s, "base_bias_Wm2": base,
                     "best_perturbed_abs_bias_Wm2": best,
                     "fraction_of_bias_remaining": best / abs(base),
                     "sign_reversed": False})
out["probe_envelope_vs_own_bias"] = {
    "rows": rows,
    "min_fraction_remaining_extreme_records": min(
        r["fraction_of_bias_remaining"] for r in rows
        if r["site"] in ("US-Minneapolis2", "KR-Ochang", "US-WestPhoenix",
                         "PL-Lipowa", "AU-Preston")),
    "note": ("Discriminant per external review: compare the one-at-a-time "
             "envelope against each scheme's own site bias, not the "
             "inter-scheme gap."),
}

d["review3_additions"] = out
json.dump(d, open(STATS, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("=== pooled conventions ===")
print(json.dumps(out["bias_aggregates_Wm2"], indent=1)[:400])
print("=== forcing covariates ===")
for k, v in covs.items():
    print(f"  {k:16s} rho={v['spearman_rho']:+.3f} (p={v['spearman_p']:.4f}) "
          f"partial r={v['partial_r_dtsa_albedo_given_this']:+.3f} (p={v['partial_p']:.4f})")
print("=== no-burn-in ===")
print(f"  max|shift|={out['no_burnin_sensitivity']['max_abs_shift_K']:.3f} K "
      f"mean={out['no_burnin_sensitivity']['mean_abs_shift_K']:.3f} K "
      f"r19={r_nb:+.3f} r17={r_nb17:+.3f}")
print("=== envelope vs own bias ===")
for r0 in rows:
    print(f"  {r0['scheme']:5s} {r0['site']:16s} base={r0['base_bias_Wm2']:+7.2f} "
          f"best={r0['best_perturbed_abs_bias_Wm2']:6.2f} "
          f"frac={r0['fraction_of_bias_remaining']:.2f}")
