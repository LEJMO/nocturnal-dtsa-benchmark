# -*- coding: utf-8 -*-
"""Round-2 peer-review additions (2026-09-07).

Computes and appends to results/paper_stats_v1.json (section 'review2_additions'):
  1. Cross-scheme bias decorrelation r(TEB, CLMU5) on all 19 / excl-Mpls 17 /
     preregistered 13-site unseen subset, each with a city-cluster bootstrap
     95% CI (5000 draws) and parametric p.  [R1 major #1]
  2. Two extra covariates for the attribution family: T_a sensor height
     (sitedata measurement_height_above_ground) and roof plan fraction
     (sitedata roof_area_fraction): Spearman + partial r(dtsa, albedo | X).
     [R1 major #3, R2 reason-against #2]
  3. Cluster-permutation p for the conservative-core variant
     (excl Mpls + Lipowa), 10000 permutations.  [R1 minor]
  4. Per-probe-site check: material envelope width vs |TEB-CLMU5| site gap.
     [R1 major #2 / R3 overclaim #1 reformulation]
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

rng = np.random.default_rng(20260907)

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


def cluster_boot_r(x, y, clusters, n=5000):
    """Cluster bootstrap 95% CI for Pearson r (resample city clusters)."""
    uniq = sorted(set(clusters))
    idx_by_c = {c: [i for i, cc in enumerate(clusters) if cc == c] for c in uniq}
    rs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = [i for c in pick for i in idx_by_c[c]]
        if len(set(np.asarray(x)[idx])) < 3:
            continue
        rs.append(pearsonr(np.asarray(x)[idx], np.asarray(y)[idx])[0])
    return [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]


def cluster_perm_p(x, y, clusters, n=10000):
    """Permute x across clusters (keep within-cluster ties), two-sided on |r|."""
    uniq = sorted(set(clusters))
    xc = {c: np.mean([x[i] for i, cc in enumerate(clusters) if cc == c]) for c in uniq}
    yv = np.asarray(y, float)
    xv = np.asarray([xc[c] for c in clusters])
    r_obs = pearsonr(xv, yv)[0]
    cnt = 0
    vals = list(xc.values())
    for _ in range(n):
        perm = rng.permutation(vals)
        pm = dict(zip(uniq, perm))
        xp = np.asarray([pm[c] for c in clusters])
        if abs(pearsonr(xp, yv)[0]) >= abs(r_obs):
            cnt += 1
    return float(r_obs), (cnt + 1) / (n + 1)


out = {"meta": {"script": "analysis/revalidation_2026-08/scripts/review2_additions.py",
                "date": "2026-09-07", "seed": 20260907}}

# ---- 1. cross-scheme decorrelation variants -------------------------------
teb = {s: P[s]["teb_lwup_bias"] for s in sites}
clm = {s: P[s]["clmu_lwup_bias"] for s in sites}
clus = {s: P[s]["cluster"] for s in sites}
variants = {
    "all19": sites,
    "excl_mpls17": [s for s in sites if not s.startswith("US-Minneapolis")],
    "unseen13": [s for s in sites if P[s]["in_g2"]],
}
dec = {}
for name, ss in variants.items():
    x = [teb[s] for s in ss]
    y = [clm[s] for s in ss]
    cl = [clus[s] for s in ss]
    r, p = pearsonr(x, y)
    dec[name] = {"n": len(ss), "r": r, "p_parametric": p,
                 "r_ci95_cluster_bootstrap": cluster_boot_r(x, y, cl)}
out["cross_scheme_decorrelation"] = dec
out["unseen13_rule"] = ("Preregistered: the six excluded records are the frozen G1 "
                        "gate evaluation set (PREREGISTRATION_G1.md, git-frozen "
                        "before any cross-scheme comparison): US-Minneapolis1/2, "
                        "KR-Ochang, US-WestPhoenix (warm extremes), PL-Lipowa "
                        "(negative extreme), FR-Capitole (near-zero control); these "
                        "records were used to develop and gate the pipeline, so the "
                        "decorrelation is reported on the 13 development-untouched "
                        "records, with full-sample variants now added.")

# ---- 2. extra covariates ---------------------------------------------------
dtsa = np.array([P[s]["dtsa_std"] for s in sites])
alb = np.array([P[s]["albedo"] for s in sites])
extra = {
    "tair_sensor_height_m": np.array(
        [sitedata(s, "measurement_height_above_ground") for s in sites]),
    "roof_area_fraction": np.array(
        [sitedata(s, "roof_area_fraction") for s in sites]),
}
cov = {}
heights = {}
for name, x in extra.items():
    ok = np.isfinite(x)
    r, p = pearsonr(x[ok], dtsa[ok])
    rs, ps = spearmanr(x[ok], dtsa[ok])
    X = np.column_stack([x[ok], np.ones(int(ok.sum()))])
    ra = alb[ok] - X @ np.linalg.lstsq(X, alb[ok], rcond=None)[0]
    rd = dtsa[ok] - X @ np.linalg.lstsq(X, dtsa[ok], rcond=None)[0]
    pr, pp = pearsonr(ra, rd)
    cov[name] = {"n": int(ok.sum()), "pearson_r": r, "pearson_p": p,
                 "spearman_rho": rs, "spearman_p": ps,
                 "partial_r_dtsa_albedo_given_this": pr, "partial_p": pp}
out["extra_covariates"] = cov
out["tair_sensor_height_per_site"] = {
    s: sitedata(s, "measurement_height_above_ground") for s in sites}

# ---- 3. permutation p for conservative core --------------------------------
core = [s for s in sites
        if not s.startswith("US-Minneapolis") and s != "PL-Lipowa"]
r_c, p_c = cluster_perm_p([P[s]["albedo"] for s in core],
                          [P[s]["dtsa_std"] for s in core],
                          [clus[s] for s in core])
out["core_variant_permutation"] = {
    "n_sites": len(core), "n_clusters": len(set(clus[s] for s in core)),
    "r_cluster": r_c, "p_perm": p_c, "nperm": 10000}

# ---- 4. per-probe-site envelope vs inter-scheme gap ------------------------
elim = d["elimination"]
rows = []
for scheme, key in [("TEB", "teb_material_envelope"),
                    ("CLMU5", "clmu_material_envelope")]:
    for s, v in elim[key]["per_site"].items():
        gap = abs(teb[s] - clm[s])
        rows.append({"scheme": scheme, "site": s, "envelope_Wm2": v["env"],
                     "inter_scheme_gap_Wm2": gap,
                     "envelope_lt_gap": bool(v["env"] < gap)})
out["probe_envelope_vs_gap"] = {
    "rows": rows,
    "all_below": all(r["envelope_lt_gap"] for r in rows),
    "n_below": sum(r["envelope_lt_gap"] for r in rows), "n_total": len(rows)}

# ---- constants --------------------------------------------------------------
SIG = 5.670374419e-8
out["K_per_Wm2_conversion"] = {
    "formula": "dT = dLW / (4*eps*sigma*T^3), eps=0.95, T=285 K",
    "Wm2_per_K": 4 * 0.95 * SIG * 285.0 ** 3, "K_per_Wm2": 1.0 / (4 * 0.95 * SIG * 285.0 ** 3)}
out["clear_calm_definition"] = (
    "Per site, nocturnal samples with sky emissivity eps_sky = LWdown/(sigma*Ta^4) "
    "below the site median AND 10-m wind speed below the site median; "
    "cloudy-windy = both above the site median (conditioning_analysis.py).")

d["review2_additions"] = out
json.dump(d, open(STATS, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("=== cross-scheme decorrelation ===")
for k, v in dec.items():
    print(f"  {k:12s} n={v['n']:2d} r={v['r']:+.3f} (p={v['p_parametric']:.3f}) "
          f"CI95={v['r_ci95_cluster_bootstrap'][0]:+.2f}..{v['r_ci95_cluster_bootstrap'][1]:+.2f}")
print("=== extra covariates ===")
for k, v in cov.items():
    print(f"  {k:24s} rho={v['spearman_rho']:+.3f} (p={v['spearman_p']:.4f}) "
          f"partial r={v['partial_r_dtsa_albedo_given_this']:+.3f} (p={v['partial_p']:.4f})")
print(f"=== core permutation: r={r_c:+.3f} p={p_c:.2e} (n={len(core)}) ===")
print("=== envelope vs gap ===")
for r in rows:
    print(f"  {r['scheme']:5s} {r['site']:16s} env={r['envelope_Wm2']:5.2f} "
          f"gap={r['inter_scheme_gap_Wm2']:5.2f} below={r['envelope_lt_gap']}")
print("all_below:", out["probe_envelope_vs_gap"]["all_below"])
print("Wm2_per_K:", round(out["K_per_Wm2_conversion"]["Wm2_per_K"], 3))
