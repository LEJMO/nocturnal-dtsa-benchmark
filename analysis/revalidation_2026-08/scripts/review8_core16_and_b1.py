# -*- coding: utf-8 -*-
"""review8 (v27, supervisor review 2026-09-23): statistics the core-16-first
rewrite prints that were not yet in the package. READ-ONLY on every existing
result; no evaluation code is modified. Output: results/review8_core16_and_b1.json
(merged into paper_stats_v1.json by review5_merge.py as review8_additions).

Blocks
  core16_metric_families : medians of the per-record nocturnal LW_up scores
                           (r, d, cRMSE, MAE, MBE), record-mean and
                           sample-weighted LW_up bias, warm counts, benchmark
                           win counts (leave-one-city-cluster-out, eval-only)
                           and the own-bias-removed diagnostic counts, for the
                           three record sets. Source arrays: review5 per-record
                           skill, city-holdout per-record MAE, review7 debiased
                           per-record MAE, per_site biases.
  b1_albedo_skyemissivity: the supervisor's B-1 checks: (a) albedo-only,
                           sky-emissivity-only and two-descriptor models of the
                           record-mean offset under leave-one-city-cluster-out,
                           full-sample coefficients; (b) the albedo relation
                           inside the Cfb climate class with an exact
                           permutation p (720 arrangements); within-class r
                           for every class with >= 3 records.
  material_sensitivity_K : per-site, per-perturbation change of the nocturnal
                           LW_up bias relative to the default run (W m-2) and
                           its kelvin equivalent (0.2005 K per W m-2), TEB from
                           probe_rows.csv and CLM-Urban from the clmu_probe
                           run summaries.
"""
import io, sys, json, csv, glob, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
S = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
A = S["review5_additions"]; PS = S["per_site"]
SITES = sorted(PS)
HM = {"US-Minneapolis1", "US-Minneapolis2"}
SETS = {"all19": SITES,
        "excl_mpls17": [s for s in SITES if s not in HM],
        "core16": [s for s in SITES if s not in HM and s != "PL-Lipowa"]}
K_PER_WM2 = S["review2_additions"]["K_per_Wm2_conversion"]["K_per_Wm2"] if isinstance(S["review2_additions"].get("K_per_Wm2_conversion"), dict) and "K_per_Wm2" in S["review2_additions"]["K_per_Wm2_conversion"] else 0.2004927923369955
CL = {"US-Minneapolis1": "Minneapolis", "US-Minneapolis2": "Minneapolis", "PL-Lipowa": "Lodz",
      "PL-Narutowicza": "Lodz", "FI-Kumpula": "Helsinki", "FI-Torni": "Helsinki"}
out = {"meta": {"script": "review8_core16_and_b1.py", "date": "2026-09-26",
                "purpose": "core-16-first rewrite (v27) and supervisor B-1 checks; read-only on prior results",
                "K_per_Wm2": K_PER_WM2}}

# ------------------------------------------------------------- metric families
PRS = A["nocturnal_benchmark_skill"]["per_record_skill"]
CH = A["benchmark_city_holdout"]["city_holdout"]
DEB = A["benchmark_debiased_mae"]["schemes"]
DECs = A["benchmark_error_decomposition"]["schemes"]
mf = {}
for setname, ss in SETS.items():
    mf[setname] = {"n": len(ss)}
    for k, biaskey in (("TEB", "teb_lwup_bias"), ("CLMU5", "clmu_lwup_bias")):
        med = lambda m: float(np.median([PRS[k][s][m] for s in ss]))
        n_night = np.array([PS[s]["n_night"] for s in ss], float)
        b = np.array([PS[s][biaskey] for s in ss], float)
        pm = CH["per_record_mae"][k]
        wins = CH["win_counts_eval_only"][k][setname]
        assert wins["n"] == len(ss)
        # own-bias-removed diagnostic counts inside the set (review7 per-record flags)
        d = DEB[k]["per_record"]
        deb_km3 = sum(1 for s in ss if d[s]["beats_KM3_debiased_city"])
        deb_reg2 = sum(1 for s in ss if d[s]["beats_REG2_debiased_city"])
        # bias share of MSE per record: MBE^2 / (MBE^2 + cRMSE^2)
        share = [PRS[k][s]["MBE"] ** 2 / (PRS[k][s]["MBE"] ** 2 + PRS[k][s]["cRMSE"] ** 2) for s in ss]
        mf[setname][k] = {
            "median_r": med("r"), "median_d": med("d"), "median_cRMSE_Wm2": med("cRMSE"),
            "median_RMSE_Wm2": med("RMSE"), "median_MAE_Wm2": float(np.median([pm[s]["scheme_MAE"] for s in ss])),
            "record_mean_bias_Wm2": float(b.mean()), "sample_weighted_bias_Wm2": float((b * n_night).sum() / n_night.sum()),
            "warm_records": int((b > 0).sum()),
            "median_bias_share_of_MSE": float(np.median(share)),
            "beats_REG2": wins["beats_REG2"], "beats_KM3": wins["beats_KM3"], "beats_both": wins["beats_meaningful_two_REG2_KM3"],
            "debiased_beats_REG2": deb_reg2, "debiased_beats_KM3": deb_km3,
        }
# consistency with the published all-19 numbers (Table 5 of v26)
assert abs(mf["all19"]["TEB"]["record_mean_bias_Wm2"] - 9.2) < 0.05 and abs(mf["all19"]["CLMU5"]["record_mean_bias_Wm2"] - (-2.9)) < 0.05
assert mf["all19"]["TEB"]["beats_both"] == 3 and mf["all19"]["CLMU5"]["beats_both"] == 13
assert mf["all19"]["TEB"]["debiased_beats_KM3"] == 19 and mf["all19"]["CLMU5"]["debiased_beats_REG2"] == 18
assert round(mf["all19"]["TEB"]["median_bias_share_of_MSE"], 2) == round(DECs["TEB"]["bias_share_of_MSE"]["median"], 2)
out["core16_metric_families"] = mf

# ------------------------------------------------------------- B-1
EPS = S["review3_additions"]["forcing_covariates"]["covariates"]["sky_emissivity"]["per_site_mean"]
def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X]); return np.linalg.lstsq(X1, y, rcond=None)[0]
def loco(X, y, cl):
    pred = np.empty_like(y)
    for c in np.unique(cl):
        tr = cl != c; b = ols(X[tr], y[tr]); pred[~tr] = np.column_stack([np.ones((~tr).sum()), X[~tr]]) @ b
    res = y - pred; return float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()), float(np.abs(res).mean())
b1 = {"definition": "record-mean nocturnal offset regressed on observed midday albedo and/or mean nocturnal sky emissivity; leave-one-city-cluster-out predictions; R2_LOCO = 1 - SSres/SStot about the full-sample mean; MAE of held-out residuals (K)", "sets": {}}
for setname, ss in SETS.items():
    alb = np.array([PS[s]["albedo"] for s in ss]); y = np.array([PS[s]["dtsa_std"] for s in ss])
    eps = np.array([EPS[s] for s in ss]); cl = np.array([CL.get(s, s) for s in ss])
    blk = {"n": len(ss), "n_clusters": int(len(np.unique(cl))),
           "r_albedo_offset": float(np.corrcoef(alb, y)[0, 1]), "r_skyemis_offset": float(np.corrcoef(eps, y)[0, 1]),
           "r_albedo_skyemis": float(np.corrcoef(alb, eps)[0, 1])}
    for lab, X in (("albedo_only", alb[:, None]), ("skyemis_only", eps[:, None]), ("albedo_plus_skyemis", np.column_stack([alb, eps]))):
        coef = ols(X, y); r2, mae = loco(X, y, cl)
        blk[lab] = {"coef_K_per_unit": [float(c) for c in coef[1:]], "intercept_K": float(coef[0]), "R2_LOCO": r2, "LOCO_MAE_K": mae}
    b1["sets"][setname] = blk
# Cfb subset with exact permutation
cfb = [s for s in SITES if PS[s]["koppen"] == "Cfb"]
alb = np.array([PS[s]["albedo"] for s in cfb]); y = np.array([PS[s]["dtsa_std"] for s in cfb])
r = float(np.corrcoef(alb, y)[0, 1]); slope = float(ols(alb[:, None], y)[1])
perm = np.array([np.corrcoef(alb, np.array(pp))[0, 1] for pp in itertools.permutations(y)])
b1["cfb_subset"] = {"sites": cfb, "n": len(cfb), "r": r, "slope_K_per_0p1_albedo": slope / 10,
                    "exact_permutation_two_sided_p": float((np.abs(perm) >= abs(r) - 1e-12).mean()),
                    "exact_permutation_one_sided_p": float((perm <= r + 1e-12).mean()), "n_arrangements": int(len(perm))}
wc = {}
for kclass in sorted(set(PS[s]["koppen"] for s in SITES)):
    ss = [s for s in SITES if PS[s]["koppen"] == kclass]
    if len(ss) >= 3:
        a_ = np.array([PS[s]["albedo"] for s in ss]); y_ = np.array([PS[s]["dtsa_std"] for s in ss])
        wc[kclass] = {"sites": ss, "n": len(ss), "r": float(np.corrcoef(a_, y_)[0, 1])}
b1["within_class_r"] = wc
out["b1_albedo_skyemissivity"] = b1

# ------------------------------------------------------------- material sensitivity in K
mat = {"definition": "change of the record-mean nocturnal LW_up bias relative to the default configuration (perturbed minus default, W m-2) and its kelvin equivalent at K_per_Wm2; TEB configurations: TI (thermal inertia = heat capacity) x0.5/x2, TC (conductivity) x0.5/x2, ALB_site (observed albedo on every facet), EMIS_low (emissivity reduced), ROOF_light (lightweight roof archetype); CLM-Urban: CV (heat capacity) x0.5/x2, TK (conductivity) x0.5/x2, ALB_site, EMIS_low",
       "TEB": {}, "CLMU5": {}}
rows = list(csv.DictReader(io.open(ROOT / "analysis/revalidation_2026-08/evidence/probe_rows.csv", encoding="utf-8")))
for r_ in rows:
    site, cfg = r_["site"], r_["config"]
    mat["TEB"].setdefault(site, {})[cfg] = {"lwup_bias_Wm2": float(r_["lwup_bias"]), "delta_Wm2": float(r_["dlwup_vs_base"]),
                                           "delta_K": float(r_["dlwup_vs_base"]) * K_PER_WM2}
for f in sorted(glob.glob(str(ROOT / "external/clmu_probe/*/*.json"))):
    d = json.load(open(f, encoding="utf-8")); site, cfg = d["site"], d["config"]
    mat["CLMU5"].setdefault(site, {})[cfg] = {"lwup_bias_Wm2": float(d["lwup_bias"])}
for site, cfgs in mat["CLMU5"].items():
    base = cfgs["base"]["lwup_bias_Wm2"]
    for cfg, v in cfgs.items():
        v["delta_Wm2"] = v["lwup_bias_Wm2"] - base; v["delta_K"] = v["delta_Wm2"] * K_PER_WM2
# cross-check against the published envelopes
for k, key in (("TEB", "teb_material_envelope"), ("CLMU5", "clmu_material_envelope")):
    env = S["elimination"][key]["per_site"]
    for site, e in env.items():
        vals = [v["lwup_bias_Wm2"] for v in mat[k][site].values()]
        assert abs(min(vals) - e["vmin"]) < 0.02 and abs(max(vals) - e["vmax"]) < 0.02, (k, site, min(vals), e["vmin"], max(vals), e["vmax"])
        assert abs(mat[k][site]["base"]["lwup_bias_Wm2"] - e["base"]) < 0.02
out["material_sensitivity_K"] = mat

p = ROOT / "results/review8_core16_and_b1.json"
io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1, ensure_ascii=False))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("wrote", p)
    for setname in SETS:
        for k in ("TEB", "CLMU5"):
            m = mf[setname][k]
            print(f"  {setname:11s} {k:5s} r={m['median_r']:.3f} d={m['median_d']:.2f} cRMSE={m['median_cRMSE_Wm2']:.1f} MAE={m['median_MAE_Wm2']:.1f} bias={m['record_mean_bias_Wm2']:+.1f} ({m['sample_weighted_bias_Wm2']:+.1f} wt) warm={m['warm_records']} share={m['median_bias_share_of_MSE']:.2f} wins REG2/KM3/both={m['beats_REG2']}/{m['beats_KM3']}/{m['beats_both']} debiased REG2/KM3={m['debiased_beats_REG2']}/{m['debiased_beats_KM3']}")
    c = b1["cfb_subset"]; print(f"  Cfb n={c['n']} r={c['r']:+.2f} slope={c['slope_K_per_0p1_albedo']:+.2f} p2={c['exact_permutation_two_sided_p']:.3f}")
    for setname in SETS:
        b = b1["sets"][setname]; print(f"  B1 {setname}: alb {b['albedo_only']['LOCO_MAE_K']:.2f} K | eps {b['skyemis_only']['LOCO_MAE_K']:.2f} | both {b['albedo_plus_skyemis']['LOCO_MAE_K']:.2f} coef {b['albedo_plus_skyemis']['coef_K_per_unit']}")
    for k in ("TEB", "CLMU5"):
        for site, cfgs in mat[k].items():
            print(f"  {k} {site}: " + ", ".join(f"{c}={v['delta_K']:+.2f}K" for c, v in cfgs.items() if c != "base"))
