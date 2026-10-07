# -*- coding: utf-8 -*-
"""Evaluate TEB campaign runs on the host (numpy/xarray available here).

(0) VALIDATE: fresh baseline reproduction (_validate_AU-Preston/output/LWU.txt)
    vs the preserved external/teb_runs/AU-Preston/output/LWU_base.txt.
(A) albedo-constrained runs: external/teb_albedo19/<SITE>/LWU_alb.txt
(B) spinup runs (if present):  external/teb_spinup10/<SITE>/LWU_spin.txt
    (record-period portion only)

Alignment convention (frozen, from kg1_radiometric.py / paper_stats.py):
  LWU row i  ->  corpus forcing step i+1 ; nocturnal mask =
  night_mask & ~pre_spinup_flag ; obs_LWup from corpus ;
  dTsa via Ts=((LWup-(1-eps)LWdown)/(eps*sigma))^0.25 - Tair, eps=0.95.
Writes external/teb_campaign_results.json (does NOT modify paper_stats_v1.json).
"""
import io, json, sys
from pathlib import Path
import numpy as np
import xarray as xr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
SIG, EPS = 5.67e-8, 0.95
K_PER_WM2 = 1.0 / (4 * EPS * SIG * 285.0 ** 3)

stats = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
P = stats["per_site"]
SITES = sorted(P)


def corpus(site):
    ds = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
    night = ds["night_mask"].values.astype(bool)
    spin = ds["pre_spinup_flag"].values.astype(bool)
    olw = ds["obs_LWup"].values
    ta = ds["forcing_Tair"].values
    ld = ds["forcing_LWdown"].values
    ds.close()
    m = night & ~spin & np.isfinite(olw) & np.isfinite(ta) & np.isfinite(ld)
    return olw, ta, ld, m


def eval_lwup(model_lwup_full, site, tail=None):
    """model_lwup_full: driver LWU column. tail: if given, use last `tail`+1
    rows (spinup case) so row i -> corpus step i+1 over the record period."""
    olw, ta, ld, m = corpus(site)
    n = len(olw)
    lw = np.asarray(model_lwup_full, float)
    if tail is not None:
        lw = lw[-(n - 1):]                  # record-period portion: last n-1 rows
    # align: model row i -> corpus step i+1  => model[0:n-1] matches corpus[1:n]
    if len(lw) < n - 1:
        return None
    mm = m[1:n]
    model = lw[0:n - 1][mm]
    obs = olw[1:n][mm]
    ld_ = ld[1:n][mm]
    ta_ = ta[1:n][mm]
    bias = float(np.mean(model - obs))
    ts = ((model - (1 - EPS) * ld_) / (EPS * SIG)) ** 0.25
    dtsa = float(np.mean(ts - ta_))
    return {"bias_Wm2": bias, "dtsa_K": dtsa, "n": int(mm.sum())}


out = {"meta": {"K_per_Wm2": K_PER_WM2, "eps": EPS,
                "alignment": "model row i -> corpus step i+1"}}

# (0) validation
vw = ROOT / "external/teb_albedo19/_validate_AU-Preston/output/LWU.txt"
base = ROOT / "external/teb_runs/AU-Preston/output/LWU_base.txt"
if vw.exists() and base.exists():
    a = np.loadtxt(vw); b = np.loadtxt(base)
    d = float(np.max(np.abs(a - b))) if len(a) == len(b) else None
    out["validation"] = {"max_abs_diff_Wm2": d, "n": len(a),
                         "verdict": ("EXACT" if d is not None and d < 1e-6
                                     else "SOFT" if d is not None and d < 1e-2
                                     else "MISMATCH")}
    print(f"[validate] AU-Preston baseline reproduction max|diff|={d} -> {out['validation']['verdict']}")

# (A) albedo-constrained
A = {}
for s in SITES:
    f = ROOT / f"external/teb_albedo19/{s}/LWU_alb.txt"
    if not f.exists():
        A[s] = {"status": "MISSING"}; continue
    r = eval_lwup(np.loadtxt(f), s)
    if r is None:
        A[s] = {"status": "LEN_SHORT"}; continue
    r["status"] = "OK"
    r["baseline_bias_Wm2"] = P[s]["teb_lwup_bias"]
    r["baseline_dtsa_K"] = P[s]["teb_dtsa"]
    r["obs_dtsa_K"] = P[s]["dtsa_std"]
    r["albedo_obs"] = P[s]["albedo"]
    A[s] = r
out["campaign_A_albedo"] = A

# (B) spinup — only sites where metforcing tail == corpus (identical
#     half-hourly record window); the 4 hourly-metforcing sites cannot be
#     cleanly compared to the half-hourly baseline and are EXCLUDED.
def tail_matches(site):
    mf = xr.open_dataset(ROOT / f"data/urban-plumber/FullCollection/{site}/timeseries/{site}_metforcing_v1.nc").squeeze(drop=True)
    co = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
    n = co.sizes["time"]; cta = co["forcing_Tair"].values
    mta = mf["Tair"].values.astype(float); mf.close(); co.close()
    return len(mta) >= n and float(np.nanmax(np.abs(mta[-n:] - cta))) < 1e-6

B = {}
anyB = False
for s in SITES:
    f = ROOT / f"external/teb_spinup10/{s}/LWU_spin.txt"
    if not f.exists():
        continue
    anyB = True
    if not tail_matches(s):
        B[s] = {"status": "EXCLUDED_RESOLUTION_MISMATCH"}; continue
    r = eval_lwup(np.loadtxt(f), s, tail=True)
    if r is None:
        B[s] = {"status": "LEN_SHORT"}; continue
    r["status"] = "OK"; r["baseline_bias_Wm2"] = P[s]["teb_lwup_bias"]
    r["baseline_dtsa_K"] = P[s]["teb_dtsa"]
    r["obs_dtsa_K"] = P[s]["dtsa_std"]
    B[s] = r
if anyB:
    okB = [s for s in SITES if B.get(s, {}).get("status") == "OK"]
    if okB:
        from scipy.stats import pearsonr as _pr
        obsB = np.array([P[s]["dtsa_std"] for s in okB])
        baseB = np.array([P[s]["teb_dtsa"] for s in okB])
        spinB = np.array([B[s]["dtsa_K"] for s in okB])
        out["campaign_B_summary"] = {
            "n_clean": len(okB),
            "n_excluded": sum(1 for s in B if B[s]["status"].startswith("EXCLUDED")),
            "excluded_sites": [s for s in B if B[s]["status"].startswith("EXCLUDED")],
            "max_abs_dtsa_change_K": float(np.max(np.abs(spinB - baseB))),
            "mean_abs_dtsa_change_K": float(np.mean(np.abs(spinB - baseB))),
            "max_abs_bias_change_Wm2": float(np.max(np.abs(
                [B[s]["bias_Wm2"] - P[s]["teb_lwup_bias"] for s in okB]))),
            "baseline_slope": float(np.polyfit(obsB, baseB, 1)[0]),
            "spinup_slope": float(np.polyfit(obsB, spinB, 1)[0]),
        }
        print("\n=== Campaign B spin-up (clean 15 sites) ===")
        cs = out["campaign_B_summary"]
        print(f"  excluded (hourly metforcing): {cs['excluded_sites']}")
        print(f"  max|dtsa change|={cs['max_abs_dtsa_change_K']:.4f} K  "
              f"mean={cs['mean_abs_dtsa_change_K']:.4f} K  "
              f"max|bias change|={cs['max_abs_bias_change_Wm2']:.3f} W/m2")
        print(f"  baseline slope {cs['baseline_slope']:.3f} -> spinup slope {cs['spinup_slope']:.3f}")
    out["campaign_B_spinup"] = B

# cross-site summaries for A
okA = [s for s in SITES if A.get(s, {}).get("status") == "OK"]
if okA:
    from scipy.stats import pearsonr
    obs = np.array([P[s]["dtsa_std"] for s in okA])
    mod_base = np.array([P[s]["teb_dtsa"] for s in okA])
    mod_alb = np.array([A[s]["dtsa_K"] for s in okA])
    def slope_r(x, y):
        b = float(np.polyfit(x, y, 1)[0]); r = float(pearsonr(x, y)[0]); return b, r
    sb, rb = slope_r(obs, mod_base); sa, ra = slope_r(obs, mod_alb)
    out["campaign_A_summary"] = {
        "n": len(okA),
        "baseline_slope_model_on_obs": sb, "baseline_r": rb,
        "albedo_slope_model_on_obs": sa, "albedo_r": ra,
        "baseline_model_span_K": [float(mod_base.min()), float(mod_base.max())],
        "albedo_model_span_K": [float(mod_alb.min()), float(mod_alb.max())],
        "obs_span_K": [float(obs.min()), float(obs.max())],
        "mean_abs_bias_baseline_Wm2": float(np.mean(np.abs(
            [P[s]["teb_lwup_bias"] for s in okA]))),
        "mean_abs_bias_albedo_Wm2": float(np.mean(np.abs(
            [A[s]["bias_Wm2"] for s in okA]))),
    }
    print("\n=== Campaign A cross-site (model dTsa on observed) ===")
    print(f"  baseline: slope {sb:.3f} r {rb:.3f} span [{mod_base.min():.2f},{mod_base.max():.2f}]")
    print(f"  albedo  : slope {sa:.3f} r {ra:.3f} span [{mod_alb.min():.2f},{mod_alb.max():.2f}]")
    print(f"  observed span [{obs.min():.2f},{obs.max():.2f}]")
    print("  per-site (obs / base_dtsa / alb_dtsa):")
    for s in okA:
        print(f"    {s:16s} {P[s]['dtsa_std']:+.2f} / {P[s]['teb_dtsa']:+.2f} / {A[s]['dtsa_K']:+.2f}")

json.dump(out, open(ROOT / "external/teb_campaign_results.json", "w",
                    encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nwrote external/teb_campaign_results.json")
