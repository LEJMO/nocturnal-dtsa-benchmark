# -*- coding: utf-8 -*-
"""Round-5 constrained-experiment regenerator (external review of v10, 2026-09-16).

WHY THIS FILE EXISTS
--------------------
An audit found that 44 of 113 numeric leaves in three stats sections
  - teb_constrained_experiments
  - clmu_constrained_experiments
  - amplitude_ordering_decomposition
had NO generating script anywhere in the repository. Their printed values had
no reproducible source. This file RECOMPUTES every one of those leaves from raw
model output + the corpus, so the manuscript's numbers have a provenance.

CONCURRENCY
-----------
results/paper_stats_v1.json is READ-ONLY here. This script writes ONLY to
results/review5_constrained.json (a single top-level object with keys
"teb_constrained_experiments", "clmu_constrained_experiments",
"amplitude_ordering_decomposition", plus "_meta"). The lead merges each key at
top level into paper_stats_v1.json.

SOURCES (all real files; nothing copied from the PRE-REGEN stats snapshot)
--------------------------------------------------------------------------
  * TEB baseline   : external/teb_runs/<SITE>/output/LWU_base.txt   (n-1 rows)
  * TEB albedo     : external/teb_albedo19/<SITE>/LWU_alb.txt
  * TEB spin-up    : external/teb_spinup10/<SITE>/LWU_spin.txt (record tail)
  * TEB solar      : SWD.txt / SWU.txt under the two dirs above (eff. albedo)
  * CLMU metrics   : results/clmu_v11_rebuild.json  (v11 ALIGNMENT-FIXED dTsa /
                     lwup for baseline/albedo/spinup; script clmu_post_v11.py).
                     Used INSTEAD of the archived CLMU pipeline, which mis-aligned
                     GR-HECKOR and UK-KingsCollege and produced the spurious
                     "0.24 K spin-up response".
  * CLMU solar     : external/clmu_{baseline,albedo}19/<SITE>_{base,alb}.nc
                     FSR / FSDS  (effective albedo)
  * corpus         : data/urban-plumber/corpus/<SITE>.nc
  * per_site       : results/paper_stats_v1.json  (regenerated; carries the v11-
                     corrected CLMU dTsa) -- used ONLY for the amplitude section
                     and as observed-albedo / dtsa_std reference.

CONVENTIONS (frozen; identical to kg1_radiometric.py / teb_campaign_eval.py)
----------------------------------------------------------------------------
  * PROJECT CONSTANTS:  SIG = 5.67e-8 (NOT the exact Stefan-Boltzmann value),
    EPS = 0.95. The exact constant shifts printed offsets by 0.01 K.
  * TEB alignment:  model LWU row i -> corpus forcing step i+1, i.e.
    model[0:n-1] against corpus[1:n].
  * Nocturnal mask: night_mask & ~pre_spinup_flag & isfinite(obs_LWup) &
    isfinite(forcing_Tair) & isfinite(forcing_LWdown). isfinite(obs_LWup) is
    MANDATORY (dropping it moves the TEB slope by ~6e-3).
  * dTsa = mean( ((LWup-(1-EPS)*LWdown)/(EPS*SIG))**0.25 - Tair ) over the mask.
  * Effective albedo = sum(reflected SW)/sum(down SW) over steps with down SW
    > 300 W/m2 (high-sun flux-weighted ratio); pinned by audit inversion.
  * Urban-PLUMBER FullCollection LWup carries _FillValue=-999.0. Only the
    metforcing forcing_Tair is read from FullCollection here (tail-resolution
    check); it is read masked-aware anyway as due diligence.

Usage:  <py312> review5_constrained_experiments.py
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
SIG, EPS = 5.67e-8, 0.95
HIGH_SUN = 300.0                       # W/m2 threshold for effective-albedo steps

STATS = ROOT / "results" / "paper_stats_v1.json"           # READ-ONLY
V11 = ROOT / "results" / "clmu_v11_rebuild.json"
TEB_WARM = ROOT / "results" / "teb_persite_warming.json"   # cross-check only
OUT = ROOT / "results" / "review5_constrained.json"
PRE_REGEN = (ROOT / "results" / "paper_stats_v1.PRE_REGEN.json")   # optional cross-check file (absent in the release)

stats = json.load(open(STATS, encoding="utf-8"))
P = stats["per_site"]
SITES = sorted(P)
v11 = json.load(open(V11, encoding="utf-8"))
pre = json.load(open(PRE_REGEN, encoding="utf-8")) if PRE_REGEN.exists() else None

# variant record sets (manuscript's three-variant convention)
EXCL_MPLS17 = [s for s in SITES if not s.startswith("US-Minneapolis")]
CORE16 = [s for s in SITES
          if not s.startswith("US-Minneapolis") and s != "PL-Lipowa"]
VARIANTS = {"all19": SITES, "excl_mpls17": EXCL_MPLS17, "core16": CORE16}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def corpus(site):
    """Return night-mask-ready corpus arrays for a site."""
    ds = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
    night = ds["night_mask"].values.astype(bool)
    spin = ds["pre_spinup_flag"].values.astype(bool)
    olw = ds["obs_LWup"].values.astype(float)
    ta = ds["forcing_Tair"].values.astype(float)
    ld = ds["forcing_LWdown"].values.astype(float)
    ds.close()
    return olw, ta, ld, night, spin


def teb_eval(model_lwup, site, tail=False):
    """Evaluate a TEB LWU column against the corpus. model row i -> corpus i+1.
    tail=True: keep only the last n-1 rows (spin-up record window).
    Returns dict(bias_Wm2, dtsa_K, n) or None if too short."""
    olw, ta, ld, night, spin = corpus(site)
    n = len(olw)
    lw = np.asarray(model_lwup, float)
    if tail:
        lw = lw[-(n - 1):]
    if len(lw) < n - 1:
        return None
    m = (night & ~spin & np.isfinite(olw) & np.isfinite(ta)
         & np.isfinite(ld))[1:n]
    model = lw[0:n - 1][m]
    obs = olw[1:n][m]
    ld_ = ld[1:n][m]
    ta_ = ta[1:n][m]
    ts = ((model - (1 - EPS) * ld_) / (EPS * SIG)) ** 0.25
    return {"bias_Wm2": float(np.mean(model - obs)),
            "dtsa_K": float(np.mean(ts - ta_)), "n": int(m.sum())}


def teb_eff_albedo(swd_path, swu_path):
    """TEB effective albedo = sum(SWU)/sum(SWD) over SWD > 300 W/m2."""
    d = np.loadtxt(swd_path)
    u = np.loadtxt(swu_path)
    m = d > HIGH_SUN
    return float(u[m].sum() / d[m].sum())


def clmu_eff_albedo(nc_path):
    """CLM-Urban effective albedo = sum(FSR)/sum(FSDS) over FSDS > 300 W/m2."""
    ds = xr.open_dataset(nc_path)
    fsr = np.asarray(ds["FSR"].values).reshape(-1)
    fsds = np.asarray(ds["FSDS"].values).reshape(-1)
    ds.close()
    m = fsds > HIGH_SUN
    return float(fsr[m].sum() / fsds[m].sum())


def tail_matches(site):
    """True iff the FullCollection metforcing tail is half-hourly and echoes the
    corpus forcing_Tair over its last n steps to < 1e-6 K (i.e. the spin-up run
    is directly comparable to the half-hourly baseline)."""
    mf = ROOT / f"data/urban-plumber/FullCollection/{site}/timeseries/{site}_metforcing_v1.nc"
    if not mf.exists():
        return None
    d = xr.open_dataset(mf, mask_and_scale=True).squeeze(drop=True)
    co = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
    n = co.sizes["time"]
    cta = co["forcing_Tair"].values
    mta = np.asarray(d["Tair"].values, float)
    d.close()
    co.close()
    return bool(len(mta) >= n and float(np.nanmax(np.abs(mta[-n:] - cta))) < 1e-6)


def slope_r(x, y):
    return float(np.polyfit(x, y, 1)[0]), float(pearsonr(x, y)[0])


def cmp_leaf(check, name, new, old, tol=1e-9, reason=""):
    """Record a reproduction comparison leaf."""
    if old is None:
        check[name] = {"status": "new_leaf", "recomputed": new}
        return
    try:
        same = abs(float(new) - float(old)) <= tol
    except (TypeError, ValueError):
        same = (new == old)
    entry = {"recomputed": new, "pre_regen": old,
             "status": "reproduced" if same else "changed"}
    if not same:
        entry["delta"] = (float(new) - float(old)
                          if isinstance(new, (int, float)) else None)
        entry["reason"] = reason
    check[name] = entry


# --------------------------------------------------------------------------- #
# 1. TEB constrained experiments
# --------------------------------------------------------------------------- #
def build_teb():
    teb_pre = (pre or {}).get("teb_constrained_experiments", {})
    check = {}
    out = {"_meta": {
        "sources": {
            "baseline": "external/teb_runs/<SITE>/output/LWU_base.txt",
            "albedo": "external/teb_albedo19/<SITE>/LWU_alb.txt",
            "spinup": "external/teb_spinup10/<SITE>/LWU_spin.txt",
            "solar": "SWD.txt / SWU.txt under teb_runs & teb_albedo19",
            "corpus": "data/urban-plumber/corpus/<SITE>.nc"},
        "conventions": {"SIG": SIG, "EPS": EPS,
                        "alignment": "model row i -> corpus step i+1",
                        "eff_albedo": "sum(SWU)/sum(SWD) over SWD>300 W/m2"}}}

    # baseline reproduction: fresh validation LWU vs preserved baseline LWU
    vw = ROOT / "external/teb_albedo19/_validate_AU-Preston/output/LWU.txt"
    base = ROOT / "external/teb_runs/AU-Preston/output/LWU_base.txt"
    a, b = np.loadtxt(vw), np.loadtxt(base)
    md = float(np.max(np.abs(a - b))) if len(a) == len(b) else None
    out["baseline_reproduction"] = {
        "max_abs_diff_Wm2": md, "n": int(len(a)),
        "verdict": "EXACT" if md is not None and md < 1e-6 else "MISMATCH"}

    # per-site baseline & albedo (raw)
    base_dtsa, base_bias, alb_dtsa, alb_bias = {}, {}, {}, {}
    for s in SITES:
        rb = teb_eval(np.loadtxt(ROOT / f"external/teb_runs/{s}/output/LWU_base.txt"), s)
        ra = teb_eval(np.loadtxt(ROOT / f"external/teb_albedo19/{s}/LWU_alb.txt"), s)
        base_dtsa[s], base_bias[s] = rb["dtsa_K"], rb["bias_Wm2"]
        alb_dtsa[s], alb_bias[s] = ra["dtsa_K"], ra["bias_Wm2"]

    obs = np.array([P[s]["dtsa_std"] for s in SITES])
    bd = np.array([base_dtsa[s] for s in SITES])
    ad = np.array([alb_dtsa[s] for s in SITES])
    sb, rb_ = slope_r(obs, bd)
    sa, ra_ = slope_r(obs, ad)
    out["albedo_constrained"] = {
        "n": len(SITES),
        "baseline_slope_model_on_obs": sb, "baseline_r": rb_,
        "albedo_slope_model_on_obs": sa, "albedo_r": ra_,
        "baseline_model_span_K": [float(bd.min()), float(bd.max())],
        "albedo_model_span_K": [float(ad.min()), float(ad.max())],
        "obs_span_K": [float(obs.min()), float(obs.max())],
        "mean_abs_bias_baseline_Wm2": float(np.mean(np.abs(
            [base_bias[s] for s in SITES]))),
        "mean_abs_bias_albedo_Wm2": float(np.mean(np.abs(
            [alb_bias[s] for s in SITES])))}

    # NEW: per-record albedo-constrained warming (albedo - baseline)
    warm = {s: alb_dtsa[s] - base_dtsa[s] for s in SITES}
    wv = np.array([warm[s] for s in SITES])
    outside = [s for s in SITES if not (0.2 <= warm[s] <= 0.5)]
    out["albedo_constrained_warming_per_record"] = {
        "min_K": float(wv.min()), "max_K": float(wv.max()),
        "mean_K": float(wv.mean()), "median_K": float(np.median(wv)),
        "n_outside_0.2_0.5": len(outside),
        "sites_outside_0.2_0.5": outside,
        "per_record_K": {s: warm[s] for s in SITES},
        "note": ("manuscript's 'near-uniform 0.2-0.5 K' is FALSE: "
                 f"{wv.min():.4f}..{wv.max():.4f} K, {len(outside)}/19 outside")}

    # cross-check against results/teb_persite_warming.json (independent run)
    if TEB_WARM.exists():
        tw = json.load(open(TEB_WARM, encoding="utf-8"))
        dmax = max(abs(warm[s] - tw["per_record_warming_K"][s]) for s in SITES)
        out["albedo_constrained_warming_per_record"]["crosscheck_teb_persite_warming_maxabs"] = dmax

    # equilibrium spin-up (TEB): 15 half-hourly-comparable records
    excl, clean = [], []
    for s in SITES:
        tm = tail_matches(s)
        (clean if tm else excl).append(s)
    spin_dtsa, spin_bias = {}, {}
    for s in clean:
        rs = teb_eval(np.loadtxt(ROOT / f"external/teb_spinup10/{s}/LWU_spin.txt"),
                      s, tail=True)
        spin_dtsa[s], spin_bias[s] = rs["dtsa_K"], rs["bias_Wm2"]
    obsc = np.array([P[s]["dtsa_std"] for s in clean])
    bdc = np.array([base_dtsa[s] for s in clean])
    sdc = np.array([spin_dtsa[s] for s in clean])
    dchg = np.abs(sdc - bdc)
    out["equilibrium_spinup"] = {
        "n_clean": len(clean), "n_excluded": len(excl),
        "excluded_sites": excl,
        "excluded_reason": "hourly FullCollection metforcing (not half-hourly-comparable)",
        "max_abs_dtsa_change_K": float(dchg.max()),
        "mean_abs_dtsa_change_K": float(dchg.mean()),
        "max_abs_bias_change_Wm2": float(np.max(np.abs(
            [spin_bias[s] - base_bias[s] for s in clean]))),
        "baseline_slope": float(np.polyfit(obsc, bdc, 1)[0]),
        "spinup_slope": float(np.polyfit(obsc, sdc, 1)[0])}

    # native vs observed albedo (effective albedo from raw solar columns)
    nat = {s: teb_eff_albedo(ROOT / f"external/teb_runs/{s}/output/SWD.txt",
                             ROOT / f"external/teb_runs/{s}/output/SWU.txt")
           for s in SITES}
    ach = {s: teb_eff_albedo(ROOT / f"external/teb_albedo19/{s}/output/SWD.txt",
                             ROOT / f"external/teb_albedo19/{s}/output/SWU.txt")
           for s in SITES}
    obs_alb = {s: P[s]["albedo"] for s in SITES}
    nv = np.array([nat[s] for s in SITES])
    av = np.array([ach[s] for s in SITES])
    ov = np.array([obs_alb[s] for s in SITES])
    out["native_vs_observed_albedo"] = {
        "native_range": [float(nv.min()), float(nv.max())],
        "observed_range": [float(ov.min()), float(ov.max())],
        "achieved_effective_range": [float(av.min()), float(av.max())],
        "r_native_vs_observed": float(pearsonr(nv, ov)[0]),
        "r_achieved_vs_observed": float(pearsonr(av, ov)[0]),
        "native_effective_per_site": nat,
        "achieved_effective_per_site": ach,
        "observed_albedo_per_site": obs_alb}

    # ---- reproduction check vs PRE-REGEN --------------------------------- #
    if teb_pre:
        p = teb_pre
        cmp_leaf(check, "baseline_reproduction.max_abs_diff_Wm2",
                 out["baseline_reproduction"]["max_abs_diff_Wm2"],
                 p.get("baseline_reproduction", {}).get("max_abs_diff_Wm2"))
        for k in ("baseline_slope_model_on_obs", "baseline_r",
                  "albedo_slope_model_on_obs", "albedo_r",
                  "mean_abs_bias_baseline_Wm2", "mean_abs_bias_albedo_Wm2"):
            cmp_leaf(check, f"albedo_constrained.{k}",
                     out["albedo_constrained"][k],
                     p.get("albedo_constrained", {}).get(k), tol=1e-6)
        for k in ("baseline_model_span_K", "albedo_model_span_K", "obs_span_K"):
            newr = out["albedo_constrained"][k]
            oldr = p.get("albedo_constrained", {}).get(k)
            if oldr is not None:
                dmax = max(abs(newr[0] - oldr[0]), abs(newr[1] - oldr[1]))
                check[f"albedo_constrained.{k}"] = {
                    "recomputed": newr, "pre_regen": oldr,
                    "status": "reproduced" if dmax < 1e-6 else "changed",
                    "max_abs_residual": dmax}
        for k in ("max_abs_dtsa_change_K", "mean_abs_dtsa_change_K",
                  "max_abs_bias_change_Wm2", "baseline_slope", "spinup_slope"):
            cmp_leaf(check, f"equilibrium_spinup.{k}",
                     out["equilibrium_spinup"][k],
                     p.get("equilibrium_spinup", {}).get(k), tol=1e-6)
        # native/observed ranges (pre stored ROUNDED to 3 dp)
        for k in ("native_range", "observed_range", "achieved_effective_range"):
            newr = out["native_vs_observed_albedo"][k]
            oldr = p.get("native_vs_observed_albedo", {}).get(k)
            if oldr is not None:
                dmax = max(abs(newr[0] - oldr[0]), abs(newr[1] - oldr[1]))
                check[f"native_vs_observed_albedo.{k}"] = {
                    "recomputed": newr, "pre_regen_rounded3dp": oldr,
                    "status": "reproduced" if dmax < 5e-4 else "changed",
                    "max_abs_residual": dmax}
    out["_reproduction_check"] = check
    return out


# --------------------------------------------------------------------------- #
# 2. CLMU constrained experiments (v11 alignment; within matched pairs)
# --------------------------------------------------------------------------- #
def build_clmu():
    clmu_pre = (pre or {}).get("clmu_constrained_experiments", {})
    check = {}
    camp = v11["campaigns"]
    ctrl = v11["control"]
    base_d = {s: camp["baseline"][s]["dTsa_model"] for s in SITES}
    alb_d = {s: camp["albedo"][s]["dTsa_model"] for s in SITES}
    spin_d = {s: camp["spinup"][s]["dTsa_model"] for s in SITES}

    out = {"_meta": {
        "sources": {
            "dtsa_lwup": "results/clmu_v11_rebuild.json (v11 alignment-fixed; "
                         "script scripts/clmu_post_v11.py)",
            "solar": "external/clmu_{baseline,albedo}19/<SITE>_{base,alb}.nc FSR/FSDS",
            "corpus": "data/urban-plumber/corpus/<SITE>.nc"},
        "conventions": {"eff_albedo": "sum(FSR)/sum(FSDS) over FSDS>300 W/m2",
                        "within_pair": "campaign minus v11 baseline, same record"},
        "note": ("archived CLMU pipeline mis-aligned GR-HECKOR & UK-KingsCollege; "
                 "its '0.24 K spin-up response' was a cross-alignment artefact. "
                 "All numbers here reference the v11 baseline within matched pairs.")}}

    obs = np.array([P[s]["dtsa_std"] for s in SITES])
    bd = np.array([base_d[s] for s in SITES])
    ad = np.array([alb_d[s] for s in SITES])
    sd = np.array([spin_d[s] for s in SITES])
    sb, rb_ = slope_r(obs, bd)
    sa, ra_ = slope_r(obs, ad)
    ss, rs_ = slope_r(obs, sd)

    # within-pair changes
    alb_chg = np.abs(ad - bd)
    spin_chg = np.abs(sd - bd)

    out["albedo_constrained"] = {
        "n": len(SITES),
        "baseline_slope_model_on_obs": sb, "baseline_r": rb_,
        "campaign_slope_model_on_obs": sa, "campaign_r": ra_,
        "baseline_span_K": [float(bd.min()), float(bd.max())],
        "campaign_span_K": [float(ad.min()), float(ad.max())],
        "obs_span_K": [float(obs.min()), float(obs.max())],
        "max_abs_dtsa_change_K": float(alb_chg.max()),
        "mean_abs_dtsa_change_K": float(alb_chg.mean()),
        "per_record_withinpair_delta_dtsa_K":
            {s: alb_d[s] - base_d[s] for s in SITES}}

    out["equilibrium_spinup_5yr"] = {
        "n": len(SITES),
        "baseline_slope_model_on_obs": sb, "baseline_r": rb_,
        "campaign_slope_model_on_obs": ss, "campaign_r": rs_,
        "baseline_span_K": [float(bd.min()), float(bd.max())],
        "campaign_span_K": [float(sd.min()), float(sd.max())],
        "obs_span_K": [float(obs.min()), float(obs.max())],
        "max_abs_dtsa_change_K": float(spin_chg.max()),
        "mean_abs_dtsa_change_K": float(spin_chg.mean()),
        "max_change_site": SITES[int(np.argmax(spin_chg))],
        "per_record_withinpair_delta_dtsa_K":
            {s: spin_d[s] - base_d[s] for s in SITES}}

    # effective albedo (native = baseline .nc, constrained = albedo .nc)
    nat = {s: clmu_eff_albedo(ROOT / f"external/clmu_baseline19/{s}_base.nc")
           for s in SITES}
    con = {s: clmu_eff_albedo(ROOT / f"external/clmu_albedo19/{s}_alb.nc")
           for s in SITES}
    obs_alb = {s: P[s]["albedo"] for s in SITES}
    nv = np.array([nat[s] for s in SITES])
    cv = np.array([con[s] for s in SITES])
    ov = np.array([obs_alb[s] for s in SITES])
    out["native_effective_albedo"] = nat
    out["constrained_effective_albedo"] = con
    out["observed_albedo"] = obs_alb
    out["effective_albedo_correlations"] = {
        "r_native_vs_observed": float(pearsonr(nv, ov)[0]),
        "r_constrained_vs_observed": float(pearsonr(cv, ov)[0]),
        "native_range": [float(nv.min()), float(nv.max())],
        "constrained_range": [float(cv.min()), float(cv.max())]}

    # alignment provenance per record
    out["alignment_provenance"] = {s: {
        "align_offset": camp["baseline"][s]["align_offset"],
        "truncated": camp["baseline"][s]["truncated"],
        "align_rmse_Tair": camp["baseline"][s]["align_rmse_Tair"]}
        for s in SITES}

    # baseline reproduction: v11 vs archived (the alignment correction)
    dd = {s: ctrl[s]["d_dtsa"] for s in SITES}
    worst = max(SITES, key=lambda s: abs(dd[s]))
    out["baseline_reproduction"] = {
        "definition": "v11 dTsa minus archived dTsa, per record",
        "max_abs_dtsa_diff_K": float(abs(dd[worst])),
        "max_abs_diff_site": worst,
        "n_records_changed_gt_1e_3": int(sum(abs(dd[s]) > 1e-3 for s in SITES)),
        "changed_records": {s: dd[s] for s in SITES if abs(dd[s]) > 1e-3},
        "note": ("17 records reproduce archived to ~1.7e-6 (library drift); "
                 "GR-HECKOR & UK-KingsCollege change by the alignment fix.")}

    # ---- reproduction check vs PRE-REGEN --------------------------------- #
    if clmu_pre:
        p = clmu_pre
        RC = "the CLMU v11 alignment correction (GR-HECKOR, UK-KingsCollege)"
        for k in ("baseline_slope_model_on_obs", "baseline_r",
                  "campaign_slope_model_on_obs", "campaign_r",
                  "max_abs_dtsa_change_K", "mean_abs_dtsa_change_K"):
            cmp_leaf(check, f"albedo_constrained.{k}",
                     out["albedo_constrained"][k],
                     p.get("albedo_constrained", {}).get(k),
                     tol=1e-6, reason=RC)
        for k in ("baseline_slope_model_on_obs", "baseline_r",
                  "campaign_slope_model_on_obs", "campaign_r",
                  "max_abs_dtsa_change_K", "mean_abs_dtsa_change_K"):
            cmp_leaf(check, f"equilibrium_spinup_5yr.{k}",
                     out["equilibrium_spinup_5yr"][k],
                     p.get("equilibrium_spinup_5yr", {}).get(k),
                     tol=1e-6, reason=RC + " + archived cross-alignment artefact")
        # spans (list leaves): baseline/campaign spans shift where GR-HECKOR moved
        for sec_key in ("albedo_constrained", "equilibrium_spinup_5yr"):
            for k in ("baseline_span_K", "campaign_span_K", "obs_span_K"):
                newr = out[sec_key].get(k)
                oldr = p.get(sec_key, {}).get(k)
                if newr is not None and oldr is not None:
                    dmax = max(abs(newr[0] - oldr[0]), abs(newr[1] - oldr[1]))
                    check[f"{sec_key}.{k}"] = {
                        "recomputed": newr, "pre_regen": oldr,
                        "status": "reproduced" if dmax < 1e-6 else "changed",
                        "max_abs_residual": dmax,
                        "reason": ("" if dmax < 1e-6 else RC)}
        # observed_albedo is just per_site albedo -> must reproduce bit-exactly
        if "observed_albedo" in p:
            omax = max(abs(obs_alb[s] - p["observed_albedo"][s]) for s in SITES)
            check["observed_albedo.max_abs_residual"] = {
                "value": omax,
                "status": "reproduced" if omax < 1e-12 else "changed"}
        # superseded scalar keys — restructured into 'baseline_reproduction';
        # listed explicitly so they are NOT silently dropped.
        check["baseline_reproduction_maxdiff_K"] = {
            "pre_regen": p.get("baseline_reproduction_maxdiff_K"),
            "status": "unreproducible_superseded",
            "explanation": (
                "PRE-REGEN 0.262 K was the max diff of a MIS-ALIGNED fresh 19-record "
                "re-run vs archived (itself an artefact of the same alignment bug). "
                "The meaningful, alignment-fixed correction is v11-vs-archived = "
                f"{out['baseline_reproduction']['max_abs_dtsa_diff_K']:.5f} K at "
                f"{out['baseline_reproduction']['max_abs_diff_site']}; see the "
                "'baseline_reproduction' block.")}
        check["fresh_baseline_slope"] = {
            "pre_regen": p.get("fresh_baseline_slope"),
            "status": "unreproducible_superseded",
            "explanation": (
                "PRE-REGEN 0.106 was a fresh-run slope on a differently-aligned "
                "baseline; the authoritative v11 baseline slope on obs is "
                f"{out['albedo_constrained']['baseline_slope_model_on_obs']:.5f}.")}
        check["fresh_baseline_note"] = {
            "pre_regen": p.get("fresh_baseline_note"),
            "status": "dropped_narrative",
            "explanation": "free-text note about the fresh re-run; not a numeric leaf."}
        # effective albedo per record should reproduce bit-exactly
        emax_n = max(abs(nat[s] - p["native_effective_albedo"][s])
                     for s in SITES) if "native_effective_albedo" in p else None
        emax_c = max(abs(con[s] - p["constrained_effective_albedo"][s])
                     for s in SITES) if "constrained_effective_albedo" in p else None
        check["native_effective_albedo.max_abs_residual"] = {
            "value": emax_n, "status": "reproduced" if (emax_n or 0) < 1e-9 else "changed"}
        check["constrained_effective_albedo.max_abs_residual"] = {
            "value": emax_c, "status": "reproduced" if (emax_c or 0) < 1e-9 else "changed"}
    out["_reproduction_check"] = check
    return out


# --------------------------------------------------------------------------- #
# 3. amplitude / ordering decomposition (regenerated per_site)
# --------------------------------------------------------------------------- #
def build_amplitude():
    amp_pre = (pre or {}).get("amplitude_ordering_decomposition", {})
    check = {}
    out = {"note": ("model-on-obs slope beta = r * (s_model/s_obs); amplitude "
                    "(sd/range/iqr ratio) and ordering (r, rho) reported "
                    "separately. per_site source = regenerated "
                    "results/paper_stats_v1.json (v11-corrected CLMU dTsa)."),
           "_beta_identity_max_residual": 0.0}
    max_resid = 0.0
    for vn, ss in VARIANTS.items():
        o = np.array([P[s]["dtsa_std"] for s in ss])
        so = o.std(ddof=1)
        oq = np.subtract(*np.percentile(o, [75, 25]))
        orange = o.max() - o.min()
        row = {}
        for sch, key in [("TEB", "teb_dtsa"), ("CLMU5", "clmu_dtsa")]:
            m = np.array([P[s][key] for s in ss])
            r = float(pearsonr(m, o)[0])
            rho = float(spearmanr(m, o)[0])
            sm = m.std(ddof=1)
            beta = r * (sm / so)
            ols = float(np.polyfit(o, m, 1)[0])
            max_resid = max(max_resid, abs(beta - ols))
            row[sch] = {
                "slope": beta,
                "slope_ols_check": ols,
                "beta_minus_ols": beta - ols,
                "r": r, "rho": rho,
                "sd_ratio": float(sm / so),
                "range_ratio": float((m.max() - m.min()) / orange),
                "iqr_ratio": float(np.subtract(*np.percentile(m, [75, 25])) / oq)}
        out[vn] = row
    out["_beta_identity_max_residual"] = max_resid

    # ---- reproduction check vs PRE-REGEN (only all19 & core16 existed) ---- #
    if amp_pre:
        for vn in ("all19", "core16"):
            if vn not in amp_pre:
                continue
            for sch in ("TEB", "CLMU5"):
                for k in ("slope", "r", "rho", "sd_ratio",
                          "range_ratio", "iqr_ratio"):
                    reason = ("CLMU v11 alignment correction propagated into "
                              "per_site clmu_dtsa" if sch == "CLMU5" else "")
                    cmp_leaf(check, f"{vn}.{sch}.{k}",
                             out[vn][sch][k], amp_pre[vn][sch].get(k),
                             tol=1e-9, reason=reason)
        check["excl_mpls17"] = {"status": "new_variant",
                                "note": "absent from PRE-REGEN (violated 3-variant convention)"}
    out["_reproduction_check"] = check
    return out


# --------------------------------------------------------------------------- #
def main():
    teb = build_teb()
    clmu = build_clmu()
    amp = build_amplitude()
    result = {
        "_meta": {
            "script": "analysis/revalidation_2026-08/scripts/review5_constrained_experiments.py",
            "date": "2026-09-16",
            "purpose": ("regenerate the 3 constrained-experiment stats sections "
                        "from raw model output (closes 44/113 unsourced leaves)"),
            "constants": {"SIG": SIG, "EPS": EPS, "high_sun_Wm2": HIGH_SUN},
            "output_is_read_only_merge": ("lead merges each top-level key into "
                                          "results/paper_stats_v1.json")},
        "teb_constrained_experiments": teb,
        "clmu_constrained_experiments": clmu,
        "amplitude_ordering_decomposition": amp}
    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print(f"wrote {OUT}")

    # console summary
    print("\n=== TEB albedo-constrained warming (per record) ===")
    w = teb["albedo_constrained_warming_per_record"]
    print(f"  min {w['min_K']:.4f}  max {w['max_K']:.4f}  mean {w['mean_K']:.4f} "
          f"median {w['median_K']:.4f}  outside[0.2,0.5]={w['n_outside_0.2_0.5']}/19")
    print(f"  crosscheck vs teb_persite_warming.json max|d|="
          f"{w.get('crosscheck_teb_persite_warming_maxabs')}")
    print("\n=== CLMU spin-up within-pair (v11) ===")
    e = clmu["equilibrium_spinup_5yr"]
    print(f"  max|dTsa change|={e['max_abs_dtsa_change_K']:.4f} K at "
          f"{e['max_change_site']}  mean={e['mean_abs_dtsa_change_K']:.4f} K "
          f"(archived artefact was 0.2425 K)")
    print("\n=== amplitude/ordering (beta identity max residual = "
          f"{amp['_beta_identity_max_residual']:.2e}) ===")
    for vn in VARIANTS:
        for sch in ("TEB", "CLMU5"):
            r = amp[vn][sch]
            print(f"  {vn:12s} {sch:6s} beta={r['slope']:+.4f} r={r['r']:+.4f} "
                  f"rho={r['rho']:+.4f} sd={r['sd_ratio']:.4f} "
                  f"range={r['range_ratio']:.4f} iqr={r['iqr_ratio']:.4f}")


if __name__ == "__main__":
    main()
