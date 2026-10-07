# -*- coding: utf-8 -*-
"""review5_benchmark_skill.py

Conventional per-record NOCTURNAL skill on upwelling longwave (LW-up) for the
two urban land-surface schemes actually run in this project (TEB baseline and
CLM-Urban / CLMU5 baseline), plus Urban-PLUMBER-style OUT-OF-SAMPLE benchmark
analogues, all computed on the ALIGNMENT-CORRECTED (v11) model output.

Purpose: test whether the manuscript's implicit claim -- that conventional
per-site nighttime benchmarks look "satisfactory" for THESE runs while only the
cross-site diagnostic exposes the failure -- is supportable. It never was
computed for our runs before at scheme level; a prior audit found it false on the
pre-correction CLMU values (TEB beat all three benchmark analogues at 3/19).
This recomputes on corrected output.

FROZEN CONVENTIONS (inherited, NOT modified here):
  SIG = 5.67e-8, EPS = 0.95  (project constants; NOT exact Stefan-Boltzmann)
  TEB : external/teb_runs/<S>/output/LWU_base.txt , model row i -> corpus step i+1
        nocturnal record-mean mask (from paper_stats.py, frozen):
          night_mask & ~pre_spinup_flag & isfinite(obs_LWup) & isfinite(Tair)
          & isfinite(LWdown) & isfinite(model)
  CLMU: external/clmu_baseline19/<S>_base.nc (FIRE up-LW, FLDS down-LW, Tair echo)
        alignment via clmu_post_v11.align (CORRECTED; NOT clmu_post.py).
        nocturnal LW-up record-mean mask (from clmu_post_v11, frozen):
          (night_mask & ~pre_spinup_flag)[:m] & isfinite(obs_LWup) & isfinite(model)

results/paper_stats_v1.json is READ-ONLY. Output ->
results/review5_benchmark_skill.json, _merge_target
"review5_additions.nocturnal_benchmark_skill".

Interpreter: python
Run with OPENBLAS_NUM_THREADS<=8 (OpenBLAS segfaults at the machine default).
"""
import os
# thread caps MUST precede numpy/sklearn import (OpenBLAS crashes otherwise)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import io, sys, json
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis/revalidation_2026-08/scripts"))
import clmu_post_v11 as v11          # imported alignment (CORRECTED)

SIG, EPS = 5.67e-8, 0.95
KM_K = 27
KM_SEED = 0
KM_NINIT = 10                        # full k-means, reproducible
RTOL_BIAS = 1e-6                     # per-record MBE must reproduce frozen record mean

STATS = json.load(io.open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
P = STATS["per_site"]
SITES = sorted(P)
CLMU_BASE = json.load(io.open(ROOT / "results/clmu_v11_rebuild.json",
                              encoding="utf-8"))["campaigns"]["baseline"]


# --------------------------------------------------------------------------- #
# Task (3): what benchmark suite / metric families did Urban-PLUMBER Phase 1    #
# actually use (incl. at night)?  Verified against primary sources below.       #
#   - Urban-PLUMBER Phase 1 benchmark archive, Zenodo 7330052                   #
#     (https://zenodo.org/records/7330052)                                      #
#   - Best et al. (2015) PLUMBER, J.Hydrometeorol. 16:1425-1442,               #
#     doi:10.1175/JHM-D-14-0158.1 (read at PMC5884676, open access) -- the      #
#     benchmark + metric methodology Urban-PLUMBER "adopts ... following the    #
#     PLUMBER project".                                                         #
#   - Lipson et al. (2024) QJRMS 150:126-169, doi:10.1002/qj.4589 (CC BY);      #
#     Wiley full text returned HTTP 403 to automated fetch, so body-text quotes #
#     attributed to it are flagged UNVERIFIED-AT-SOURCE.                        #
# --------------------------------------------------------------------------- #
LIPSON_METRIC_FAMILIES = {
    "question": "What benchmark suite and metric families did Urban-PLUMBER Phase 1 "
                "actually use (including at night), so the manuscript need not call "
                "them 'level-based'?",
    "verdict": "NOT level-based. Phase 1 benchmarking (adopted from PLUMBER, Best "
               "et al. 2015) scores models with a metric set spanning THREE families "
               "-- (i) level/magnitude bias, (ii) variability/distribution, (iii) "
               "association/timing -- each evaluable within any time subset incl. "
               "night. Level (bias) is one of three families, not the whole. This "
               "SUPPORTS the manuscript's existing wording ('whether level, "
               "correlation, or distribution').",
    "metric_families": {
        "level_magnitude_bias": {
            "metrics": ["Mean Bias Error (MBE)", "Mean Absolute Error (MAE)",
                        "Normalised Mean Error (NME)"],
            "note": "NME is the SUMMARY metric; it aggregates the other families."},
        "variability_distribution": {
            "metrics": ["Standard-Deviation difference / normalised standard "
                        "deviation (nSD)"]},
        "association_timing": {
            "metrics": ["Correlation coefficient (Pearson r)", "regression slope"]},
    },
    "summary_metric_quote": {
        "text": "the first three metrics provide independent information about model "
                "performance, while normalized mean error contains information about "
                "all three previous metrics, and is commonly used as a summary metric",
        "source": "Best et al. (2015), doi:10.1175/JHM-D-14-0158.1, read at "
                  "https://pmc.ncbi.nlm.nih.gov/articles/PMC5884676/",
        "confidence": "verbatim-as-extracted from the PMC full text"},
    "benchmark_suite_actual": {
        "source_primary": "Urban-PLUMBER Phase 1 benchmark archive (AU-Preston), "
                          "Zenodo 7330052 (https://zenodo.org/records/7330052)",
        "source_method": "Best et al. (2015) PLUMBER, doi:10.1175/JHM-D-14-0158.1; "
                         "Urban-PLUMBER adopts benchmarks following PLUMBER",
        "out_of_sample_empirical": {
            "REG1-SWdown": "linear regression on one variable (SWdown); PLUMBER "
                           "name '1lin'",
            "REG2-SWdown-Tair": "two-variable (SWdown, Tair) linear regression; "
                                "PLUMBER '2lin'",
            "KM3-SWdown-Tair-RH": "three predictors (SWdown, Tair, RH) sorted by "
                                  "k-means into 27 groups with a WITHIN-CLUSTER "
                                  "piecewise-LINEAR regression per group (PLUMBER "
                                  "'3km27'); NOT a cluster mean"},
        "in_sample_empirical": {
            "KM3-IS-SWdown-Tair-RH": "as KM3 but trained on the test site's own data",
            "KM4-IS-SWdown-Tair-Wind": "adds wind; 81 (3^4) clusters; in-sample"},
        "physically_based": {"Manabe_1T": "single-tile slab-and-bucket model"},
        "out_of_sample_fitting_quote": {
            "text": "Each empirical model was applied out-of-sample separately at "
                    "each Fluxnet site, by calibrating on data from the 19 other "
                    "sites to establish regression parameters, and then using the "
                    "meteorological data from the testing site to predict flux "
                    "variables using these parameters",
            "source": "Best et al. (2015), PMC5884676",
            "confidence": "verbatim-as-extracted"}},
    "night_handling": {
        "diurnal_decomposition_exists": True,
        "manuscript_attributed_quote": {
            "text": "at night, most models beat all out-of-sample benchmarks",
            "attributed_to": "Lipson et al. (2024), doi:10.1002/qj.4589",
            "confidence": "UNVERIFIED-AT-SOURCE: quoted in main.tex and attributed "
                          "to Lipson 2024; Wiley full text returned HTTP 403 to "
                          "automated fetch, so this exact sentence was not "
                          "re-verified against the paper body. Paper is CC BY; a "
                          "human should confirm wording/location before print."},
        "reg1_night_degeneracy_quote": {
            "text": "At night predicted values are constant where SWdown = 0",
            "source": "Urban-PLUMBER Phase 1 benchmark archive, Zenodo 7330052 "
                      "(dataset description, re REG1)",
            "independent_confirmation": "Confirmed here: 96.7% of nocturnal steps "
                      "have SWdown exactly 0 (max 0.9994 W/m2, sd 0.086); LOO REG1 "
                      "collapses to ~the pooled nocturnal mean LW-up, so REG1 is NOT "
                      "a meaningful nighttime hurdle."}},
    "lipson_metric_list_secondary": {
        "text": "Urban-PLUMBER Phase 1 statistical measures include MAE, MBE, "
                "normalised standard deviation (nSD), normalised mean error (NME), "
                "slope and correlation, at sub-hourly resolution",
        "source": "secondary (search-index descriptions of Lipson 2024 / project "
                  "docs); consistent with the PLUMBER metric set above",
        "confidence": "secondary/paraphrase; primary metric definitions are Best 2015"},
    "longwave_finding": {
        "text": "little or no improvement in long-wave radiation (and momentum) "
                "relative to the previous major intercomparison at the same site",
        "source": "Lipson et al. (2024) open-access abstract, doi:10.1002/qj.4589",
        "confidence": "paraphrase of the open-access abstract; substance verified"},
    "how_our_analogues_differ": [
        "REG1/REG2 here match Urban-PLUMBER REG1-SWdown / REG2-SWdown-Tair in "
        "predictors and OLS form.",
        "Our KM3 predicts the CLUSTER-MEAN LW-up; Urban-PLUMBER's KM3/3km27 uses a "
        "within-cluster piecewise-LINEAR regression -> ours is a SIMPLIFIED (weaker, "
        "hence conservative) analogue.",
        "Urban-PLUMBER trains out-of-sample benchmarks on a large external FLUXNET "
        "collection; we train leave-one-record-out across the 19 urban records "
        "(an out-of-sample analogue restricted to the urban corpus).",
        "Urban-PLUMBER Phase 1 scored the 30-model ensemble at ONE site "
        "(AU-Preston); we score OUR two schemes at all 19 records, nocturnal LW-up "
        "only."],
}


# --------------------------------------------------------------------------- #
# corpus loader (all forcing variables share the corpus time axis, aligned to  #
# obs_LWup / night_mask / pre_spinup_flag; forcing is gap-filled = complete)   #
# --------------------------------------------------------------------------- #
def corpus(site):
    ds = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
    d = dict(olw=ds["obs_LWup"].values.astype(float),
             ta=ds["forcing_Tair"].values.astype(float),
             ld=ds["forcing_LWdown"].values.astype(float),
             sw=ds["forcing_SWdown"].values.astype(float),
             qa=ds["forcing_Qair"].values.astype(float),
             night=ds["night_mask"].values.astype(bool),
             presp=ds["pre_spinup_flag"].values.astype(bool))
    ds.close()
    return d


# --------------------------------------------------------------------------- #
# per-scheme aligned NOCTURNAL frame: obs / model / forcing on identical steps  #
# --------------------------------------------------------------------------- #
def frame_teb(site):
    C = corpus(site)
    n = len(C["olw"])
    teb = np.loadtxt(ROOT / f"external/teb_runs/{site}/output/LWU_base.txt")
    nmod = min(teb.size, n - 1)
    sl = slice(1, nmod + 1)                       # model row i -> corpus step i+1
    m = (C["night"] & ~C["presp"] & np.isfinite(C["olw"])
         & np.isfinite(C["ta"]) & np.isfinite(C["ld"]))
    mt = m[sl] & np.isfinite(teb[:nmod])
    return dict(model=teb[:nmod][mt], obs=C["olw"][sl][mt],
                sw=C["sw"][sl][mt], ta=C["ta"][sl][mt], qa=C["qa"][sl][mt],
                align_offset=1, align_window=int(nmod), truncated=bool(nmod < n - 1))


def frame_clmu(site):
    C = v11.corpus(site)                          # same corpus fields as above
    n = len(C["olw"])
    h = xr.open_dataset(ROOT / f"external/clmu_baseline19/{site}_base.nc")
    fire = np.asarray(h["FIRE"].values).reshape(-1)
    tair_m = (np.asarray(h["Tair"].values).reshape(-1) if "Tair" in h
              else np.asarray(h["TBOT"].values).reshape(-1))
    h.close()
    off, m, _ = v11.align(tair_m, C["ta"], n)     # CORRECTED alignment
    lw = fire[off:off + m]
    night = C["night"][:m] & ~C["presp"][:m]
    okb = night & np.isfinite(C["olw"][:m]) & np.isfinite(lw)  # frozen LW-up mask
    Cf = corpus(site)                             # SWdown / Qair on same axis
    return dict(model=lw[okb], obs=C["olw"][:m][okb],
                sw=Cf["sw"][:m][okb], ta=C["ta"][:m][okb], qa=Cf["qa"][:m][okb],
                align_offset=int(off), align_window=int(m), truncated=bool(m < n))


FRAME = {"TEB": frame_teb, "CLMU5": frame_clmu}


# --------------------------------------------------------------------------- #
# (1) per-record nocturnal skill on LW-up                                       #
# --------------------------------------------------------------------------- #
def skill(model, obs):
    e = model - obs
    mbe = float(np.mean(e))
    rmse = float(np.sqrt(np.mean(e * e)))
    dm = model - model.mean(); do = obs - obs.mean()
    crmse = float(np.sqrt(np.mean((dm - do) ** 2)))
    r = float(pearsonr(model, obs)[0])
    sd_o = float(np.std(obs)); sd_m = float(np.std(model))
    nsd = float(sd_m / sd_o) if sd_o > 0 else float("nan")
    # PALS / PLUMBER normalized mean error: sum|M-O| / sum|O-mean(O)|
    denom = float(np.sum(np.abs(obs - obs.mean())))
    nme = float(np.sum(np.abs(e)) / denom) if denom > 0 else float("nan")
    # Willmott (1981) index of agreement d
    obar = obs.mean()
    wd_den = float(np.sum((np.abs(model - obar) + np.abs(obs - obar)) ** 2))
    d = float(1.0 - np.sum(e * e) / wd_den) if wd_den > 0 else float("nan")
    p95 = float(np.percentile(np.abs(e), 95))
    return dict(MBE=mbe, RMSE=rmse, cRMSE=crmse, r=r, nSD=nsd,
                NME=nme, d=d, p95_abs_err=p95, n=int(obs.size),
                sd_obs=sd_o, sd_model=sd_m)


# --------------------------------------------------------------------------- #
# (2) Urban-PLUMBER-style out-of-sample benchmark analogues (leave-one-record  #
#     -out), scored by MAE on nocturnal LW-up.                                  #
#     REG1 = OLS on SWdown ; REG2 = OLS on SWdown+Tair ;                        #
#     KM3  = k-means(27) on standardized (SWdown,Tair,Qair), cluster-mean obs.  #
# --------------------------------------------------------------------------- #
def loo_benchmarks(frames, order):
    obs = {s: frames[s]["obs"] for s in order}
    sw = {s: frames[s]["sw"] for s in order}
    ta = {s: frames[s]["ta"] for s in order}
    qa = {s: frames[s]["qa"] for s in order}
    out = {}
    reg1_pred_sd = {}
    for held in order:
        tr = [s for s in order if s != held]
        y_tr = np.concatenate([obs[s] for s in tr])
        # REG1: SWdown only
        X1_tr = np.concatenate([sw[s] for s in tr])[:, None]
        X1_te = sw[held][:, None]
        p1 = LinearRegression().fit(X1_tr, y_tr).predict(X1_te)
        # REG2: SWdown + Tair
        X2_tr = np.column_stack([np.concatenate([sw[s] for s in tr]),
                                 np.concatenate([ta[s] for s in tr])])
        X2_te = np.column_stack([sw[held], ta[held]])
        p2 = LinearRegression().fit(X2_tr, y_tr).predict(X2_te)
        # KM3: standardized (SWdown,Tair,Qair), cluster-mean obs
        X3_tr = np.column_stack([np.concatenate([sw[s] for s in tr]),
                                 np.concatenate([ta[s] for s in tr]),
                                 np.concatenate([qa[s] for s in tr])])
        X3_te = np.column_stack([sw[held], ta[held], qa[held]])
        sc = StandardScaler().fit(X3_tr)
        km = KMeans(n_clusters=KM_K, n_init=KM_NINIT, max_iter=300,
                    random_state=KM_SEED).fit(sc.transform(X3_tr))
        lab_tr = km.labels_
        cmean = np.array([y_tr[lab_tr == c].mean() if np.any(lab_tr == c)
                          else y_tr.mean() for c in range(KM_K)])
        p3 = cmean[km.predict(sc.transform(X3_te))]
        o = obs[held]
        out[held] = dict(REG1=float(np.mean(np.abs(p1 - o))),
                         REG2=float(np.mean(np.abs(p2 - o))),
                         KM3=float(np.mean(np.abs(p3 - o))))
        reg1_pred_sd[held] = float(np.std(p1))
    return out, reg1_pred_sd


def bias_decomposition(frames, bench_per_record):
    """Decompose each scheme's benchmark result into bias vs temporal structure.

    At night nocturnal LW-up has small within-record temporal variance, so
    per-record MAE is dominated by the mean bias; the out-of-sample benchmark
    test is therefore, in effect, a bias test. 'Debiased' error removes the
    per-record mean error: mean|e - MBE| (the mean absolute deviation of the
    errors). We recount benchmark wins using the debiased scheme error so the
    reader can see how much of a scheme's benchmark failure is pure bias."""
    out = {"note": "debiased error = mean|e - MBE| (per-record mean bias removed); "
                   "REG1 omitted (degenerate at night). abs_MBE_over_MAE near 1 "
                   "means MAE is essentially the mean bias.", "per_record": {}}
    for scheme in ("TEB", "CLMU5"):
        pr = {}
        raw = {"REG2": 0, "KM3": 0, "both": 0}
        deb = {"REG2": 0, "KM3": 0, "both": 0}
        fracs = []
        absmbe = []
        maes = []
        for s in frames[scheme]:
            e = frames[scheme][s]["model"] - frames[scheme][s]["obs"]
            mbe = float(e.mean()); mae = float(np.abs(e).mean())
            demae = float(np.abs(e - mbe).mean())
            reg2 = bench_per_record[scheme][s]["REG2_MAE"]
            km3 = bench_per_record[scheme][s]["KM3_MAE"]
            rb2 = mae < reg2; rbk = mae < km3
            db2 = demae < reg2; dbk = demae < km3
            raw["REG2"] += rb2; raw["KM3"] += rbk; raw["both"] += (rb2 and rbk)
            deb["REG2"] += db2; deb["KM3"] += dbk; deb["both"] += (db2 and dbk)
            fracs.append(abs(mbe) / mae if mae > 0 else float("nan"))
            absmbe.append(abs(mbe)); maes.append(mae)
            pr[s] = dict(MBE=mbe, MAE=mae, abs_MBE_over_MAE=abs(mbe) / mae,
                         debiased_MAE=demae, REG2_MAE=reg2, KM3_MAE=km3,
                         raw_beats_both=bool(rb2 and rbk),
                         debiased_beats_both=bool(db2 and dbk))
        out[scheme] = dict(
            raw_beats={k: int(v) for k, v in raw.items()},
            debiased_beats={k: int(v) for k, v in deb.items()},
            median_absMBE_over_MAE=float(np.median(fracs)),
            corr_absMBE_MAE=float(np.corrcoef(absmbe, maes)[0, 1]))
        out["per_record"][scheme] = pr
    return out


def summarize(per):
    keys = ["MBE", "RMSE", "cRMSE", "r", "nSD", "NME", "d", "p95_abs_err"]
    s = {}
    for k in keys:
        v = np.array([per[site][k] for site in per])
        s[k] = dict(median=float(np.median(v)),
                    min=float(np.min(v)), max=float(np.max(v)))
    s["n_records_r_ge_0.90"] = int(np.sum(
        np.array([per[site]["r"] for site in per]) >= 0.90))
    s["n_records"] = len(per)
    return s


def main():
    out = {
        "_merge_target": "review5_additions.nocturnal_benchmark_skill",
        "meta": {
            "purpose": "Conventional per-record nocturnal LW-up skill for TEB and "
                       "CLMU5 baseline runs, plus Urban-PLUMBER-style out-of-sample "
                       "benchmark analogues, on the alignment-corrected (v11) output.",
            "generator": "analysis/revalidation_2026-08/scripts/review5_benchmark_skill.py",
            "interpreter": "Python312 (numpy/scipy/xarray/netCDF4/sklearn)",
            "constants": {"SIG": SIG, "EPS": EPS, "note": "SIG is the project "
                          "constant 5.67e-8, NOT exact Stefan-Boltzmann"},
            "alignment": {
                "TEB": "model LWU row i -> corpus forcing step i+1 (model[0:n-1] "
                       "vs corpus[1:n]); mask night&~presp&finite(obs,Tair,LWdown,model)",
                "CLMU5": "clmu_post_v11.align (corrected): offset min(17520,n)+1, "
                         "truncated final window allowed; SG-TelokKurau06 degeneracy "
                         "resolves to offset 16081 (window 16080); mask "
                         "(night&~presp)[:m]&finite(obs,model)"},
            "metric_definitions": {
                "MBE": "mean(model-obs)  [W/m2]",
                "RMSE": "sqrt(mean((model-obs)^2))",
                "cRMSE": "centred RMSE = sqrt(mean(((model-mean_m)-(obs-mean_o))^2)); "
                         "cRMSE^2 = RMSE^2 - MBE^2",
                "r": "Pearson correlation(model, obs)",
                "nSD": "std(model)/std(obs), population std (ddof=0)",
                "NME": "PALS/PLUMBER normalized mean error = sum|M-O| / sum|O-mean(O)|",
                "d": "Willmott (1981) index of agreement = 1 - sum(e^2)/"
                     "sum((|M-Obar|+|O-Obar|)^2)",
                "p95_abs_err": "95th percentile of |model-obs|"},
            "benchmarks": {
                "REG1": "OLS linear regression of nocturnal LW-up on SWdown, "
                        "leave-one-record-out (fit on other 18 records, predict held-out)",
                "REG2": "OLS on SWdown and Tair, leave-one-record-out",
                "KM3": "k-means(27) on standardized (SWdown,Tair,Qair) with "
                       "cluster-mean obs as prediction, leave-one-record-out; "
                       "SIMPLIFIED ANALOGUE of PLUMBER 3km27 (which uses a "
                       "within-cluster piecewise-LINEAR regression, not a cluster mean)",
                "km_settings": {"k": KM_K, "n_init": KM_NINIT, "seed": KM_SEED,
                                "standardized": True},
                "score": "mean absolute error (MAE) on nocturnal LW-up; a scheme "
                         "BEATS a benchmark at a record when scheme MAE < benchmark MAE"},
        },
        "validation": {},
        "per_record_skill": {},
        "skill_summary": {},
        "benchmark": {},
        "lipson_metric_families": {},   # filled from verified web reading (see report)
    }

    # ---- build frames, per-record skill, validation ------------------------ #
    frames = {"TEB": {}, "CLMU5": {}}
    val = {"TEB": {"per_record_MBE_matches_frozen": True, "records": {}},
           "CLMU5": {"per_record_MBE_matches_frozen": True, "records": {}}}
    for scheme in ("TEB", "CLMU5"):
        per = {}
        absbias = []
        for s in SITES:
            fr = FRAME[scheme](s)
            frames[scheme][s] = fr
            sk = skill(fr["model"], fr["obs"])
            sk["align_offset"] = fr["align_offset"]
            sk["align_window"] = fr["align_window"]
            sk["truncated"] = fr["truncated"]
            per[s] = sk
            absbias.append(abs(sk["MBE"]))
            # validate MBE reproduces the frozen record mean
            if scheme == "TEB":
                ref = P[s]["teb_lwup_bias"]
            else:
                ref = CLMU_BASE[s]["lwup_bias"]
            dv = abs(sk["MBE"] - ref)
            ok = dv < RTOL_BIAS
            val[scheme]["records"][s] = {"MBE": sk["MBE"], "frozen": ref,
                                         "abs_diff": dv, "pass": bool(ok)}
            if not ok:
                val[scheme]["per_record_MBE_matches_frozen"] = False
        out["per_record_skill"][scheme] = per
        out["skill_summary"][scheme] = summarize(per)
        mab = float(np.mean(absbias))
        val[scheme]["mean_abs_MBE"] = mab
    val["TEB"]["mean_abs_MBE_expected"] = 10.768001
    val["TEB"]["mean_abs_MBE_pass"] = abs(val["TEB"]["mean_abs_MBE"] - 10.768001) < 1e-4
    val["CLMU5"]["mean_abs_MBE_expected"] = 7.686262929993649
    val["CLMU5"]["mean_abs_MBE_pass"] = abs(
        val["CLMU5"]["mean_abs_MBE"] - 7.686262929993649) < 1e-6
    out["validation"] = val

    # ---- benchmarks -------------------------------------------------------- #
    bench = {"per_record_mae": {}, "win_counts": {}, "reg1_pred_sd": {}}
    # REG1 degeneracy diagnostics (pooled nocturnal SWdown, per scheme frame)
    for scheme in ("TEB", "CLMU5"):
        allsw = np.concatenate([frames[scheme][s]["sw"] for s in SITES])
        bench.setdefault("reg1_degeneracy", {})[scheme] = dict(
            sw_max=float(np.max(allsw)), sw_mean=float(np.mean(allsw)),
            sw_sd=float(np.std(allsw)), frac_sw_eq_zero=float(np.mean(allsw == 0)),
            frac_sw_gt_1Wm2=float(np.mean(allsw > 1.0)))
    for scheme in ("TEB", "CLMU5"):
        mae_b, r1sd = loo_benchmarks(frames[scheme], SITES)
        per_mae = {}
        wins = {"beats_REG1": 0, "beats_REG2": 0, "beats_KM3": 0,
                "beats_all_three": 0, "beats_meaningful_two_REG2_KM3": 0}
        for s in SITES:
            m_scheme = float(np.mean(np.abs(
                frames[scheme][s]["model"] - frames[scheme][s]["obs"])))
            b = mae_b[s]
            rec = {"scheme_MAE": m_scheme, "REG1_MAE": b["REG1"],
                   "REG2_MAE": b["REG2"], "KM3_MAE": b["KM3"]}
            rec["beats_REG1"] = m_scheme < b["REG1"]
            rec["beats_REG2"] = m_scheme < b["REG2"]
            rec["beats_KM3"] = m_scheme < b["KM3"]
            rec["beats_all_three"] = all([rec["beats_REG1"], rec["beats_REG2"],
                                          rec["beats_KM3"]])
            rec["beats_meaningful_two_REG2_KM3"] = rec["beats_REG2"] and rec["beats_KM3"]
            per_mae[s] = rec
            for k in wins:
                wins[k] += int(rec[k])
        bench["per_record_mae"][scheme] = per_mae
        bench["win_counts"][scheme] = wins
        bench["reg1_pred_sd"][scheme] = r1sd
    bench["bias_decomposition"] = bias_decomposition(frames, bench["per_record_mae"])
    out["benchmark"] = bench
    out["lipson_metric_families"] = LIPSON_METRIC_FAMILIES

    outpath = ROOT / "results/review5_benchmark_skill.json"
    json.dump(out, io.open(outpath, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    # ---- console summary --------------------------------------------------- #
    print("VALIDATION")
    for scheme in ("TEB", "CLMU5"):
        v = out["validation"][scheme]
        print(f"  {scheme}: per-record MBE reproduces frozen record mean: "
              f"{v['per_record_MBE_matches_frozen']} | mean|MBE|={v['mean_abs_MBE']:.6f} "
              f"(expected {v['mean_abs_MBE_expected']}, pass={v['mean_abs_MBE_pass']})")
    print("\nPER-SCHEME SKILL SUMMARY (median [min, max] across 19 records)")
    for scheme in ("TEB", "CLMU5"):
        S = out["skill_summary"][scheme]
        print(f"  --- {scheme} ---  (# records r>=0.90: {S['n_records_r_ge_0.90']}/19)")
        for k in ["MBE", "RMSE", "cRMSE", "r", "nSD", "NME", "d", "p95_abs_err"]:
            print(f"    {k:12s} median={S[k]['median']:+9.4f}  "
                  f"[{S[k]['min']:+9.4f}, {S[k]['max']:+9.4f}]")
    print("\nBENCHMARK WIN COUNTS (records where scheme MAE < benchmark MAE, /19)")
    for scheme in ("TEB", "CLMU5"):
        w = out["benchmark"]["win_counts"][scheme]
        print(f"  {scheme}: REG1={w['beats_REG1']}  REG2={w['beats_REG2']}  "
              f"KM3={w['beats_KM3']}  all_three={w['beats_all_three']}  "
              f"REG2&KM3={w['beats_meaningful_two_REG2_KM3']}")
    print("\nREG1 degeneracy (pooled nocturnal SWdown):")
    for scheme in ("TEB", "CLMU5"):
        d = out["benchmark"]["reg1_degeneracy"][scheme]
        print(f"  {scheme}: SW max={d['sw_max']:.4f} mean={d['sw_mean']:.4f} "
              f"sd={d['sw_sd']:.4f} frac==0={d['frac_sw_eq_zero']:.4f}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
