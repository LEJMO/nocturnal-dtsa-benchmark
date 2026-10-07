# -*- coding: utf-8 -*-
"""review5_benchmark_cityholdout.py

REVIEWER ISSUE (round 6): the published out-of-sample benchmark protocol is
leave-one-RECORD-out, and the Methods claim that "no record contributes to its
own predictor and the temporal dependence within a record cannot leak between
training and evaluation" is FALSE for the Minneapolis pair, because
US-Minneapolis1 and US-Minneapolis2 are two wind-sector views of the SAME KUOM
tower and share the SAME 2-m radiometer. This script

  (a) AUDITS the duplication directly from the raw Urban-PLUMBER collection and
      from the corpus files the analysis consumes (bit-level comparison of
      obs_LWup / forcing_LWdown / forcing_Tair / forcing_SWdown / forcing_Qair /
      night_mask / pre_spinup_flag, plus the LWup_qc flag vector), and does the
      same check for the Helsinki and Lodz pairs;

  (b) REFITS the three out-of-sample benchmarks (REG1, REG2, KM3) LEAVE-ONE-
      CITY-CLUSTER-OUT on the 16 city clusters the rest of the manuscript
      already uses (Minneapolis = {US-Minneapolis1, US-Minneapolis2},
      Lodz = {PL-Lipowa, PL-Narutowicza}, Helsinki = {FI-Kumpula, FI-Torni},
      13 singletons), with the KM3 standardizer AND the k-means fitted inside
      each training fold, and rescores both schemes on the identical MAE
      convention (a scheme beats a benchmark at a record when its MAE is the
      smaller);

  (c) reports the win counts for all19 / excl_mpls17 / core16 / core15 under
      BOTH restriction conventions -- "eval_only" (the published convention:
      the benchmark training pool is always the full 19-record corpus minus the
      held-out city, and the record set only restricts which records are
      counted) and "refit_within_variant" (the excluded records are removed
      from the training pool as well) -- so the reviewer's third question is
      answered numerically rather than asserted;

  (d) quantifies the two protocol details the reviewer asked about: whether
      fitting the KM3 standardizer on the pooled data instead of inside the
      fold changes anything, and whether standardizing the REG1/REG2 design
      matrix changes anything (it cannot -- OLS is affine-equivariant -- and
      that is verified numerically, not asserted);

  (e) redoes the bias-versus-structure decomposition against the CITY-HOLDOUT
      benchmark MAEs. The bias share of MSE is a property of the model runs and
      is reproduced unchanged; the "debiased proxy" win counts are not, and are
      recomputed with both proxies used elsewhere in the package
      (mean|e-MBE| and cRMSE).

Everything reuses the FROZEN frame builders of review5_benchmark_skill.py by
import, so the masks, the TEB row->step offset and the corrected CLMU alignment
are byte-identical to the published run. As a regression test the record-LOO
counts are recomputed here and asserted against the published values in
results/paper_stats_v1.json.

results/paper_stats_v1.json is READ-ONLY. Output ->
results/review5_benchmark_cityholdout.json,
_merge_target "review5_additions.benchmark_city_holdout".

Interpreter: python
Run with OPENBLAS_NUM_THREADS<=8 (OpenBLAS segfaults at the machine default).
Runtime ~30 min (about 190 k-means fits at k=27, n_init=10).
"""
import os
# thread caps MUST precede numpy/sklearn import (OpenBLAS crashes otherwise)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import io
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "analysis/revalidation_2026-08/scripts"
sys.path.insert(0, str(SCRIPTS))

import review5_benchmark_skill as bs      # FROZEN frames, masks, alignment, k-means settings

# review5_benchmark_skill rebinds sys.stdout to its own UTF-8 TextIOWrapper at
# import time. Do NOT rewrap it here: building a second wrapper over the same
# .buffer and rebinding sys.stdout drops the last reference to the first
# wrapper, whose finalizer closes the shared BufferedWriter, and every later
# print raises "ValueError: I/O operation on closed file". Reuse it instead and
# only turn on line buffering so progress is visible in a redirected log.
sys.stdout.reconfigure(line_buffering=True)

KM_K = bs.KM_K              # 27
KM_SEED = bs.KM_SEED        # 0
KM_NINIT = bs.KM_NINIT      # 10
SIG, EPS = bs.SIG, bs.EPS

STATS = json.load(io.open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
P = STATS["per_site"]
SITES = sorted(bs.STATS["per_site"])
CLUSTER = {s: P[s]["cluster"] for s in SITES}          # authoritative 16-cluster map
PUB = STATS["review5_additions"]["nocturnal_benchmark_skill"]["benchmark"]

PAIRS = {"Minneapolis": ("US-Minneapolis1", "US-Minneapolis2"),
         "Lodz": ("PL-Lipowa", "PL-Narutowicza"),
         "Helsinki": ("FI-Kumpula", "FI-Torni")}

RECORD_SETS = {
    "all19": lambda S: list(S),
    "excl_mpls17": lambda S: [s for s in S if not s.startswith("US-Minneapolis")],
    "core16": lambda S: [s for s in S if not s.startswith("US-Minneapolis")
                         and s != "PL-Lipowa"],
    "core15": lambda S: [s for s in S if not s.startswith("US-Minneapolis")
                         and s not in ("PL-Lipowa", "PL-Narutowicza")],
}

FLAGS = ("beats_REG1", "beats_REG2", "beats_KM3", "beats_all_three",
         "beats_meaningful_two_REG2_KM3")


# --------------------------------------------------------------------------- #
# (a) duplicate-record audit                                                   #
# --------------------------------------------------------------------------- #
def _raw_obs(site):
    p = (ROOT / f"data/urban-plumber/FullCollection/{site}/timeseries/"
                f"{site}_clean_observations_v1.nc")
    ds = xr.open_dataset(p).squeeze(drop=True)
    d = {v: np.asarray(ds[v].values, float).reshape(-1)
         for v in ("LWup", "LWdown", "SWdown", "SWup", "Tair", "Qh", "Qle")}
    d["LWup_qc"] = np.asarray(ds["LWup_qc"].values, float).reshape(-1)
    d["time"] = np.asarray(ds.time.values)
    d["comment"] = str(ds.attrs.get("comment", ""))
    d["obs_reference"] = str(ds.attrs.get("observations_reference", ""))
    d["obs_contact"] = str(ds.attrs.get("observations_contact", ""))
    ds.close()
    return d


def _corpus(site):
    ds = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{site}.nc")
    d = dict(obs_LWup=np.asarray(ds["obs_LWup"].values, float).reshape(-1),
             forcing_LWdown=np.asarray(ds["forcing_LWdown"].values, float).reshape(-1),
             forcing_Tair=np.asarray(ds["forcing_Tair"].values, float).reshape(-1),
             forcing_SWdown=np.asarray(ds["forcing_SWdown"].values, float).reshape(-1),
             forcing_Qair=np.asarray(ds["forcing_Qair"].values, float).reshape(-1),
             night_mask=np.asarray(ds["night_mask"].values).reshape(-1).astype(bool),
             pre_spinup_flag=np.asarray(ds["pre_spinup_flag"].values).reshape(-1).astype(bool))
    ds.close()
    return d


def _cmp(a, b):
    """Bit-level comparison of two float series of equal length."""
    fin = np.isfinite(a) & np.isfinite(b)
    n_both = int(fin.sum())
    if n_both == 0:
        return dict(n_both_finite=0, n_finite_a=int(np.isfinite(a).sum()),
                    n_finite_b=int(np.isfinite(b).sum()),
                    finite_mask_disagreements=int(np.sum(np.isfinite(a) != np.isfinite(b))),
                    max_abs_diff=None, mean_abs_diff=None, pearson_r=None,
                    n_exactly_equal=0, bit_identical=False,
                    note="no time step is finite in both records")
    d = np.abs(a[fin] - b[fin])
    sa = float(np.std(a[fin]))
    r = (float(np.corrcoef(a[fin], b[fin])[0, 1]) if sa > 0 and np.std(b[fin]) > 0
         else None)
    return dict(n_both_finite=n_both,
                n_finite_a=int(np.isfinite(a).sum()),
                n_finite_b=int(np.isfinite(b).sum()),
                finite_mask_disagreements=int(np.sum(np.isfinite(a) != np.isfinite(b))),
                max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                pearson_r=r, n_exactly_equal=int((d == 0).sum()),
                bit_identical=bool(d.max() == 0.0
                                   and np.array_equal(np.isfinite(a), np.isfinite(b))))


def duplicate_audit():
    out = {"question": ("do any two of the 19 records share the same radiation "
                        "measurements, so that leave-one-RECORD-out leaves the "
                        "held-out targets in the training pool?"),
           "pairs": {}}
    for city, (s1, s2) in PAIRS.items():
        A, B = _raw_obs(s1), _raw_obs(s2)
        CA, CB = _corpus(s1), _corpus(s2)
        n = min(len(CA["obs_LWup"]), len(CB["obs_LWup"]))
        blk = {
            "records": [s1, s2],
            "raw_collection": {
                "file": f"data/urban-plumber/FullCollection/<S>/timeseries/"
                        f"<S>_clean_observations_v1.nc",
                "time_axes_identical": bool(np.array_equal(A["time"], B["time"])),
                "n_steps": int(A["time"].size),
                "comment_attr": {s1: A["comment"], s2: B["comment"]},
                "observations_reference": {s1: A["obs_reference"], s2: B["obs_reference"]},
                "observations_contact": {s1: A["obs_contact"], s2: B["obs_contact"]},
                "variables": {v: _cmp(A[v], B[v])
                              for v in ("LWup", "LWdown", "SWdown", "SWup",
                                        "Tair", "Qh", "Qle")},
                "LWup_qc_flag_vector_identical": bool(np.array_equal(
                    np.nan_to_num(A["LWup_qc"], nan=-9.0),
                    np.nan_to_num(B["LWup_qc"], nan=-9.0))),
            },
            "corpus_analysis_input": {
                "file": "data/urban-plumber/corpus/<S>.nc",
                "lengths": [int(len(CA["obs_LWup"])), int(len(CB["obs_LWup"]))],
                "variables": {v: _cmp(CA[v][:n], CB[v][:n])
                              for v in ("obs_LWup", "forcing_LWdown", "forcing_Tair",
                                        "forcing_SWdown", "forcing_Qair")},
                "night_mask_identical": bool(np.array_equal(
                    CA["night_mask"][:n], CB["night_mask"][:n])),
                "pre_spinup_flag_identical": bool(np.array_equal(
                    CA["pre_spinup_flag"][:n], CB["pre_spinup_flag"][:n])),
            },
            "ledger": {"dtsa_std": [P[s1]["dtsa_std"], P[s2]["dtsa_std"]],
                       "albedo": [P[s1]["albedo"], P[s2]["albedo"]],
                       "teb_dtsa": [P[s1]["teb_dtsa"], P[s2]["teb_dtsa"]],
                       "clmu_dtsa": [P[s1]["clmu_dtsa"], P[s2]["clmu_dtsa"]]},
        }
        mA = (CA["night_mask"][:n] & ~CA["pre_spinup_flag"][:n]
              & np.isfinite(CA["obs_LWup"][:n]) & np.isfinite(CA["forcing_Tair"][:n])
              & np.isfinite(CA["forcing_LWdown"][:n]))
        mB = (CB["night_mask"][:n] & ~CB["pre_spinup_flag"][:n]
              & np.isfinite(CB["obs_LWup"][:n]) & np.isfinite(CB["forcing_Tair"][:n])
              & np.isfinite(CB["forcing_LWdown"][:n]))
        blk["nocturnal_evaluation_mask"] = {
            "n_a": int(mA.sum()), "n_b": int(mB.sum()),
            "identical": bool(np.array_equal(mA, mB)),
            "n_intersection": int((mA & mB).sum())}
        radvars = ("obs_LWup", "forcing_LWdown", "forcing_Tair",
                   "forcing_SWdown", "forcing_Qair")
        blk["verdict_duplicate_targets_and_predictors"] = bool(
            all(blk["corpus_analysis_input"]["variables"][v]["bit_identical"]
                for v in radvars)
            and blk["nocturnal_evaluation_mask"]["identical"])
        out["pairs"][city] = blk
    return out


# --------------------------------------------------------------------------- #
# (b) benchmark refit machinery                                                #
# --------------------------------------------------------------------------- #
def _pool(frames, sites, field):
    return np.concatenate([frames[s][field] for s in sites])


def benchmarks_by_fold(frames, folds, train_pool=None,
                       standardize="in_fold", reg_standardize=False,
                       verbose_tag=""):
    """Fit REG1/REG2/KM3 once per FOLD and predict every record in that fold.

    folds        : list of lists of site names (a fold is held out together)
    train_pool   : sites eligible for training (default: every site in any fold's
                   complement, i.e. the union of folds); the fold's own sites are
                   always removed
    standardize  : 'in_fold'  -> StandardScaler fitted on the training pool only
                   'pooled'   -> StandardScaler fitted on ALL records incl. held-out
                                 (the leakage variant, for sensitivity only)
    reg_standardize : also standardize the REG1/REG2 design matrix (OLS is
                   affine-equivariant, so predictions must be unchanged)
    """
    allsites = sorted({s for f in folds for s in f})
    pool = list(allsites) if train_pool is None else list(train_pool)
    obs, sw, ta, qa = (dict(), dict(), dict(), dict())
    for s in set(allsites) | set(pool):
        obs[s] = frames[s]["obs"]
        sw[s] = frames[s]["sw"]
        ta[s] = frames[s]["ta"]
        qa[s] = frames[s]["qa"]

    sc_pooled = None
    if standardize == "pooled":
        sc_pooled = StandardScaler().fit(np.column_stack(
            [_pool(frames, allsites, "sw"), _pool(frames, allsites, "ta"),
             _pool(frames, allsites, "qa")]))

    out, meta = {}, {}
    for fi, fold in enumerate(folds):
        tr = [s for s in pool if s not in fold]
        te = [s for s in fold if s in allsites]
        if not te:
            continue
        y_tr = np.concatenate([obs[s] for s in tr])
        sw_tr = np.concatenate([sw[s] for s in tr])
        ta_tr = np.concatenate([ta[s] for s in tr])
        qa_tr = np.concatenate([qa[s] for s in tr])

        X1_tr = sw_tr[:, None]
        X2_tr = np.column_stack([sw_tr, ta_tr])
        X3_tr = np.column_stack([sw_tr, ta_tr, qa_tr])

        if reg_standardize:
            s1 = StandardScaler().fit(X1_tr)
            s2 = StandardScaler().fit(X2_tr)
            m1 = LinearRegression().fit(s1.transform(X1_tr), y_tr)
            m2 = LinearRegression().fit(s2.transform(X2_tr), y_tr)
        else:
            s1 = s2 = None
            m1 = LinearRegression().fit(X1_tr, y_tr)
            m2 = LinearRegression().fit(X2_tr, y_tr)

        sc = sc_pooled if standardize == "pooled" else StandardScaler().fit(X3_tr)
        t0 = time.time()
        km = KMeans(n_clusters=KM_K, n_init=KM_NINIT, max_iter=300,
                    random_state=KM_SEED).fit(sc.transform(X3_tr))
        lab_tr = km.labels_
        cmean = np.array([y_tr[lab_tr == c].mean() if np.any(lab_tr == c)
                          else y_tr.mean() for c in range(KM_K)])
        meta[",".join(fold)] = dict(n_train_records=len(tr),
                                    train_records=tr,
                                    n_train_steps=int(y_tr.size),
                                    kmeans_seconds=round(time.time() - t0, 2),
                                    n_empty_clusters=int(sum(
                                        1 for c in range(KM_K)
                                        if not np.any(lab_tr == c))))
        for s in te:
            X1_te = sw[s][:, None]
            X2_te = np.column_stack([sw[s], ta[s]])
            X3_te = np.column_stack([sw[s], ta[s], qa[s]])
            p1 = m1.predict(s1.transform(X1_te) if s1 is not None else X1_te)
            p2 = m2.predict(s2.transform(X2_te) if s2 is not None else X2_te)
            p3 = cmean[km.predict(sc.transform(X3_te))]
            o = obs[s]
            out[s] = dict(REG1=float(np.mean(np.abs(p1 - o))),
                          REG2=float(np.mean(np.abs(p2 - o))),
                          KM3=float(np.mean(np.abs(p3 - o))),
                          REG1_pred_sd=float(np.std(p1)),
                          REG2_pred_mean=float(np.mean(p2)),
                          n=int(o.size))
        if verbose_tag:
            print(f"    [{verbose_tag}] fold {fi+1}/{len(folds)} "
                  f"({'+'.join(fold)}) n_train={len(tr)} "
                  f"{meta[','.join(fold)]['kmeans_seconds']}s", flush=True)
    return out, meta


def score(frames, bench, sites):
    """Per-record flags + win counts, identical MAE convention to the published run."""
    per = {}
    for s in sites:
        e = frames[s]["model"] - frames[s]["obs"]
        mae = float(np.mean(np.abs(e)))
        b = bench[s]
        rec = {"scheme_MAE": mae, "REG1_MAE": b["REG1"],
               "REG2_MAE": b["REG2"], "KM3_MAE": b["KM3"], "n": b["n"]}
        rec["beats_REG1"] = bool(mae < b["REG1"])
        rec["beats_REG2"] = bool(mae < b["REG2"])
        rec["beats_KM3"] = bool(mae < b["KM3"])
        rec["beats_all_three"] = bool(rec["beats_REG1"] and rec["beats_REG2"]
                                      and rec["beats_KM3"])
        rec["beats_meaningful_two_REG2_KM3"] = bool(rec["beats_REG2"] and rec["beats_KM3"])
        per[s] = rec
    counts = {f: int(sum(per[s][f] for s in sites)) for f in FLAGS}
    counts["n"] = len(sites)
    counts["median_scheme_MAE_Wm2"] = float(np.median([per[s]["scheme_MAE"] for s in sites]))
    counts["median_REG2_MAE_Wm2"] = float(np.median([per[s]["REG2_MAE"] for s in sites]))
    counts["median_KM3_MAE_Wm2"] = float(np.median([per[s]["KM3_MAE"] for s in sites]))
    return per, counts


# --------------------------------------------------------------------------- #
# (e) bias / structure decomposition against the city-holdout benchmarks       #
# --------------------------------------------------------------------------- #
def decompose(frames, bench, sites):
    per, agg = {}, {}
    for s in sites:
        e = frames[s]["model"] - frames[s]["obs"]
        mbe = float(e.mean())
        mae = float(np.abs(e).mean())
        demae = float(np.abs(e - mbe).mean())          # proxy 1: mean|e-MBE|
        rmse = float(np.sqrt(np.mean(e * e)))
        crmse = float(np.sqrt(max(rmse ** 2 - mbe ** 2, 0.0)))   # proxy 2
        reg2, km3 = bench[s]["REG2"], bench[s]["KM3"]
        per[s] = dict(MBE=mbe, MAE=mae, RMSE=rmse, cRMSE=crmse,
                      debiased_MAE=demae,
                      abs_MBE_over_MAE=abs(mbe) / mae if mae > 0 else float("nan"),
                      bias_share_of_MSE=(mbe ** 2 / (mbe ** 2 + crmse ** 2)
                                         if (mbe ** 2 + crmse ** 2) > 0 else float("nan")),
                      REG2_MAE=reg2, KM3_MAE=km3,
                      raw_beats_REG2=bool(mae < reg2), raw_beats_KM3=bool(mae < km3),
                      raw_beats_both=bool(mae < reg2 and mae < km3),
                      demae_beats_REG2=bool(demae < reg2), demae_beats_KM3=bool(demae < km3),
                      demae_beats_both=bool(demae < reg2 and demae < km3),
                      crmse_beats_REG2=bool(crmse < reg2), crmse_beats_KM3=bool(crmse < km3),
                      crmse_beats_both=bool(crmse < reg2 and crmse < km3))
    keys = ("raw_beats_REG2", "raw_beats_KM3", "raw_beats_both",
            "demae_beats_REG2", "demae_beats_KM3", "demae_beats_both",
            "crmse_beats_REG2", "crmse_beats_KM3", "crmse_beats_both")
    agg = {k: int(sum(per[s][k] for s in sites)) for k in keys}
    agg["n"] = len(sites)
    for k, f in (("median_bias_share_of_MSE", "bias_share_of_MSE"),
                 ("median_absMBE_over_MAE", "abs_MBE_over_MAE"),
                 ("median_abs_MBE_Wm2", None),
                 ("median_cRMSE_Wm2", "cRMSE"),
                 ("median_debiased_MAE_Wm2", "debiased_MAE")):
        if f is None:
            agg[k] = float(np.median([abs(per[s]["MBE"]) for s in sites]))
        else:
            agg[k] = float(np.median([per[s][f] for s in sites]))
    mae_v = np.array([per[s]["MAE"] for s in sites])
    agg["corr_MAE_with_absMBE"] = float(np.corrcoef(
        mae_v, np.array([abs(per[s]["MBE"]) for s in sites]))[0, 1])
    agg["corr_MAE_with_cRMSE"] = float(np.corrcoef(
        mae_v, np.array([per[s]["cRMSE"] for s in sites]))[0, 1])
    agg["n_records_bias_dominated"] = int(sum(
        per[s]["bias_share_of_MSE"] > 0.5 for s in sites))
    return per, agg


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-record-loo", action="store_true",
                    help="skip the record-LOO regression test (saves ~6 min)")
    ap.add_argument("--skip-pooled-scaler", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    out = {
        "_merge_target": "review5_additions.benchmark_city_holdout",
        "meta": {
            "generator": "analysis/revalidation_2026-08/scripts/review5_benchmark_cityholdout.py",
            "date": "2026-09-17",
            "purpose": ("refit the out-of-sample nocturnal LW-up benchmarks "
                        "leave-one-CITY-CLUSTER-out, because leave-one-RECORD-out "
                        "leaves the held-out Minneapolis observations in the "
                        "training pool via the duplicate record"),
            "frames": ("imported verbatim from review5_benchmark_skill.py "
                       "(frame_teb / frame_clmu): frozen nocturnal masks, TEB "
                       "model row i -> corpus step i+1, corrected CLMU alignment "
                       "clmu_post_v11.align"),
            "constants": {"SIG": SIG, "EPS": EPS,
                          "note": "SIG is the project constant 5.67e-8, NOT exact "
                                  "Stefan-Boltzmann"},
            "kmeans": {"k": KM_K, "n_init": KM_NINIT, "seed": KM_SEED,
                       "max_iter": 300,
                       "standardizer": "StandardScaler fitted INSIDE each training "
                                       "fold (training records only); refitted per fold",
                       "prediction": "training-fold cluster mean of observed LW-up"},
            "score": ("mean absolute error on nocturnal LW-up; a scheme BEATS a "
                      "benchmark at a record when scheme MAE < benchmark MAE "
                      "(identical to the published convention)"),
            "reg_standardization": ("REG1/REG2 are fitted on the RAW design matrix, "
                                    "as in the published run; OLS is affine-"
                                    "equivariant so this cannot matter, and the "
                                    "'ols_affine_invariance_check' block verifies it"),
            "restriction_conventions": {
                "eval_only": ("benchmark training pool = all 19 records minus the "
                              "held-out city cluster; the record set restricts only "
                              "which records are COUNTED. This is what the published "
                              "core15/core16 benchmark counts did."),
                "refit_within_variant": ("the excluded records are removed from the "
                                         "benchmark TRAINING POOL as well, then the "
                                         "benchmarks are refitted leave-one-city-out "
                                         "within the variant")},
        },
        "fold_definition": {},
        "duplicate_record_audit": {},
        "published_protocol_audit": {},
        "validation": {},
        "city_holdout": {},
        "bias_structure_decomposition": {},
    }

    # ---- fold definition --------------------------------------------------- #
    clusters = sorted(set(CLUSTER[s] for s in SITES))
    folds_all = [[s for s in SITES if CLUSTER[s] == c] for c in clusters]
    out["fold_definition"] = {
        "unit": "city cluster (the same 16 clusters used for every bootstrap, "
                "permutation and LOCO correction elsewhere in the manuscript)",
        "source": "results/paper_stats_v1.json per_site[*].cluster",
        "n_clusters": len(clusters),
        "clusters": {c: [s for s in SITES if CLUSTER[s] == c] for c in clusters},
        "n_records": len(SITES),
    }
    print(f"16-cluster folds: {len(clusters)} clusters over {len(SITES)} records")

    # ---- (a) duplicate audit ---------------------------------------------- #
    print("\n[a] duplicate-record audit ...", flush=True)
    out["duplicate_record_audit"] = duplicate_audit()
    for city, blk in out["duplicate_record_audit"]["pairs"].items():
        print(f"  {city:12s} duplicate targets+predictors: "
              f"{blk['verdict_duplicate_targets_and_predictors']}  "
              f"(obs_LWup max|diff|="
              f"{blk['corpus_analysis_input']['variables']['obs_LWup']['max_abs_diff']}, "
              f"r={blk['corpus_analysis_input']['variables']['obs_LWup']['pearson_r']})")

    # ---- (b) audit of the published protocol, by quotation ----------------- #
    src = (SCRIPTS / "review5_benchmark_skill.py").read_text(encoding="utf-8").splitlines()
    src15 = (SCRIPTS / "review5_core15.py").read_text(encoding="utf-8").splitlines()

    def q(lineno):
        return {"line": lineno, "text": src[lineno - 1].rstrip()}

    def q15(needle):
        """Quote the (unique) line of review5_core15.py containing `needle`."""
        hits = [i + 1 for i, ln in enumerate(src15) if needle in ln]
        if len(hits) != 1:
            return {"line": None, "text": None,
                    "error": f"needle {needle!r} matched {len(hits)} lines"}
        return {"line": hits[0], "text": src15[hits[0] - 1].rstrip()}

    out["published_protocol_audit"] = {
        "file": "analysis/revalidation_2026-08/scripts/review5_benchmark_skill.py",
        "fold_definition": {
            "verdict": "leave-one-RECORD-out; the fold is a single site record",
            "quotes": [q(266), q(273), q(274), q(459)]},
        "feature_standardization": {
            "verdict": ("KM3 only, and fitted INSIDE the fold on the training pool "
                        "(X3_tr) -- not on pooled data. REG1/REG2 are fitted on the "
                        "raw design matrix, which is equivalent because OLS is "
                        "affine-equivariant."),
            "quotes": [q(290), q(277), q(281), q(282)]},
        "kmeans_refit": {
            "verdict": "refitted inside every fold, on the training pool only; the "
                       "cluster means are training-only as well",
            "quotes": [q(291), q(292), q(293), q(294), q(295)]},
        "core15_counts_provenance": {
            "verdict": ("the core15/core16/excl_mpls17 benchmark counts in "
                        "review5_core15.benchmark_win_counts are a RE-COUNT of the "
                        "frozen per-record flags from the all-19 record-LOO fit. The "
                        "excluded records were removed from EVALUATION only; they "
                        "stayed in every other fold's TRAINING pool."),
            "file": "analysis/revalidation_2026-08/scripts/review5_core15.py",
            "quotes": [q15("def benchmark_core15"),
                       q15("Restrict review5 benchmark win counts"),
                       q15("per = bm['benchmark']['per_record_mae']"),
                       q15("blk[sch] = {f: int(sum(bool(")],
            "consequence": ("in the published core15 count, PL-Narutowicza and "
                            "PL-Lipowa are not evaluated but still train every fold, "
                            "and each Minneapolis record still trains the other's "
                            "fold -- the leak the reviewer identified")},
        "leakage_consequence": ("because US-Minneapolis1 and US-Minneapolis2 carry "
                               "bit-identical obs_LWup, forcing_Tair, forcing_SWdown "
                               "and forcing_Qair on an identical nocturnal mask, the "
                               "record-LOO training pool for either Minneapolis fold "
                               "contains the held-out record's (y, X) pairs verbatim; "
                               "the Methods sentence asserting that within-record "
                               "temporal dependence cannot leak is false for that pair")}

    # ---- build frames ------------------------------------------------------ #
    print("\nbuilding frozen frames (imported builders) ...", flush=True)
    frames = {}
    for scheme, builder in (("TEB", bs.frame_teb), ("CLMU5", bs.frame_clmu)):
        t0 = time.time()
        frames[scheme] = {s: builder(s) for s in SITES}
        print(f"  {scheme}: {sum(frames[scheme][s]['obs'].size for s in SITES)} "
              f"nocturnal steps, {round(time.time()-t0,1)}s", flush=True)

    # frame-level confirmation that the two Minneapolis frames are duplicates
    fchk = {}
    for scheme in ("TEB", "CLMU5"):
        A = frames[scheme]["US-Minneapolis1"]
        B = frames[scheme]["US-Minneapolis2"]
        same_len = A["obs"].size == B["obs"].size
        fchk[scheme] = {
            "n": [int(A["obs"].size), int(B["obs"].size)],
            "obs_bit_identical": bool(same_len and np.array_equal(A["obs"], B["obs"])),
            "sw_bit_identical": bool(same_len and np.array_equal(A["sw"], B["sw"])),
            "ta_bit_identical": bool(same_len and np.array_equal(A["ta"], B["ta"])),
            "qa_bit_identical": bool(same_len and np.array_equal(A["qa"], B["qa"])),
            "model_max_abs_diff": (float(np.max(np.abs(A["model"] - B["model"])))
                                   if same_len else None),
            "model_mean_abs_diff": (float(np.mean(np.abs(A["model"] - B["model"])))
                                    if same_len else None)}
    out["duplicate_record_audit"]["benchmark_frame_level"] = {
        "note": ("checked on the exact arrays the benchmark consumes: the two "
                 "Minneapolis records supply identical targets and identical "
                 "predictors and differ only in the modelled LW-up, because the "
                 "sitedata morphology differs"),
        "schemes": fchk}
    print("  frame-level Minneapolis duplication:",
          {k: v["obs_bit_identical"] for k, v in fchk.items()}, flush=True)

    # ---- validation: reproduce the published record-LOO counts -------------- #
    if not args.skip_record_loo:
        print("\n[validation] recomputing the published record-LOO counts ...", flush=True)
        vrec = {}
        for scheme in ("TEB", "CLMU5"):
            b, _ = benchmarks_by_fold(frames[scheme], [[s] for s in SITES],
                                      standardize="in_fold",
                                      verbose_tag=f"recLOO-{scheme}")
            _, cnt = score(frames[scheme], b, SITES)
            pub = PUB["win_counts"][scheme]
            vrec[scheme] = {"recomputed": {f: cnt[f] for f in FLAGS},
                            "published": {f: pub[f] for f in FLAGS},
                            "match": all(cnt[f] == pub[f] for f in FLAGS)}
            print(f"  {scheme}: recomputed {[cnt[f] for f in FLAGS]} vs published "
                  f"{[pub[f] for f in FLAGS]} -> match={vrec[scheme]['match']}", flush=True)
        out["validation"]["record_loo_reproduction"] = vrec
    else:
        out["validation"]["record_loo_reproduction"] = {"skipped": True}

    # ---- (c) city-cluster holdout, eval_only convention -------------------- #
    print("\n[c] leave-one-city-cluster-out refit (16 folds) ...", flush=True)
    city_bench, city_meta, city_per, city_counts = {}, {}, {}, {}
    for scheme in ("TEB", "CLMU5"):
        b, m = benchmarks_by_fold(frames[scheme], folds_all, standardize="in_fold",
                                  verbose_tag=f"city-{scheme}")
        city_bench[scheme] = b
        city_meta[scheme] = m
        per, _ = score(frames[scheme], b, SITES)
        city_per[scheme] = per
        city_counts[scheme] = {}
        for vk, sel in RECORD_SETS.items():
            ss = sel(SITES)
            _, cnt = score(frames[scheme], b, ss)
            city_counts[scheme][vk] = cnt
            print(f"  {scheme:6s} {vk:12s} REG1={cnt['beats_REG1']}/{cnt['n']} "
                  f"REG2={cnt['beats_REG2']} KM3={cnt['beats_KM3']} "
                  f"REG2&KM3={cnt['beats_meaningful_two_REG2_KM3']}", flush=True)

    out["city_holdout"]["per_record_mae"] = city_per
    out["city_holdout"]["fold_meta"] = city_meta
    out["city_holdout"]["win_counts_eval_only"] = city_counts

    # ---- (c2) refit within each variant ------------------------------------ #
    print("\n[c2] refit with the excluded records removed from TRAINING too ...",
          flush=True)
    refit = {}
    for scheme in ("TEB", "CLMU5"):
        refit[scheme] = {}
        for vk, sel in RECORD_SETS.items():
            ss = sel(SITES)
            if vk == "all19":
                refit[scheme][vk] = dict(city_counts[scheme][vk])
                refit[scheme][vk]["identical_to_eval_only"] = True
                continue
            cl = sorted(set(CLUSTER[s] for s in ss))
            folds_v = [[s for s in ss if CLUSTER[s] == c] for c in cl]
            b, _ = benchmarks_by_fold(frames[scheme], folds_v, train_pool=ss,
                                      standardize="in_fold",
                                      verbose_tag=f"refit-{vk}-{scheme}")
            _, cnt = score(frames[scheme], b, ss)
            cnt["n_folds"] = len(folds_v)
            cnt["identical_to_eval_only"] = bool(
                all(cnt[f] == city_counts[scheme][vk][f] for f in FLAGS))
            refit[scheme][vk] = cnt
            print(f"  {scheme:6s} {vk:12s} n_folds={len(folds_v)} "
                  f"REG2={cnt['beats_REG2']}/{cnt['n']} KM3={cnt['beats_KM3']} "
                  f"REG2&KM3={cnt['beats_meaningful_two_REG2_KM3']}", flush=True)
    out["city_holdout"]["win_counts_refit_within_variant"] = refit

    # ---- (d) delta versus the published record-LOO counts ------------------ #
    pubc = STATS["review5_additions"]["core15_sensitivity"]["benchmark_win_counts"]
    delta = {}
    for vk in RECORD_SETS:
        delta[vk] = {}
        for scheme in ("TEB", "CLMU5"):
            pb = pubc[vk][scheme]
            cc = city_counts[scheme][vk]
            delta[vk][scheme] = {
                "published_record_loo": {f: int(pb[f]) for f in FLAGS if f in pb},
                "city_holdout": {f: cc[f] for f in FLAGS},
                "delta": {f: cc[f] - int(pb[f]) for f in FLAGS if f in pb}}
    out["city_holdout"]["delta_vs_published_record_loo"] = delta

    # ---- (d) protocol sensitivities ---------------------------------------- #
    print("\n[d] protocol sensitivities ...", flush=True)
    sens = {}
    # OLS affine-equivariance: standardize the REG design matrix inside the fold
    inv = {}
    for scheme in ("TEB", "CLMU5"):
        b_std, _ = benchmarks_by_fold(frames[scheme], folds_all,
                                      standardize="in_fold", reg_standardize=True)
        d1 = max(abs(b_std[s]["REG1"] - city_bench[scheme][s]["REG1"]) for s in SITES)
        d2 = max(abs(b_std[s]["REG2"] - city_bench[scheme][s]["REG2"]) for s in SITES)
        inv[scheme] = {"max_abs_REG1_MAE_change_Wm2": float(d1),
                       "max_abs_REG2_MAE_change_Wm2": float(d2)}
        print(f"  {scheme}: OLS standardization changes REG1 MAE by <= {d1:.2e}, "
              f"REG2 by <= {d2:.2e} W/m2", flush=True)
    sens["ols_affine_invariance_check"] = {
        "statement": ("standardizing the REG1/REG2 design matrix inside the training "
                      "fold leaves the predictions, and hence the MAEs and win "
                      "counts, unchanged to floating-point tolerance; standardization "
                      "is therefore only a KM3 question"),
        "per_scheme": inv}

    # KM3 standardizer fitted on pooled data (incl. held-out city) -- leakage variant
    if not args.skip_pooled_scaler:
        pooled = {}
        for scheme in ("TEB", "CLMU5"):
            b_p, _ = benchmarks_by_fold(frames[scheme], folds_all, standardize="pooled",
                                        verbose_tag=f"pooled-scaler-{scheme}")
            _, cnt = score(frames[scheme], b_p, SITES)
            dmax = max(abs(b_p[s]["KM3"] - city_bench[scheme][s]["KM3"]) for s in SITES)
            pooled[scheme] = {
                "win_counts_all19": {f: cnt[f] for f in FLAGS},
                "win_counts_all19_in_fold_scaler": {
                    f: city_counts[scheme]["all19"][f] for f in FLAGS},
                "max_abs_KM3_MAE_change_Wm2": float(dmax),
                "counts_identical": bool(all(
                    cnt[f] == city_counts[scheme]["all19"][f] for f in FLAGS))}
            print(f"  {scheme}: pooled KM3 scaler moves KM3 MAE by <= {dmax:.4f} W/m2, "
                  f"counts identical={pooled[scheme]['counts_identical']}", flush=True)
        sens["km3_pooled_vs_in_fold_standardizer"] = {
            "statement": ("the KM3 standardizer is fitted inside the fold in both the "
                          "published run and here; refitting it on the pooled corpus "
                          "(a mild leak) is reported only to show the choice is not "
                          "doing any work"),
            "per_scheme": pooled}
    out["city_holdout"]["protocol_sensitivity"] = sens

    # ---- (e) bias / structure decomposition under city folds --------------- #
    print("\n[e] bias/structure decomposition under city folds ...", flush=True)
    dec = {"note": ("bias share of MSE is a property of the model runs and is "
                    "unchanged by the fold definition; the debiased-proxy win counts "
                    "are recomputed against the CITY-HOLDOUT benchmark MAEs. Two "
                    "proxies are reported: mean|e-MBE| (the proxy used in "
                    "nocturnal_benchmark_skill.bias_decomposition) and cRMSE (the "
                    "proxy used in benchmark_error_decomposition)."),
           "per_record": {}, "by_recordset": {}}
    for scheme in ("TEB", "CLMU5"):
        per, _ = decompose(frames[scheme], city_bench[scheme], SITES)
        dec["per_record"][scheme] = per
        dec["by_recordset"][scheme] = {}
        for vk, sel in RECORD_SETS.items():
            ss = sel(SITES)
            _, agg = decompose(frames[scheme], city_bench[scheme], ss)
            dec["by_recordset"][scheme][vk] = agg
            print(f"  {scheme:6s} {vk:12s} raw_both={agg['raw_beats_both']}/{agg['n']} "
                  f"demae_both={agg['demae_beats_both']} "
                  f"crmse_both={agg['crmse_beats_both']} "
                  f"bias_share_med={agg['median_bias_share_of_MSE']:.3f}", flush=True)
    # published bias share, for the invariance claim
    dec["bias_share_unchanged_check"] = {}
    pubdec = PUB["bias_decomposition"]
    for scheme in ("TEB", "CLMU5"):
        dec["bias_share_unchanged_check"][scheme] = {
            "published_median_absMBE_over_MAE": pubdec[scheme]["median_absMBE_over_MAE"],
            "recomputed_median_absMBE_over_MAE":
                dec["by_recordset"][scheme]["all19"]["median_absMBE_over_MAE"],
            "published_raw_beats_both": pubdec[scheme]["raw_beats"]["both"],
            "published_debiased_beats_both": pubdec[scheme]["debiased_beats"]["both"]}
    out["bias_structure_decomposition"] = dec

    out["meta"]["runtime_seconds"] = round(time.time() - t_start, 1)
    outpath = ROOT / "results/review5_benchmark_cityholdout.json"
    with io.open(outpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {outpath}  ({out['meta']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
