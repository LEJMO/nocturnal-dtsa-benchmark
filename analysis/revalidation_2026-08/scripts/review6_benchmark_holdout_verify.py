# -*- coding: utf-8 -*-
"""ADVERSARIAL independent recompute of the leave-one-city-cluster-out benchmark
refit. Frames re-derived here; only the frozen CLMU alignment module is reused.
Own seeds plus the published seed. Writes scratchpad/adv_cityholdout.json only.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import io, sys, json, time, argparse
from pathlib import Path
import numpy as np, xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "\ud22c\uace0\uac00\ub2a5\uc131\ud3c9\uac00"
                       / "\uc7ac\uac80\uc99d_2026-08" / "scripts"))
import clmu_post_v11 as v11
SIG, EPS = 5.67e-8, 0.95
KM_K, KM_NINIT = 27, 10
STATS = json.load(io.open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
P = STATS["per_site"]
SITES = sorted(P)


def corpus(site):
    ds = xr.open_dataset(ROOT / ("data/urban-plumber/corpus/%s.nc" % site))
    d = dict(olw=ds["obs_LWup"].values.astype(float),
             ta=ds["forcing_Tair"].values.astype(float),
             ld=ds["forcing_LWdown"].values.astype(float),
             sw=ds["forcing_SWdown"].values.astype(float),
             qa=ds["forcing_Qair"].values.astype(float),
             night=ds["night_mask"].values.astype(bool),
             presp=ds["pre_spinup_flag"].values.astype(bool))
    ds.close()
    return d


def frame_teb(site):
    C = corpus(site)
    n = len(C["olw"])
    teb = np.loadtxt(ROOT / ("external/teb_runs/%s/output/LWU_base.txt" % site))
    nmod = min(teb.size, n - 1)
    sl = slice(1, nmod + 1)
    m = (C["night"] & ~C["presp"] & np.isfinite(C["olw"])
         & np.isfinite(C["ta"]) & np.isfinite(C["ld"]))
    mt = m[sl] & np.isfinite(teb[:nmod])
    return dict(model=teb[:nmod][mt], obs=C["olw"][sl][mt], sw=C["sw"][sl][mt],
                ta=C["ta"][sl][mt], qa=C["qa"][sl][mt])


def frame_clmu(site):
    C = v11.corpus(site)
    n = len(C["olw"])
    h = xr.open_dataset(ROOT / ("external/clmu_baseline19/%s_base.nc" % site))
    fire = np.asarray(h["FIRE"].values).reshape(-1)
    tair_m = (np.asarray(h["Tair"].values).reshape(-1) if "Tair" in h
              else np.asarray(h["TBOT"].values).reshape(-1))
    h.close()
    off, m, _ = v11.align(tair_m, C["ta"], n)
    lw = fire[off:off + m]
    night = C["night"][:m] & ~C["presp"][:m]
    okb = night & np.isfinite(C["olw"][:m]) & np.isfinite(lw)
    Cf = corpus(site)
    return dict(model=lw[okb], obs=C["olw"][:m][okb], sw=Cf["sw"][:m][okb],
                ta=C["ta"][:m][okb], qa=Cf["qa"][:m][okb])


def bench_folds(fr, folds, pool, seed, pooled_scaler=False):
    out, meta = {}, {}
    sc_pool = None
    if pooled_scaler:
        Xall = np.column_stack([np.concatenate([fr[s][k] for s in pool])
                                for k in ("sw", "ta", "qa")])
        sc_pool = StandardScaler().fit(Xall)
    for fold in folds:
        tr = [s for s in pool if s not in fold]
        y = np.concatenate([fr[s]["obs"] for s in tr])
        X1 = np.concatenate([fr[s]["sw"] for s in tr])[:, None]
        m1 = LinearRegression().fit(X1, y)
        X2 = np.column_stack([X1[:, 0],
                              np.concatenate([fr[s]["ta"] for s in tr])])
        m2 = LinearRegression().fit(X2, y)
        X3 = np.column_stack([X2[:, 0], X2[:, 1],
                              np.concatenate([fr[s]["qa"] for s in tr])])
        sc = sc_pool if pooled_scaler else StandardScaler().fit(X3)
        km = KMeans(n_clusters=KM_K, n_init=KM_NINIT, max_iter=300,
                    random_state=seed).fit(sc.transform(X3))
        lab = km.labels_
        cm = np.array([y[lab == c].mean() if np.any(lab == c) else y.mean()
                       for c in range(KM_K)])
        meta["|".join(fold)] = {"n_train_records": len(tr),
                                "n_train_steps": int(y.size),
                                "n_empty_clusters": int(sum(
                                    1 for c in range(KM_K) if not np.any(lab == c))),
                                "train": tr}
        for s in fold:
            o = fr[s]["obs"]
            p1 = m1.predict(fr[s]["sw"][:, None])
            p2 = m2.predict(np.column_stack([fr[s]["sw"], fr[s]["ta"]]))
            p3 = cm[km.predict(sc.transform(np.column_stack(
                [fr[s]["sw"], fr[s]["ta"], fr[s]["qa"]])))]
            out[s] = dict(REG1=float(np.mean(np.abs(p1 - o))),
                          REG2=float(np.mean(np.abs(p2 - o))),
                          KM3=float(np.mean(np.abs(p3 - o))),
                          REG1_pred_sd=float(np.std(p1)))
    return out, meta


def counts(fr, bmae, ss):
    w = {"beats_REG1": 0, "beats_REG2": 0, "beats_KM3": 0,
         "beats_all_three": 0, "beats_both_informative": 0}
    flags = {}
    for s in ss:
        mae = float(np.mean(np.abs(fr[s]["model"] - fr[s]["obs"])))
        b = bmae[s]
        f = {"beats_REG1": bool(mae < b["REG1"]),
             "beats_REG2": bool(mae < b["REG2"]),
             "beats_KM3": bool(mae < b["KM3"])}
        f["beats_all_three"] = bool(f["beats_REG1"] and f["beats_REG2"] and f["beats_KM3"])
        f["beats_both_informative"] = bool(f["beats_REG2"] and f["beats_KM3"])
        f["scheme_MAE"] = mae
        flags[s] = f
        for k in w:
            w[k] += int(f[k])
    return w, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    frames = {"TEB": {s: frame_teb(s) for s in SITES},
              "CLMU5": {s: frame_clmu(s) for s in SITES}}
    clmap = {}
    for s in SITES:
        clmap.setdefault(P[s]["cluster"], []).append(s)
    CITY = [sorted(v) for k, v in sorted(clmap.items())]
    REC = [[s] for s in SITES]
    no_m = [s for s in SITES if not s.startswith("US-Minneapolis")]
    SETS = {"all19": SITES, "excl_mpls17": no_m,
            "core16": [s for s in no_m if s != "PL-Lipowa"],
            "core15": [s for s in no_m if s not in ("PL-Lipowa", "PL-Narutowicza")]}
    R = {"meta": {"n_clusters": len(CITY),
                  "clusters": {k: sorted(v) for k, v in clmap.items()},
                  "frame_n": {sch: {s: int(frames[sch][s]["obs"].size) for s in SITES}
                              for sch in frames}}}
    fd = {}
    for sch in ("TEB", "CLMU5"):
        A, B = frames[sch]["US-Minneapolis1"], frames[sch]["US-Minneapolis2"]
        fd[sch] = {"n": [int(A["obs"].size), int(B["obs"].size)]}
        for k in ("obs", "sw", "ta", "qa", "model"):
            if A[k].size == B[k].size:
                d = np.abs(A[k] - B[k])
                fd[sch][k] = {"maxabsdiff": float(d.max()),
                              "meanabsdiff": float(d.mean())}
            else:
                fd[sch][k] = "len-mismatch"
    R["frame_duplication_minneapolis"] = fd
    R["pooled_steps"] = {sch: {k: int(sum(frames[sch][s]["obs"].size for s in ss))
                               for k, ss in SETS.items()} for sch in frames}
    SEEDS = [0] if a.quick else [0, 12345, 777]
    R["record_loo"] = {}
    R["city_holdout"] = {}
    for sch in ("TEB", "CLMU5"):
        R["record_loo"][sch] = {}
        R["city_holdout"][sch] = {}
        for sd in SEEDS:
            bm, _ = bench_folds(frames[sch], REC, SITES, sd)
            cs = {k: counts(frames[sch], bm, ss)[0] for k, ss in SETS.items()}
            fl = counts(frames[sch], bm, SITES)[1]
            R["record_loo"][sch][str(sd)] = {"per_record_mae": bm, "counts": cs,
                                             "flags": fl}
            bc, mt = bench_folds(frames[sch], CITY, SITES, sd)
            cc = {k: counts(frames[sch], bc, ss)[0] for k, ss in SETS.items()}
            fc = counts(frames[sch], bc, SITES)[1]
            R["city_holdout"][sch][str(sd)] = {"per_record_mae": bc, "counts": cc,
                                               "flags": fc, "fold_meta": mt}
            print("[%7.1fs] %s seed=%s recLOO=%s cityHO=%s"
                  % (time.time() - t0, sch, sd, cs, cc), flush=True)
    R["refit_within_variant"] = {}
    for sch in ("TEB", "CLMU5"):
        R["refit_within_variant"][sch] = {}
        for k, ss in SETS.items():
            folds = [[x for x in f if x in ss] for f in CITY]
            folds = [f for f in folds if f]
            bm, _ = bench_folds(frames[sch], folds, ss, 0)
            R["refit_within_variant"][sch][k] = counts(frames[sch], bm, ss)[0]
            print("[%7.1fs] refit %s %s -> %s"
                  % (time.time() - t0, sch, k, R["refit_within_variant"][sch][k]),
                  flush=True)
    R["pooled_scaler_city"] = {}
    for sch in ("TEB", "CLMU5"):
        bp, _ = bench_folds(frames[sch], CITY, SITES, 0, pooled_scaler=True)
        base = R["city_holdout"][sch]["0"]["per_record_mae"]
        R["pooled_scaler_city"][sch] = {
            "max_km3_shift": float(max(abs(bp[s]["KM3"] - base[s]["KM3"])
                                       for s in SITES)),
            "counts": {k: counts(frames[sch], bp, ss)[0] for k, ss in SETS.items()}}
        print("[%7.1fs] pooled-scaler %s %s"
              % (time.time() - t0, sch, R["pooled_scaler_city"][sch]), flush=True)
    R["reg_standardize_check"] = {}
    for sch in ("TEB", "CLMU5"):
        mx1 = mx2 = 0.0
        for fold in CITY:
            tr = [s for s in SITES if s not in fold]
            y = np.concatenate([frames[sch][s]["obs"] for s in tr])
            X1 = np.concatenate([frames[sch][s]["sw"] for s in tr])[:, None]
            X2 = np.column_stack([X1[:, 0],
                                  np.concatenate([frames[sch][s]["ta"] for s in tr])])
            s1 = StandardScaler().fit(X1)
            s2 = StandardScaler().fit(X2)
            m1 = LinearRegression().fit(X1, y)
            m1s = LinearRegression().fit(s1.transform(X1), y)
            m2 = LinearRegression().fit(X2, y)
            m2s = LinearRegression().fit(s2.transform(X2), y)
            for s in fold:
                o = frames[sch][s]["obs"]
                T1 = frames[sch][s]["sw"][:, None]
                T2 = np.column_stack([frames[sch][s]["sw"], frames[sch][s]["ta"]])
                mx1 = max(mx1, abs(np.mean(np.abs(m1.predict(T1) - o))
                                   - np.mean(np.abs(m1s.predict(s1.transform(T1)) - o))))
                mx2 = max(mx2, abs(np.mean(np.abs(m2.predict(T2) - o))
                                   - np.mean(np.abs(m2s.predict(s2.transform(T2)) - o))))
        R["reg_standardize_check"][sch] = {"max_REG1_MAE_shift": float(mx1),
                                           "max_REG2_MAE_shift": float(mx2)}
        print("[%7.1fs] regstd %s %s" % (time.time() - t0, sch,
                                         R["reg_standardize_check"][sch]), flush=True)
    R["bias_structure"] = {}
    for sch in ("TEB", "CLMU5"):
        bc = R["city_holdout"][sch]["0"]["per_record_mae"]
        pr = {}
        for s in SITES:
            e = frames[sch][s]["model"] - frames[sch][s]["obs"]
            mbe = float(e.mean())
            mae = float(np.abs(e).mean())
            demae = float(np.abs(e - mbe).mean())
            dm = frames[sch][s]["model"] - frames[sch][s]["model"].mean()
            do = frames[sch][s]["obs"] - frames[sch][s]["obs"].mean()
            crmse = float(np.sqrt(np.mean((dm - do) ** 2)))
            pr[s] = dict(MBE=mbe, MAE=mae, demae=demae, cRMSE=crmse,
                         RMSE=float(np.sqrt(np.mean(e * e))),
                         bias_share=float(mbe ** 2 / (mbe ** 2 + crmse ** 2)),
                         absMBE_over_MAE=float(abs(mbe) / mae),
                         deb_REG2=bool(demae < bc[s]["REG2"]),
                         deb_KM3=bool(demae < bc[s]["KM3"]),
                         crm_REG2=bool(crmse < bc[s]["REG2"]),
                         crm_KM3=bool(crmse < bc[s]["KM3"]))
        by = {}
        for k, ss in SETS.items():
            by[k] = {"deb_REG2": int(sum(pr[s]["deb_REG2"] for s in ss)),
                     "deb_KM3": int(sum(pr[s]["deb_KM3"] for s in ss)),
                     "deb_both": int(sum(pr[s]["deb_REG2"] and pr[s]["deb_KM3"]
                                         for s in ss)),
                     "crm_REG2": int(sum(pr[s]["crm_REG2"] for s in ss)),
                     "crm_KM3": int(sum(pr[s]["crm_KM3"] for s in ss)),
                     "crm_both": int(sum(pr[s]["crm_REG2"] and pr[s]["crm_KM3"]
                                         for s in ss))}
        R["bias_structure"][sch] = {
            "per_record": pr, "by_set": by,
            "median_bias_share": float(np.median([pr[s]["bias_share"] for s in SITES])),
            "median_absMBE_over_MAE": float(np.median(
                [pr[s]["absMBE_over_MAE"] for s in SITES])),
            "n_bias_dominated": int(sum(pr[s]["bias_share"] > 0.5 for s in SITES)),
            "median_cRMSE": float(np.median([pr[s]["cRMSE"] for s in SITES])),
            "corr_MAE_absMBE": float(np.corrcoef(
                [pr[s]["MAE"] for s in SITES],
                [abs(pr[s]["MBE"]) for s in SITES])[0, 1])}
        print("[%7.1fs] bias %s share=%.4f by_set=%s"
              % (time.time() - t0, sch, R["bias_structure"][sch]["median_bias_share"],
                 by), flush=True)
    R["meta"]["runtime_s"] = time.time() - t0
    out = Path(os.environ["SCR"]) / "adv_cityholdout.json"
    json.dump(R, io.open(out, "w", encoding="utf-8"), indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
