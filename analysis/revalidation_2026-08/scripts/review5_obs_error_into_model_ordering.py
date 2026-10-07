# -*- coding: utf-8 -*-
"""Review issue 3, the leg that review5_obs_error_into_model.py did not cover:
carry the SAME observation-error scenarios into the ORDERING claim
(P(r_{m|o} <= 0) per scheme) and into the PAIRED scheme difference d_r.

Why this is needed.  The manuscript's model-side claims are three, not two:
  (i)   slope      -- beta_{m|o} far below 1                        (Sect. 3.2, l. 806)
  (ii)  amplitude  -- s_m/s_o < 1                                   (Sect. 3.2, l. 776)
  (iii) ordering   -- TEB's r_{m|o} separable from zero on all records
                      and excluding Minneapolis, "marginal in the core
                      (0.051)"                                      (Sect. 3.2, l. 802-805)
review5_obs_error_into_model.py propagated the scenarios into (i) and (ii) only.
An observation-error field moves r_{m|o} directly -- that is the mechanism the
reviewer named -- so (iii) has to be propagated too, and it is the leg that
actually breaks.

Conventions are those of review5_scheme_ordering_paired.py / review5_core15.py:
resampling unit = city cluster, NDRAW = 100 000, SEED = 20260916, sd ddof = 1,
guard len(unique(o[ii])) < 3 evaluated on the UNPERTURBED observed means so the
same resamples are used in every scenario (the comparison stays paired).  The
published one-sided P(r<=0) values are reproduced exactly (see
reproduction_checks) before any scenario is applied.

Model record means are never perturbed; only per_site.dtsa_std is.

Reads results/paper_stats_v1.json READ-ONLY.
Writes results/review5_obs_error_ordering.json.
"""
import io, json, sys, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NDRAW = 100_000
NERR = 50_000
SEED = 20260916
SIG, EPS, TREF = 5.67e-8, 0.95, 285.0
K_PER_W = 1.0 / (4.0 * EPS * SIG * TREF ** 3)
B_GRID = (0.25, 0.5, 0.75, 1.0)
SIGMA_GRID = (2.0, 4.0, 5.0, 10.0)


def main():
    sp = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
    PSD = sp["per_site"]
    S = sorted(PSD)
    obs = np.array([PSD[s]["dtsa_std"] for s in S], float)
    teb = np.array([PSD[s]["teb_dtsa"] for s in S], float)
    clm = np.array([PSD[s]["clmu_dtsa"] for s in S], float)
    alb = np.array([PSD[s]["albedo"] for s in S], float)
    clus = np.array([PSD[s].get("cluster", s) for s in S])
    SOP = sp["review5_additions"]["scheme_ordering_paired"]
    C15 = sp["review5_additions"]["core15_sensitivity"]["model_side"]["core15"]

    MASK = {
        "all19": np.ones(len(S), bool),
        "excl_mpls17": np.array([not s.startswith("US-Minneapolis") for s in S]),
        "core16": np.array([(not s.startswith("US-Minneapolis")) and s != "PL-Lipowa" for s in S]),
        "core15": np.array([(not s.startswith("US-Minneapolis"))
                            and s not in ("PL-Lipowa", "PL-Narutowicza") for s in S]),
    }
    PUB = {"all19": SOP["all19"], "excl_mpls17": SOP["excl_mpls17"],
           "core16": SOP["core16"], "core15": C15}
    SCH = {"TEB": teb, "CLMU5": clm}

    out = {"_merge_target": "review5_additions.obs_error_ordering",
           "meta": {"script": "review5_obs_error_into_model_ordering.py",
                    "task": ("review issue 3, ordering leg: the obs-error scenarios carried into "
                             "P(r_{m|o}<=0) and the paired d_r, which "
                             "review5_obs_error_into_model.py did not cover"),
                    "draws": NDRAW, "err_draws": NERR, "seed": SEED,
                    "boot_unit": "city cluster",
                    "boot_provenance": ("identical RNG call sequence to "
                                        "review5_scheme_ordering_paired.py, stored as per-record "
                                        "multiplicity vectors; verified against the published "
                                        "one-sided P(r<=0) in reproduction_checks"),
                    "model_values_fixed": "only per_site.dtsa_std is perturbed",
                    "K_per_Wm2": K_PER_W,
                    "guard": "len(unique(o[ii]))<3 evaluated on the UNPERTURBED means (stays paired)"}}

    def multiplicity(mask):
        """reproduce the published per-draw rng.choice sequence, as multiplicities"""
        cl = clus[mask]
        uc = sorted(set(cl.tolist()))
        idx_by_c = {u: np.where(cl == u)[0] for u in uc}
        rng = np.random.default_rng(SEED)
        n = int(mask.sum())
        C = np.zeros((NDRAW, n))
        keep = np.zeros(NDRAW, bool)
        o = obs[mask]
        for j in range(NDRAW):
            pick = rng.choice(len(uc), size=len(uc), replace=True)
            ii = np.concatenate([idx_by_c[uc[p]] for p in pick])
            if len(np.unique(o[ii])) < 3:
                continue
            keep[j] = True
            np.add.at(C[j], ii, 1.0)
        return C[keep], len(uc), int((~keep).sum())

    def rboot(C, o, m):
        """multiplicity-weighted Pearson r; o is (n,) or (ndraw,n)"""
        N = C.sum(1)
        o2 = np.atleast_2d(o)
        if o2.shape[0] == 1:
            o2 = np.broadcast_to(o2, C.shape)
        mo = (C * o2).sum(1) / N
        mm = (C * m[None, :]).sum(1) / N
        do = o2 - mo[:, None]
        dm = m[None, :] - mm[:, None]
        voo = (C * do * do).sum(1) / (N - 1)
        vmm = (C * dm * dm).sum(1) / (N - 1)
        vom = (C * do * dm).sum(1) / (N - 1)
        return vom / np.sqrt(voo * vmm)

    BOOT, rep = {}, {}
    for rs, mk in MASK.items():
        C, ncl, dropped = multiplicity(mk)
        BOOT[rs] = C
        o = obs[mk]
        rep[rs] = {"n": int(mk.sum()), "n_clusters": ncl, "draws_used": int(C.shape[0]),
                   "draws_dropped": dropped}
        for sch, mv in SCH.items():
            rr = rboot(C, o, mv[mk])
            pub = PUB[rs]["one_sided"]["%s_P_r_le_0" % sch]
            got = float(np.mean(rr <= 0))
            rep[rs][sch] = {"published_P_r_le_0": pub, "recomputed_P_r_le_0": got,
                            "abs_diff": abs(pub - got)}
        dpub = PUB[rs]["paired_difference_TEB_minus_CLMU5"]["r"]["P_le_0"]
        dgot = float(np.mean((rboot(C, o, teb[mk]) - rboot(C, o, clm[mk])) <= 0))
        rep[rs]["paired_d_r"] = {"published_P_le_0": dpub, "recomputed_P_le_0": dgot,
                                 "abs_diff": abs(dpub - dgot)}
    out["reproduction_checks"] = rep
    worst = max(max(rep[rs][s]["abs_diff"] for s in SCH) for rs in MASK)
    worst = max(worst, max(rep[rs]["paired_d_r"]["abs_diff"] for rs in MASK))
    print("[check] max |published - recomputed| P(r<=0) over all cells = %.3e" % worst)
    if worst > 1e-12:
        raise SystemExit("published bootstrap stream NOT reproduced -- aborting")

    def cell(rs, o):
        mk = MASK[rs]
        C = BOOT[rs]
        rt, rc = rboot(C, o, teb[mk]), rboot(C, o, clm[mk])
        d = rt - rc
        return {"r_TEB_point": float(np.corrcoef(o, teb[mk])[0, 1]),
                "r_CLMU5_point": float(np.corrcoef(o, clm[mk])[0, 1]),
                "P_r_le_0_TEB": float(np.mean(rt <= 0)),
                "P_r_le_0_CLMU5": float(np.mean(rc <= 0)),
                "d_r_point": float(np.corrcoef(o, teb[mk])[0, 1] - np.corrcoef(o, clm[mk])[0, 1]),
                "d_r_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "P_d_r_le_0": float(np.mean(d <= 0))}

    # ---- baseline
    out["baseline"] = {rs: cell(rs, obs[MASK[rs]]) for rs in MASK}

    # ---- albedo-aligned step (the construction that defeats the observed association)
    al = {}
    print("\n=== albedo-aligned step: P(r_{m|o} <= 0)")
    for rs, mk in MASK.items():
        a = alb[mk]
        sgn = np.sign(a - a.mean())
        al[rs] = {"sign_vector": {s: float(v) for s, v in zip(np.array(S)[mk], sgn)},
                  "amplitudes": {}}
        for B in B_GRID:
            al[rs]["amplitudes"]["B_%.2fK" % B] = cell(rs, obs[mk] + B * sgn)
        b = out["baseline"][rs]
        print("  %-12s TEB %.4f -> %s | CLMU %.4f -> %s"
              % (rs, b["P_r_le_0_TEB"],
                 " / ".join("%.4f" % al[rs]["amplitudes"]["B_%.2fK" % B]["P_r_le_0_TEB"] for B in B_GRID),
                 b["P_r_le_0_CLMU5"],
                 " / ".join("%.4f" % al[rs]["amplitudes"]["B_%.2fK" % B]["P_r_le_0_CLMU5"] for B in B_GRID)))
    out["aligned_step"] = al

    # ---- independent random LW_up error, joint with the resampling (draw j <-> resample j)
    rnd = {}
    print("\n=== independent random error, joint: P(r_{m|o} <= 0)")
    for rs, mk in MASK.items():
        n = int(mk.sum())
        C = BOOT[rs][:NERR]
        rnd[rs] = {}
        for sg in SIGMA_GRID:
            tag = "random_independent_ordering|%s|%g" % (rs, sg)
            rng = np.random.default_rng(SEED + zlib.crc32(tag.encode("utf-8")))
            X = obs[mk][None, :] + rng.standard_normal((NERR, n)) * sg * K_PER_W
            rt, rc = rboot(C, X, teb[mk]), rboot(C, X, clm[mk])
            xc = X - X.mean(1, keepdims=True)

            def ptr(m):
                mc = m - m.mean()
                return (xc @ mc) / np.sqrt((xc * xc).sum(1) * (mc @ mc))
            rnd[rs]["sigma_%gWm2" % sg] = {
                "sigma_Wm2": sg, "sigma_K": sg * K_PER_W,
                "r_TEB_median_over_error_draws": float(np.median(ptr(teb[mk]))),
                "r_CLMU5_median_over_error_draws": float(np.median(ptr(clm[mk]))),
                "joint_P_r_le_0_TEB": float(np.mean(rt <= 0)),
                "joint_P_r_le_0_CLMU5": float(np.mean(rc <= 0)),
                "joint_P_d_r_le_0": float(np.mean((rt - rc) <= 0))}
        print("  %-12s TEB %s | CLMU %s"
              % (rs, " / ".join("%.4f" % rnd[rs]["sigma_%gWm2" % s]["joint_P_r_le_0_TEB"]
                                for s in SIGMA_GRID),
                 " / ".join("%.4f" % rnd[rs]["sigma_%gWm2" % s]["joint_P_r_le_0_CLMU5"]
                            for s in SIGMA_GRID)))
    out["random_independent"] = rnd

    # ---- headline
    hl = {}
    for rs in MASK:
        b = out["baseline"][rs]["P_r_le_0_TEB"]
        mx = max([al[rs]["amplitudes"][k]["P_r_le_0_TEB"] for k in al[rs]["amplitudes"]]
                 + [rnd[rs][k]["joint_P_r_le_0_TEB"] for k in rnd[rs]])
        hl[rs] = {"baseline_P_r_le_0_TEB": b, "max_over_scenarios_P_r_le_0_TEB": mx,
                  "crosses_0p05": bool(mx > 0.05)}
    out["headline"] = {
        "question": "does TEB's ordering skill stay separable from zero under the obs-error scenarios?",
        "per_record_set": hl,
        "verdict": ("all19 stays below 0.05 under the albedo-aligned field at every tested "
                    "amplitude and under random error to sigma_E = 5 W/m2 (0.037), crossing only "
                    "at sigma_E = 10 W/m2 (0.090); excl_mpls17 stays below 0.05 under the aligned "
                    "field (max 0.049 at B = 1 K) but crosses under random error from sigma_E = 4 "
                    "W/m2 (0.061); core16 and core15, where the published values are already "
                    "marginal (0.051, 0.074), cross 0.05 under EVERY scenario tested, reaching "
                    "0.378 and 0.415. The ordering leg of the model conclusion is therefore the "
                    "leg that is conditional on the published means in the conservative core -- "
                    "not the slope, and more severely than the amplitude.")}
    for rs in MASK:
        print("  %-12s TEB baseline %.4f -> max over scenarios %.4f  crosses 0.05: %s"
              % (rs, hl[rs]["baseline_P_r_le_0_TEB"], hl[rs]["max_over_scenarios_P_r_le_0_TEB"],
                 hl[rs]["crosses_0p05"]))

    p = ROOT / "results/review5_obs_error_ordering.json"
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nwrote", p)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
