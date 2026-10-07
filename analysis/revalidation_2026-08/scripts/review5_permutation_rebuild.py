# -*- coding: utf-8 -*-
"""Round-5 addition, TASK C: rebuild of the cluster-permutation inference.

Writes results/review5_permutation-rebuild.json.  It does NOT touch
results/paper_stats_v1.json (the lead merges it under
`review5_additions.permutation_v11`).

Why this script exists
----------------------
(i)  main.tex calls the headline p = 2e-4 the permutation "resolution floor".
     With nperm = 10 000 and p = (b+1)/(N+1) the floor is 1/10001 = 9.999e-5.
     2e-4 is TWICE the floor and corresponds to b = 1 exceedance.  Every
     stored p in paper_stats_v1.json inverts to an integer b (see
     `floor_audit` below), which confirms the formula and the claim.
(ii) TWO different tests are both called "cluster permutation":
       * paper_stats.py:290  `cluster_permutation`  -> scheme CM (city-mean)
       * review2_additions.py:60 `cluster_perm_p`   -> scheme SM (smeared)
     and main.tex quotes all19/excl17 from CM but core16 from SM, without
     saying so.  Their observed statistics are different numbers.

What is computed
----------------
1. floor_audit            - inverts every stored p to its exceedance count.
2. scheme_documentation   - exact description + exchangeability assessment.
3. albedo_offset          - schemes CB (primary, restricted equal-size city-
                            block record-level), CB_ws (CB + within-block
                            shuffle), CM (city-mean companion), SM (legacy
                            review2 scheme, for provenance only), each in
                            all19 / excl_mpls17 / core16 at N = 2e7, with MC
                            standard error, Clopper-Pearson interval on the
                            exceedance rate, and the number of distinct
                            attainable arrangements.
4. freedman_lane          - TEXTBOOK Freedman-Lane (1983) for albedo given
                            sky emissivity, in ALL THREE variants, with the
                            same three permutation schemes.
5. headline_slope_bootstrap - nested (two-stage) cluster bootstrap with an
                            OLS refit per resample, plus the one-stage
                            cluster bootstrap the manuscript currently
                            quotes, both with exact drop accounting.

Everything observation-side is recomputed from the raw corpus NetCDF and the
sitedata CSVs, not read from paper_stats_v1.json; the JSON is opened
read-only once, purely to cross-check.  No CLMU-derived or TEB-derived
quantity is used anywhere in this script.

CONSTANTS: SIG = 5.67e-8 (the project convention, NOT the exact
Stefan-Boltzmann value) and EPS = 0.95, to stay commensurate with Table 1.
"""
import csv
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import beta as beta_dist
from scipy.stats import pearsonr
from scipy.stats import t as t_dist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[3]
STATS = ROOT / "results" / "paper_stats_v1.json"
OUT = ROOT / "results" / "review5_permutation-rebuild.json"

SIG = 5.67e-8          # project convention (NOT 5.670374419e-8)
EPS = 0.95
SEED = 20260916
NPERM = 20_000_000
CHUNK = 500_000
NBOOT = 200_000
if len(sys.argv) > 1:          # smoke-test override: review5_permutation_rebuild.py 20000
    NPERM = int(sys.argv[1])
    CHUNK = min(CHUNK, NPERM)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- data ------
CLUSTER = {"FI-Kumpula": "Helsinki", "FI-Torni": "Helsinki",
           "PL-Lipowa": "Lodz", "PL-Narutowicza": "Lodz",
           "US-Minneapolis1": "Minneapolis", "US-Minneapolis2": "Minneapolis"}


def sitedata(site, key):
    p = ROOT / "data" / "urban-plumber" / "FullCollection" / site / f"{site}_sitedata_v1.csv"
    for row in csv.DictReader(p.open(encoding="utf-8")):
        if row["parameter"] == key:
            try:
                return float(row["value"])
            except ValueError:
                return float("nan")
    return float("nan")


stats_json = json.load(open(STATS, encoding="utf-8"))     # read-only
SITES = sorted(stats_json["per_site"])

REC = {}
for s in SITES:
    ds = xr.open_dataset(ROOT / f"data/urban-plumber/corpus/{s}.nc")
    night = ds["night_mask"].values.astype(bool)
    spin = ds["pre_spinup_flag"].values.astype(bool)
    ta = ds["forcing_Tair"].values
    ld = ds["forcing_LWdown"].values
    lw = ds["obs_LWup"].values          # corpus copy already _FillValue-cleaned to NaN
    ds.close()
    m = night & ~spin & np.isfinite(lw) & np.isfinite(ta) & np.isfinite(ld)
    ts = ((lw[m] - (1 - EPS) * ld[m]) / (EPS * SIG)) ** 0.25
    REC[s] = dict(
        dtsa=float(np.mean(ts - ta[m])),
        eps_sky=float(np.mean(ld[m] / (SIG * ta[m] ** 4))),
        albedo=sitedata(s, "average_albedo_at_midday"),
        cluster=CLUSTER.get(s, s),
        n_night=int(m.sum()),
    )

# cross-check against the (read-only) ledger
xcheck = {"max_abs_diff_dtsa_std": 0.0, "max_abs_diff_albedo": 0.0,
          "max_abs_diff_eps_sky": 0.0, "cluster_mismatches": []}
epsJ = stats_json["review3_additions"]["forcing_covariates"]["covariates"][
    "sky_emissivity"]["per_site_mean"]
for s in SITES:
    P = stats_json["per_site"][s]
    xcheck["max_abs_diff_dtsa_std"] = max(xcheck["max_abs_diff_dtsa_std"],
                                          abs(REC[s]["dtsa"] - P["dtsa_std"]))
    xcheck["max_abs_diff_albedo"] = max(xcheck["max_abs_diff_albedo"],
                                        abs(REC[s]["albedo"] - P["albedo"]))
    xcheck["max_abs_diff_eps_sky"] = max(xcheck["max_abs_diff_eps_sky"],
                                         abs(REC[s]["eps_sky"] - epsJ[s]))
    if REC[s]["cluster"] != P["cluster"]:
        xcheck["cluster_mismatches"].append(s)

VARIANTS = {
    "all19": SITES,
    "excl_mpls17": [s for s in SITES if not s.startswith("US-Minneapolis")],
    "core16": [s for s in SITES
               if not s.startswith("US-Minneapolis") and s != "PL-Lipowa"],
}


def vec(ss, field):
    return np.array([REC[s][field] for s in ss], float)


def blocks_of(ss):
    """City blocks as index arrays into ss, canonical (site-name-sorted) order."""
    by = {}
    for i, s in enumerate(ss):
        by.setdefault(REC[s]["cluster"], []).append(i)
    return [np.array(sorted(v), dtype=np.int64) for _, v in sorted(by.items())]


# ------------------------------------------------ 1. floor audit ------------
def invert_p(p, nperm):
    """(b+1)/(N+1) -> b."""
    b = p * (nperm + 1) - 1
    return int(round(b)), abs(b - round(b))


STORED = [
    ("cluster_permutation.all.p_perm",
     stats_json["cluster_permutation"]["all"]["p_perm"], 10000, "CM"),
    ("cluster_permutation.excl_mpls.p_perm",
     stats_json["cluster_permutation"]["excl_mpls"]["p_perm"], 10000, "CM"),
    ("review2_additions.core_variant_permutation.p_perm",
     stats_json["review2_additions"]["core_variant_permutation"]["p_perm"], 10000, "SM"),
    ("review3b_additions.albedo_given_skyemissivity_clusteraware.all19.cluster_perm_p",
     stats_json["review3b_additions"]["albedo_given_skyemissivity_clusteraware"]
     ["all19"]["cluster_perm_p"], 10000, "SM-partial"),
    ("review3b_additions.albedo_given_skyemissivity_clusteraware.excl_mpls17.cluster_perm_p",
     stats_json["review3b_additions"]["albedo_given_skyemissivity_clusteraware"]
     ["excl_mpls17"]["cluster_perm_p"], 10000, "SM-partial"),
    ("review4_additions.albedo_given_skyemissivity_freedman_lane.all19.freedman_lane_cluster_p",
     stats_json["review4_additions"]["albedo_given_skyemissivity_freedman_lane"]
     ["all19"]["freedman_lane_cluster_p"], 10000, "FL-hybrid"),
    ("review4_additions.albedo_given_skyemissivity_freedman_lane.excl_mpls17.freedman_lane_cluster_p",
     stats_json["review4_additions"]["albedo_given_skyemissivity_freedman_lane"]
     ["excl_mpls17"]["freedman_lane_cluster_p"], 10000, "FL-hybrid"),
]
floor_audit = {
    "formula": "p = (b+1)/(N+1)",
    "nperm_used_by_all_stored_values": 10000,
    "true_resolution_floor_at_nperm_10000": 1.0 / 10001.0,
    "manuscript_claim": ("main.tex L387-390 calls the headline 2e-4 the "
                         "'resolution floor, i.e. p<1e-3'; it is TWICE the "
                         "floor and equals b=1 exceedance."),
    "stored_values": [
        {"key": k, "p_stored": p, "nperm": n, "exceedances_b": invert_p(p, n)[0],
         "inversion_residual": invert_p(p, n)[1], "scheme": sch,
         "is_floor": invert_p(p, n)[0] == 0}
        for k, p, n, sch in STORED],
    "all_invert_to_integers": all(invert_p(p, n)[1] < 1e-9 for _, p, n, _ in STORED),
    "new_resolution_floor_at_nperm_2e7": 1.0 / (NPERM + 1),
}

# ------------------------------------------ 2. permutation machinery --------
def arrangement_count(blocks, within=False):
    """Distinct arrangements of a restricted equal-size block permutation."""
    from collections import Counter
    sizes = Counter(len(b) for b in blocks)
    n = 1
    for s, k in sizes.items():
        n *= math.factorial(k)
    if within:
        for b in blocks:
            n *= math.factorial(len(b))
    return n


def frozen_blocks(blocks):
    """Blocks whose size class has only one member -> immovable under the
    equal-size restriction."""
    from collections import Counter
    sizes = Counter(len(b) for b in blocks)
    return [int(len(b)) for b in blocks if sizes[len(b)] == 1]


class BlockPermuter:
    """Restricted equal-size city-block permutation index generator."""

    def __init__(self, blocks, n, within=False):
        from collections import defaultdict
        grp = defaultdict(list)
        for b in blocks:
            grp[len(b)].append(b)
        self.classes = []
        for size, bl in sorted(grp.items()):
            slots = np.stack(bl)                      # (k, size) destinations
            self.classes.append((size, slots))
        self.n = n
        self.within = within
        self._tiles = {}

    def idx(self, B):
        IDX = np.empty((B, self.n), dtype=np.int32)
        for ci, (size, slots) in enumerate(self.classes):
            k = slots.shape[0]
            key = (ci, B)
            if key not in self._tiles:
                self._tiles[key] = np.tile(np.arange(k, dtype=np.int32), (B, 1))
            t = self._tiles[key]
            rng.permuted(t, axis=1, out=t)
            src = slots[t]                            # (B, k, size)
            if self.within and size > 1:
                wkey = ("w", ci, B)
                if wkey not in self._tiles:
                    self._tiles[wkey] = np.tile(
                        np.arange(size, dtype=np.int32), (B, k, 1))
                w = self._tiles[wkey]
                rng.permuted(w, axis=2, out=w)
                src = np.take_along_axis(src, w, axis=2)
            IDX[:, slots.ravel()] = src.reshape(B, k * size)
        return IDX


def mc_summary(b, N):
    p = (b + 1) / (N + 1)
    phat = b / N
    se = math.sqrt(max(phat, 1.0 / N) * (1 - max(phat, 1.0 / N)) / N)
    # Clopper-Pearson 95% on the underlying exceedance rate
    lo = 0.0 if b == 0 else float(beta_dist.ppf(0.025, b, N - b + 1))
    hi = 1.0 if b == N else float(beta_dist.ppf(0.975, b + 1, N - b))
    # digits actually resolved: largest d with 2*se < 0.5*10^(floor(log10 p)-d+1)
    digits = 0
    if p > 0:
        for d in range(1, 7):
            unit = 10.0 ** (math.floor(math.log10(p)) - d + 1)
            if 2 * se < 0.5 * unit:
                digits = d
            else:
                break
    return dict(exceedances_b=int(b), nperm=int(N), p=p,
                mc_standard_error=se, p_plus_minus_2se=[p - 2 * se, p + 2 * se],
                exceedance_rate_cp95=[lo, hi],
                resolved_significant_digits=max(digits, 1))


def perm_test_r(x, y, blocks, N=NPERM, within=False, chunk=CHUNK):
    """Two-sided |Pearson r| permutation test.

    Only the numerator sum(x_perm * yc) varies: permuting x leaves Sxx and Syy
    invariant, so r_perm = (x_perm . yc) / sqrt(Sxx*Syy).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    xc = x - x.mean(); yc = y - y.mean()
    Sxx = float(xc @ xc); Syy = float(yc @ yc)
    denom = math.sqrt(Sxx * Syy)
    r_obs = float(xc @ yc) / denom
    thr = abs(r_obs) * denom - 1e-9 * denom      # compare numerators directly
    perm = BlockPermuter(blocks, n, within=within)
    b = 0
    done = 0
    t0 = time.time()
    while done < N:
        B = min(chunk, N - done)
        IDX = perm.idx(B)
        num = (x[IDX] * yc).sum(axis=1)
        b += int(np.count_nonzero(np.abs(num) >= thr))
        done += B
    out = mc_summary(b, N)
    out.update(r_observed=r_obs, n=int(n), n_blocks=len(blocks),
               distinct_arrangements=arrangement_count(blocks, within),
               frozen_block_sizes=frozen_blocks(blocks),
               wall_seconds=round(time.time() - t0, 1))
    return out


# --------------------------------- legacy scheme SM (review2_additions) -----
def sm_perm_test(ss, N=NPERM, chunk=CHUNK):
    """review2_additions.py:60 -- each record's albedo is REPLACED by its
    city-mean albedo (record-level vector with duplicated values), the m
    city-mean albedos are then permuted across cities and re-expanded.  The
    observed statistic is therefore NOT the headline record-level r."""
    alb = vec(ss, "albedo"); dt = vec(ss, "dtsa")
    cl = [REC[s]["cluster"] for s in ss]
    uniq = sorted(set(cl))
    xc = np.array([np.mean([alb[i] for i, c in enumerate(cl) if c == u]) for u in uniq])
    lab = np.array([uniq.index(c) for c in cl])
    xv = xc[lab]
    yc = dt - dt.mean()
    r_obs = float(pearsonr(xv, dt)[0])
    b = 0; done = 0
    tile = np.tile(np.arange(len(uniq), dtype=np.int32), (chunk, 1))
    while done < N:
        B = min(chunk, N - done)
        t = tile[:B]
        rng.permuted(t, axis=1, out=t)
        xp = xc[t][:, lab]                         # (B, n)
        xpc = xp - xp.mean(axis=1, keepdims=True)
        num = (xpc * yc).sum(axis=1)
        sxx = (xpc * xpc).sum(axis=1)
        rp = num / np.sqrt(sxx * float(yc @ yc))
        b += int(np.count_nonzero(np.abs(rp) >= abs(r_obs) - 1e-12))
        done += B
    out = mc_summary(b, N)
    out.update(r_observed_smeared=r_obs, n=int(len(ss)), n_cities=len(uniq),
               distinct_arrangements=math.factorial(len(uniq)),
               note=("statistic is a RECORD-level r computed with city-mean-"
                     "smeared albedo; it is neither the headline record-level "
                     "r nor the city-mean r"))
    return out


# --------------------------------------------- 3. albedo-offset tests -------
albedo_offset = {}
for v, ss in VARIANTS.items():
    alb = vec(ss, "albedo"); dt = vec(ss, "dtsa")
    bl = blocks_of(ss)
    cl = sorted({REC[s]["cluster"] for s in ss})
    # city-mean points
    cm_x = np.array([np.mean([REC[s]["albedo"] for s in ss if REC[s]["cluster"] == c])
                     for c in cl])
    cm_y = np.array([np.mean([REC[s]["dtsa"] for s in ss if REC[s]["cluster"] == c])
                     for c in cl])
    cm_blocks = [np.array([i]) for i in range(len(cl))]
    row = {
        "n_records": len(ss), "n_cities": len(cl),
        "block_sizes": sorted(int(len(b)) for b in bl),
        "r_record_level": float(pearsonr(alb, dt)[0]),
        "r_city_mean": float(pearsonr(cm_x, cm_y)[0]),
        "CB_city_block_restricted": perm_test_r(alb, dt, bl),
        "CB_ws_city_block_within_shuffle": perm_test_r(alb, dt, bl, within=True),
        "CM_city_mean": perm_test_r(cm_x, cm_y, cm_blocks),
        "SM_legacy_review2_smeared": sm_perm_test(ss),
    }
    albedo_offset[v] = row
    print(f"[albedo] {v}: r_rec={row['r_record_level']:+.4f} "
          f"CB p={row['CB_city_block_restricted']['p']:.3e} | "
          f"r_city={row['r_city_mean']:+.4f} "
          f"CM p={row['CM_city_mean']['p']:.3e} | "
          f"SM p={row['SM_legacy_review2_smeared']['p']:.3e}", flush=True)

# ---------------------------------------- 4. textbook Freedman-Lane --------
def freedman_lane(ss, scheme, N=NPERM, chunk=CHUNK):
    """Freedman & Lane (1983) for H0: beta_albedo = 0 in
         dTsa = g0 + g1*eps_sky + beta*albedo + e.
    1. reduced fit  dTsa ~ [1, eps]  -> fitted f, residuals e
    2. permute e (restricted city-block, or city-mean scheme)
    3. y* = f + e*        4. refit full model, statistic |t(beta)|
    Because f lies in span(1, eps) and is orthogonal to the full-model
    residual space, w'f = 0 and f'Mf = 0, so t* depends on e* alone:
        t* = (w'e*) / sqrt( (e*'M e*)/(n-3) * c33 ).
    The identity permutation reproduces t_obs exactly.
    """
    if scheme == "CM":
        cl = sorted({REC[s]["cluster"] for s in ss})
        alb = np.array([np.mean([REC[s]["albedo"] for s in ss
                                 if REC[s]["cluster"] == c]) for c in cl])
        eps = np.array([np.mean([REC[s]["eps_sky"] for s in ss
                                 if REC[s]["cluster"] == c]) for c in cl])
        y = np.array([np.mean([REC[s]["dtsa"] for s in ss
                               if REC[s]["cluster"] == c]) for c in cl])
        blocks = [np.array([i]) for i in range(len(cl))]
        within = False
    else:
        alb = vec(ss, "albedo"); eps = vec(ss, "eps_sky"); y = vec(ss, "dtsa")
        blocks = blocks_of(ss)
        within = (scheme == "CB_ws")
    n = y.size
    if scheme == "FREE":           # no clustering at all, reference only
        blocks = [np.array([i]) for i in range(n)]
        within = False
    Z = np.column_stack([np.ones(n), eps])
    X = np.column_stack([np.ones(n), eps, alb])
    e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    XtXi = np.linalg.inv(X.T @ X)
    W = XtXi @ X.T
    w = W[2]                                   # beta_albedo = w'y
    c33 = float(XtXi[2, 2])
    M = np.eye(n) - X @ W
    df = n - 3
    num_obs = float(w @ e)
    rss_obs = float(e @ M @ e)
    t_obs = num_obs / math.sqrt(rss_obs / df * c33)
    # partial correlation of y and albedo given eps (same sign, monotone in |t|)
    ra = alb - Z @ np.linalg.lstsq(Z, alb, rcond=None)[0]
    partial_r = float(pearsonr(ra, e)[0])
    perm = BlockPermuter(blocks, n, within=within)
    b = 0; done = 0
    t0 = time.time()
    while done < N:
        B = min(chunk, N - done)
        E = e[perm.idx(B)]
        num = E @ w
        rss = np.einsum("bi,bi->b", E @ M, E)
        tstat = num / np.sqrt(rss / df * c33)
        b += int(np.count_nonzero(np.abs(tstat) >= abs(t_obs) - 1e-9))
        done += B
    out = mc_summary(b, N)
    out.update(n=int(n), df=int(df), partial_r=partial_r, t_observed=t_obs,
               parametric_two_sided_p=float(2 * t_dist.sf(abs(t_obs), df)),
               beta_albedo_K_per_unit=num_obs,
               distinct_arrangements=arrangement_count(blocks, within),
               frozen_block_sizes=frozen_blocks(blocks),
               wall_seconds=round(time.time() - t0, 1))
    return out


def legacy_hybrid_partial(ss, N=NPERM, chunk=CHUNK):
    """The review4_additions.py:fl_perm scheme, recomputed at N=2e7 so the
    supersede is justified numerically rather than asserted.  Residualize both
    dTsa and albedo on eps_sky, then permute the CITY-MEAN albedo residuals
    re-expanded to record level; statistic |Pearson r| against the record-level
    dTsa residuals.  Not calibrated: the permuted predictor has smaller
    variance than the observed one."""
    alb = vec(ss, "albedo"); eps = vec(ss, "eps_sky"); y = vec(ss, "dtsa")
    n = y.size
    Z = np.column_stack([np.ones(n), eps])
    ra = alb - Z @ np.linalg.lstsq(Z, alb, rcond=None)[0]
    rd = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r_obs = float(pearsonr(ra, rd)[0])
    cl = [REC[s]["cluster"] for s in ss]
    uq = sorted(set(cl))
    lab = np.array([uq.index(c) for c in cl])
    rac = np.array([float(np.mean(ra[lab == i])) for i in range(len(uq))])
    rdc = rd - rd.mean()
    Syy = float(rdc @ rdc)
    b = 0; done = 0
    tile = np.tile(np.arange(len(uq), dtype=np.int32), (min(chunk, N), 1))
    while done < N:
        B = min(chunk, N - done)
        t = tile[:B]
        rng.permuted(t, axis=1, out=t)
        Xp = rac[t][:, lab]
        Xc = Xp - Xp.mean(axis=1, keepdims=True)
        rr = np.abs((Xc * rdc).sum(1) / np.sqrt((Xc * Xc).sum(1) * Syy))
        b += int(np.count_nonzero(rr >= abs(r_obs) - 1e-12))
        done += B
    out = mc_summary(b, N)
    out.update(n=int(n), observed_statistic_partial_r=r_obs,
               distinct_arrangements=math.factorial(len(uq)),
               calibration="NOT CALIBRATED -- null statistic has a different "
                           "variance from the observed statistic")
    return out


freedman = {}
for v, ss in VARIANTS.items():
    freedman[v] = {sch: freedman_lane(ss, sch)
                   for sch in ("CB", "CB_ws", "CM", "FREE")}
    freedman[v]["LEGACY_HYBRID_review4_recomputed"] = legacy_hybrid_partial(ss)
    print(f"[FL] {v}: partial_r={freedman[v]['CB']['partial_r']:+.4f} "
          f"CB p={freedman[v]['CB']['p']:.4f} "
          f"CB_ws p={freedman[v]['CB_ws']['p']:.4f} "
          f"CM p={freedman[v]['CM']['p']:.4f} "
          f"FREE p={freedman[v]['FREE']['p']:.4f} "
          f"LEGACY p={freedman[v]['LEGACY_HYBRID_review4_recomputed']['p']:.4f}",
          flush=True)

freedman["_scheme_roles"] = {
    "CB": "PRIMARY for the manuscript (restricted equal-size city-block "
          "permutation of the reduced-model residuals)",
    "CB_ws": "sensitivity (adds within-city exchangeability)",
    "CM": "city-mean companion; note the partial r is a DIFFERENT statistic "
          "(city-mean level) and n = number of cities",
    "FREE": "reference only -- ignores clustering, so it agrees with the "
            "parametric partial p and understates the p a cluster-aware test "
            "gives; must NOT be quoted",
    "LEGACY_HYBRID_review4_recomputed": "provenance only -- reproduces what "
            "review4_additions.py computes; superseded",
}
freedman["_prior_audit_reconciliation"] = {
    "audit_reported_core16_range": [0.0093, 0.0164],
    "audit_reported_core16_partial_r": -0.6362,
    "this_run_core16_partial_r": freedman["core16"]["CB"]["partial_r"],
    "resolution": ("the audit's range is NOT a disagreement: its lower end "
                   "0.0093 is the legacy review4 hybrid scheme recomputed on "
                   "core16 (this run: "
                   f"{freedman['core16']['LEGACY_HYBRID_review4_recomputed']['p']:.5f}) "
                   "and its upper end 0.0164 is the restricted city-block "
                   f"scheme (this run: {freedman['core16']['CB']['p']:.5f}). "
                   "Only the city-block value should be quoted."),
}

# ------------------------------- 5. nested-refit cluster bootstrap ---------
def bootstrap_slope(ss, nested, B=NBOOT):
    """Cluster bootstrap for the OLS slope of dTsa on albedo, reported per
    0.1 albedo.  Sufficient statistics (n, Sx, Sy, Sxy, Sxx) are accumulated
    per drawn city, so the OLS refit is exact for every resample.

    nested=False : one-stage -- resample cities, keep all their records
                   (what paper_stats.py:cluster_bootstrap_slope does).
    nested=True  : two-stage -- resample cities, then resample records with
                   replacement WITHIN each drawn city.  For a size-2 city the
                   three distinct within-city resamples {a,a},{a,b},{b,b} have
                   probabilities 1/4, 1/2, 1/4, so the stage-2 draw is exact.
    """
    alb = vec(ss, "albedo"); dt = vec(ss, "dtsa")
    bl = blocks_of(ss)
    m = len(bl)
    cand, probs = [], []
    for b_idx in bl:
        x = alb[b_idx]; y = dt[b_idx]
        k = x.size
        if not nested or k == 1:
            cand.append([np.array([k, x.sum(), y.sum(), (x * y).sum(), (x * x).sum()])])
            probs.append(np.array([1.0]))
        else:
            opts, pr = [], []
            for i in range(k):
                for j in range(i, k):
                    sel = np.array([i, j])
                    xs = x[sel]; ys = y[sel]
                    opts.append(np.array([k, xs.sum(), ys.sum(),
                                          (xs * ys).sum(), (xs * xs).sum()]))
                    pr.append((1.0 if i == j else 2.0) / (k ** 2))
            cand.append(opts); probs.append(np.array(pr))
    maxopt = max(len(c) for c in cand)
    TAB = np.zeros((m, maxopt, 5))
    PRB = np.zeros((m, maxopt))
    for i, (c, p) in enumerate(zip(cand, probs)):
        TAB[i, :len(c)] = np.stack(c)
        PRB[i, :len(p)] = p
    CDF = np.cumsum(PRB, axis=1)

    pick = rng.integers(0, m, size=(B, m))                       # stage 1
    u = rng.random((B, m))
    opt = (u[..., None] > CDF[pick]).sum(axis=2)                 # stage 2
    opt = np.minimum(opt, maxopt - 1)
    S = TAB[pick, opt].sum(axis=1)                               # (B,5)
    nn, Sx, Sy, Sxy, Sxx = S.T
    den = nn * Sxx - Sx ** 2
    ok = den > 1e-12
    slopes = np.where(ok, (nn * Sxy - Sx * Sy) / np.where(ok, den, 1.0), np.nan) * 0.1
    # how many distinct cities were drawn (proxy for design degeneracy)
    ps = np.sort(pick, axis=1)
    ndist = 1 + (np.diff(ps, axis=1) != 0).sum(axis=1)
    sl = slopes[ok]
    n_single = sum(1 for b_idx in bl if b_idx.size == 1)
    return dict(
        draws_requested=int(B), draws_used=int(ok.sum()),
        draws_dropped_degenerate=int((~ok).sum()),
        drop_fraction=float((~ok).sum()) / B,
        analytic_drop_probability_upper_bound=float(n_single * m ** (-float(m))),
        analytic_drop_probability_note=(
            "the design is degenerate only when the albedo column has a single "
            "distinct value, which for this sample requires all m stage-1 draws "
            "to hit the SAME one-record city (all city albedos are distinct, "
            "and a repeated two-record city still supplies two distinct "
            f"albedos); probability = n_singleton_cities * m^-m = {n_single} * "
            f"{m}^-{m}"),
        min_distinct_cities_drawn=int(ndist.min()),
        frac_draws_with_le2_distinct_cities=float(np.mean(ndist <= 2)),
        frac_draws_with_le3_distinct_cities=float(np.mean(ndist <= 3)),
        slope_point_estimate_K_per_0p1=float(np.polyfit(alb, dt, 1)[0] * 0.1),
        ci95=[float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))],
        ci90=[float(np.percentile(sl, 5.0)), float(np.percentile(sl, 95.0))],
        median=float(np.median(sl)),
        excludes_zero=bool(np.percentile(sl, 97.5) < 0.0),
    )


boot = {"_interpretation": (
    "Zero resamples were dropped in every variant and under both schemes, and "
    "the analytic drop probability is below 1e-16, so the CIs are plain "
    "percentile intervals over the full set of resamples and need no "
    "conditional-on-non-degeneracy caveat. The manuscript's published CIs come "
    "from 5000 one-stage draws; at 200 000 draws the endpoints move by up to "
    "0.11 K (excl_mpls17), so the published interval should be re-read off the "
    "200 000-draw run. The nested two-stage interval is the honest one if the "
    "two records of a two-record city are treated as exchangeable replicates "
    "rather than as fixed; it is wider on the lower side (all19 -6.36 vs -5.71, "
    "excl_mpls17 -7.28 vs -6.41) and essentially identical in core16, where "
    "only one two-record city survives.")}
for v, ss in VARIANTS.items():
    boot[v] = {"one_stage_cluster": bootstrap_slope(ss, nested=False),
               "nested_two_stage": bootstrap_slope(ss, nested=True)}
    o = boot[v]["one_stage_cluster"]; nn = boot[v]["nested_two_stage"]
    print(f"[boot] {v}: slope={o['slope_point_estimate_K_per_0p1']:+.3f} "
          f"one-stage CI={np.round(o['ci95'],3)} drop={o['draws_dropped_degenerate']} | "
          f"nested CI={np.round(nn['ci95'],3)} drop={nn['draws_dropped_degenerate']}",
          flush=True)

# --------------------------------------------- scheme documentation ---------
scheme_documentation = {
    "CB": {
        "name": "restricted equal-size city-block permutation (record level)",
        "role": "PRIMARY -- the headline p should come from this",
        "unit_permuted": ("whole city blocks of albedo values; a block is "
                          "reassigned only to a block of the SAME size, so n "
                          "and the block-size pattern are preserved exactly"),
        "within_block_order": "canonical (site-name sorted), not randomized",
        "within_city_albedo_contrast": ("PRESERVED -- Helsinki (0.142/0.110) "
                                        "and Lodz (0.085/0.087) keep their "
                                        "internal contrast"),
        "size1_blocks": "permuted freely among the other size-1 blocks",
        "size2_blocks": "permuted only among the other size-2 blocks",
        "test_statistic": "|Pearson r| at record level; the identity "
                          "arrangement reproduces the headline r exactly",
        "exchangeability": ("valid conditional on the observed block-size "
                            "pattern: under H0 of no city-level albedo-offset "
                            "association, equal-size city blocks are "
                            "exchangeable. Does NOT assume records within a "
                            "city are exchangeable with records of other "
                            "cities."),
        "known_limitation": ("when a size class has only one member that block "
                             "cannot move. In core16 the Lodz cluster drops to "
                             "one record, leaving Helsinki as the sole size-2 "
                             "block, so its two albedos are frozen and only 14 "
                             "of 16 records get relabelled."),
    },
    "CB_ws": {
        "name": "CB plus within-block shuffling",
        "role": "sensitivity -- the only randomization available to a frozen block",
        "unit_permuted": "as CB, and additionally the order of records inside "
                         "each block",
        "extra_assumption": "records within one city are exchangeable with each "
                            "other",
        "note": ("US-Minneapolis1/2 are numerically identical records (albedo "
                 "0.213, dTsa -3.0439, eps_sky 0.82074, 19723 nights each), so "
                 "their within-block shuffle is a no-op and the distinct "
                 "arrangement count is 4x, not 8x, the CB count in all19."),
    },
    "CM": {
        "name": "city-mean permutation (paper_stats.py:290 cluster_permutation)",
        "role": "COMPANION -- belongs next to the city-mean r, not the record r",
        "unit_permuted": "the m city-mean albedos, permuted freely (m! "
                         "arrangements)",
        "test_statistic": "|Pearson r| over the m city-mean points",
        "cluster_size_handling": ("size-2 cities are collapsed to an UNWEIGHTED "
                                  "mean, so a 2-record city counts the same as "
                                  "a 1-record city"),
        "within_city_albedo_contrast": ("DESTROYED -- Helsinki becomes a single "
                                        "point at albedo 0.126, Lodz a single "
                                        "point at 0.086"),
        "exchangeability": ("fully exchangeable under H0 if cities are iid "
                            "draws; the cleanest null, at the cost of "
                            "discarding within-city information and of not "
                            "testing the statistic the manuscript headlines"),
        "mismatch_in_current_manuscript": ("main.tex quotes this p immediately "
                                           "after the record-level r=-0.79, but "
                                           "the statistic tested is the "
                                           "city-mean r=-0.83, which the "
                                           "manuscript reports in a different "
                                           "sentence"),
    },
    "SM": {
        "name": "city-mean-smeared record-level permutation "
                "(review2_additions.py:60 cluster_perm_p)",
        "role": "LEGACY -- provenance only, should not be quoted",
        "unit_permuted": "the m city-mean albedos, then re-expanded to record "
                         "level so a size-2 city contributes its mean twice",
        "test_statistic": ("record-level |Pearson r| between city-mean-smeared "
                           "albedo and record-level dTsa -- a third quantity, "
                           "equal to neither the record-level r nor the "
                           "city-mean r"),
        "cluster_size_handling": "size-2 cities carry double weight in both the "
                                 "observed statistic and the null",
        "within_city_albedo_contrast": "DESTROYED on the albedo side, RETAINED "
                                       "on the dTsa side -- an asymmetric "
                                       "treatment with no null interpretation",
        "exchangeability": ("the observed statistic is not the headline "
                            "statistic, and the mixed record/city weighting "
                            "means the permutation distribution is not the "
                            "null distribution of any reported effect size; "
                            "this is why the core16 p in main.tex L481 is not "
                            "comparable with the all19/excl17 values beside it"),
        "field_mislabel": ("stored as 'r_cluster' in "
                           "review2_additions.core_variant_permutation, which "
                           "reads as a city-mean r but is not one"),
    },
    "FL": {
        "name": "Freedman-Lane (1983) permutation for a partial slope",
        "textbook_recipe": ("(1) fit the reduced model dTsa ~ 1 + eps_sky, keep "
                            "fitted values f and residuals e; (2) permute e "
                            "(city-block restricted, or city-mean); (3) form "
                            "y* = f + e*; (4) refit the full model "
                            "y* ~ 1 + eps_sky + albedo and take |t(beta_albedo)|; "
                            "(5) p = (b+1)/(N+1). The identity permutation "
                            "returns t_obs exactly."),
        "defect_in_review4_additions": (
            "review4_additions.py:fl_perm permutes the CITY-MEAN albedo "
            "RESIDUALS expanded back to record level and compares the resulting "
            "record-level |r| against an observed record-level partial r. The "
            "permuted predictor has smaller variance than the observed one, so "
            "the null statistic is not on the same scale as the observed "
            "statistic; the reported p is therefore not calibrated. It also "
            "permutes the predictor rather than the response residuals, which "
            "is the Kennedy/'permute-X' variant, not Freedman-Lane."),
        "coverage_defect": ("review4 ran only all19 and excl_mpls17, violating "
                            "the manuscript's own three-variant convention "
                            "(main.tex L393-397); core16 was missing."),
    },
}

# ------------------------------------------------------- assemble ----------
out = {
    "_merge_target": "review5_additions.permutation_v11",
    "_supersedes": [
        "cluster_permutation.all",
        "cluster_permutation.excl_mpls",
        "review4_additions.albedo_given_skyemissivity_freedman_lane",
    ],
    "_also_stale_but_not_listed_for_supersede": [
        "review2_additions.core_variant_permutation (scheme SM; source of the "
        "core16 p=1.0e-3 in main.tex L481)",
        "review3b_additions.albedo_given_skyemissivity_clusteraware (scheme "
        "SM-partial, nperm=10000)",
    ],
    "meta": {
        "script": "analysis/revalidation_2026-08/scripts/review5_permutation_rebuild.py",
        "date": "2026-09-16", "seed": SEED, "nperm": NPERM,
        "n_bootstrap_draws": NBOOT,
        "constants": {"sigma": SIG, "epsilon": EPS,
                      "note": "sigma is the project convention 5.67e-8, not "
                              "5.670374419e-8, to stay commensurate with Table 1"},
        "inputs": ["data/urban-plumber/corpus/<SITE>.nc (night_mask, "
                   "pre_spinup_flag, obs_LWup, forcing_Tair, forcing_LWdown)",
                   "data/urban-plumber/FullCollection/<SITE>/"
                   "<SITE>_sitedata_v1.csv (average_albedo_at_midday)"],
        "clmu_independence": ("no CLMU or TEB quantity is read or computed "
                              "anywhere in this script; it is unaffected by the "
                              "CLMU output-alignment fix"),
        "ledger_crosscheck": xcheck,
    },
    "floor_audit": floor_audit,
    "scheme_documentation": scheme_documentation,
    "p_quoting_guidance": {
        "rule": ("at N = 2e7 every albedo-offset permutation p has a Monte "
                 "Carlo standard error below 1e-5 and every Freedman-Lane p "
                 "below 7e-5, so two significant figures are safe to print "
                 "provided the MC standard error is stated once in Methods; "
                 "the conservative single-digit reading is also given per test "
                 "as `resolved_significant_digits`"),
        "never_say": ["resolution floor (unless b = 0)",
                      "exact permutation test",
                      "p < 1e-4 without naming the scheme and N"],
        "floor_at_this_N": 1.0 / (NPERM + 1),
        "why_not_exact": ("the number of distinct attainable arrangements is "
                          "1.2e10 to 2.1e13 depending on scheme and variant, "
                          "so 2e7 draws sample well under 0.2 % of the "
                          "reference set in the most restricted case and under "
                          "1e-4 % in the least; these are Monte Carlo "
                          "permutation tests, sampled with replacement from the "
                          "arrangement set, not exhaustive ones"),
    },
    "albedo_offset": albedo_offset,
    "freedman_lane_albedo_given_skyemissivity": freedman,
    "headline_slope_bootstrap": boot,
}

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def leaves(o):
    if isinstance(o, dict):
        return sum(leaves(v) for v in o.values())
    if isinstance(o, list):
        return sum(leaves(v) for v in o) if o else 1
    return 1


print(f"\nwrote {OUT}  leaves={leaves(out)}")
