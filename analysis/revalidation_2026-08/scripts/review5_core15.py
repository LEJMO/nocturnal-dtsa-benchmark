# -*- coding: utf-8 -*-
"""core15 sensitivity: drop PL-Narutowicza from the conservative core, and carry
the drop through the MODEL evaluation, not only the observed correlation.

WHY. The PL-Narutowicza 2011-01-17/18 longwave step is documented in
review5_additions.temporal_sampling.narutowicza_regime_step, but that block only
propagates the step into the OBSERVED albedo-offset relation. LW-down is also a
MODEL FORCING input, so the step has to be carried into the model-side numbers
(spread ratio, r, rho, observed-aligned slope beta, scheme ordering) and into the
benchmark win counts. This script does that, and first establishes -- from the
NetCDF, the TEB input deck and the CLM-Urban history file -- whether the forcing
actually carries the step.

WHAT IT COMPUTES
  (a) forcing_step: nocturnal mean LW-down before/after the break in
      - data/urban-plumber/corpus/PL-Narutowicza.nc  forcing_LWdown (what the runs read)
      - FullCollection raw_observations / metforcing / era5_corrected LWdown
      - external/teb_runs/PL-Narutowicza/input/Forc_LW.txt (TEB's actual input deck)
      - external/clmu_baseline19/PL-Narutowicza_base.nc FLDS (what CLM-Urban received)
      each against PL-Lipowa over the same dates, raw and calendar-month-matched
      (a January break splits the record by season, so the raw level difference is
      seasonally confounded; the month-matched and the paired-vs-Lipowa forms are not).
  (b) model_side: for all19 / excl_mpls17 / core16 / core15, both schemes,
      s_m/s_o, Pearson r, Spearman rho, beta = r*(s_m/s_o), city-cluster bootstrap
      95% CIs, one-sided P(beta>=1) and P(r<=0), and the paired TEB-minus-CLMU5
      difference with CI -- EXACTLY the conventions of
      review5_scheme_ordering_paired.py (100,000 draws, seed 20260916, resampling
      unit = city cluster, CLMU from the alignment-corrected per_site.clmu_dtsa).
  (c) observed: core15 albedo-offset Pearson r, Spearman rho and OLS slope in
      K per 0.1 albedo (paper_stats.py convention), recomputed independently from
      the corpus NetCDF as well as from per_site, plus cluster-bootstrap CIs and
      one-sided P(r>=0); and the benchmark win counts restricted to core15 from
      review5_additions.nocturnal_benchmark_skill.benchmark.per_record_mae.

READ-ONLY: results/paper_stats_v1.json is opened for reading only and never written.
Output: results/review5_core15.json, for the lead to merge under
review5_additions.core15_sensitivity.

Usage: python review5_core15.py [--draws 100000]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / 'data/urban-plumber/corpus'
FULLCOL = ROOT / 'data/urban-plumber/FullCollection'
STATS = ROOT / 'results/paper_stats_v1.json'
OUT = ROOT / 'results/review5_core15.json'

SIG, EPS = 5.67e-8, 0.95          # frozen project constants (SIG is NOT exact S-B)
NDRAW = 100_000
SEED = 20260916                   # identical to review5_scheme_ordering_paired.py
SEED_OBS = 20260904               # paper_stats.py seed, fresh stream (see note)

SITE_N, SITE_L = 'PL-Narutowicza', 'PL-Lipowa'
CUT_LWUP = np.datetime64('2011-01-17')    # obs LW-up changepoint (stats file)
CUT_LWDOWN = np.datetime64('2011-01-18')  # forcing LW-down changepoint (stats file)
WIN0, WIN1 = np.datetime64('2008-01-01'), np.datetime64('2013-01-01')
SPINUP_YEAR = 17520               # clmu_post_v11 alignment constant


# ----------------------------------------------------------------- helpers ---
def inv_ts(lwup, lwdown):
    """Frozen radiometric surface-temperature inversion."""
    return ((lwup - (1.0 - EPS) * lwdown) / (EPS * SIG)) ** 0.25


def load_corpus(site):
    ds = xr.open_dataset(CORPUS / f'{site}.nc')
    out = dict(
        t=ds.time.values,
        night=(np.asarray(ds.night_mask.values, bool)
               & ~np.asarray(ds.pre_spinup_flag.values, bool)),
        ld=np.asarray(ds.forcing_LWdown.values, float),
        ta=np.asarray(ds.forcing_Tair.values, float),
        lwup=np.asarray(ds.obs_LWup.values, float),
    )
    ds.close()
    return out


def prepost(values, mask, day, cut, months=None):
    """Raw and calendar-month-matched pre/post means of `values` under `mask`."""
    pre, post = mask & (day < cut), mask & (day >= cut)
    a, b = float(values[pre].mean()), float(values[post].mean())
    row = dict(n_pre=int(pre.sum()), n_post=int(post.sum()),
               raw_pre=a, raw_post=b, raw_step=b - a)
    if months is not None:
        pm, qm = [], []
        for mm in range(1, 13):
            x, y = values[pre & (months == mm)], values[post & (months == mm)]
            if x.size >= 100 and y.size >= 100:
                pm.append(x.mean()); qm.append(y.mean())
        if pm:
            row.update(month_matched_pre=float(np.mean(pm)),
                       month_matched_post=float(np.mean(qm)),
                       month_matched_step=float(np.mean(qm) - np.mean(pm)),
                       months_used=len(pm))
    return row


def fullcol_series(site, suffix, var='LWdown'):
    ds = xr.open_dataset(FULLCOL / site / 'timeseries' / f'{site}{suffix}')
    v = np.asarray(ds[var].squeeze().values, float)
    sw = np.asarray(ds['SWdown'].squeeze().values, float)
    t = ds.time.values
    ds.close()
    return t, v, sw


def spearman_np(a, b):
    """Rank correlation, same implementation as review5_scheme_ordering_paired."""
    def rk(x):
        x = np.asarray(x, float)
        o = x.argsort()
        r = np.empty(len(x)); r[o] = np.arange(len(x))
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r)
        return (s / cnt)[inv]
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


def model_stats(o, m):
    o = np.asarray(o, float); m = np.asarray(m, float)
    r = float(np.corrcoef(o, m)[0, 1])
    sdr = float(m.std(ddof=1) / o.std(ddof=1))
    return dict(r=r, rho=spearman_np(o, m), sd_ratio=sdr, beta=r * sdr)


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


# ------------------------------------------------ (a) the forcing question ---
def forcing_step():
    N, L = load_corpus(SITE_N), load_corpus(SITE_L)
    assert np.array_equal(N['t'], L['t']), 'Lodz pair time axes differ'
    t = N['t']
    day = t.astype('datetime64[D]')
    mon = (t.astype('datetime64[M]').astype(int) % 12) + 1

    out = {'break_dates': {'obs_LWup': str(CUT_LWUP), 'forcing_LWdown': str(CUT_LWDOWN)},
           'levels': {}, 'paired_vs_lipowa': {}, 'what_the_runs_read': {}}

    # -- corpus forcing_LWdown, absolute nocturnal levels, both Lodz records
    for tag, R in ((SITE_N, N), (SITE_L, L)):
        m = R['night'] & np.isfinite(R['ld'])
        out['levels'][f'{tag}::corpus.forcing_LWdown'] = prepost(
            R['ld'], m, day, CUT_LWDOWN, mon)

    # -- FullCollection provenance chain: tower -> model forcing -> reanalysis
    chain = (('raw_observations_tower', '_raw_observations_v1.nc'),
             ('metforcing_model_input', '_metforcing_v1.nc'),
             ('era5_corrected_reanalysis', '_era5_corrected_v1.nc'))
    for site in (SITE_N, SITE_L):
        for tag, suf in chain:
            try:
                tt, vv, ss = fullcol_series(site, suf)
            except Exception as exc:                       # pragma: no cover
                out['levels'][f'{site}::{tag}'] = {'error': repr(exc)}
                continue
            dd = tt.astype('datetime64[D]')
            mm = (tt.astype('datetime64[M]').astype(int) % 12) + 1
            msk = (ss <= 1.0) & np.isfinite(vv) & (dd >= WIN0) & (dd < WIN1)
            out['levels'][f'{site}::{tag}'] = prepost(vv, msk, dd, CUT_LWDOWN, mm)

    # -- paired (Narutowicza minus Lipowa) differences, the instrument-offset form
    both_ld = N['night'] & L['night'] & np.isfinite(N['ld']) & np.isfinite(L['ld'])
    out['paired_vs_lipowa']['forcing_LWdown'] = prepost(
        N['ld'] - L['ld'], both_ld, day, CUT_LWDOWN, mon)
    both_lu = both_ld & np.isfinite(N['lwup']) & np.isfinite(L['lwup'])
    out['paired_vs_lipowa']['obs_LWup'] = prepost(
        N['lwup'] - L['lwup'], both_lu, day, CUT_LWUP, mon)
    out['paired_vs_lipowa']['forcing_Tair_at_LWup_break'] = prepost(
        N['ta'] - L['ta'], both_lu, day, CUT_LWUP, mon)

    # -- what the two runs ACTUALLY read
    lw_teb = np.loadtxt(ROOT / 'external/teb_runs' / SITE_N / 'input/Forc_LW.txt')
    n = len(N['ld'])
    k = min(len(lw_teb), n)
    m = N['night'][:k] & np.isfinite(lw_teb[:k])
    out['what_the_runs_read']['TEB_Forc_LW_txt'] = {
        **prepost(lw_teb[:k], m, day[:k], CUT_LWDOWN, mon[:k]),
        'len': int(lw_teb.size), 'corpus_len': int(n),
        'max_abs_diff_vs_corpus_forcing': float(np.nanmax(np.abs(lw_teb[:k] - N['ld'][:k]))),
    }

    h = xr.open_dataset(ROOT / 'external/clmu_baseline19' / f'{SITE_N}_base.nc')
    flds = np.asarray(h['FLDS'].values).reshape(-1)
    tair_m = np.asarray(h['Tair'].values).reshape(-1)
    h.close()
    off = min(SPINUP_YEAR, n) + 1
    win = min(n, len(tair_m) - off)
    echo = float(np.nanmax(np.abs(tair_m[off:off + win] - N['ta'][:win])))
    ld_clmu = flds[off:off + win]
    m = N['night'][:win] & np.isfinite(ld_clmu)
    out['what_the_runs_read']['CLMU5_history_FLDS'] = {
        **prepost(ld_clmu, m, day[:win], CUT_LWDOWN, mon[:win]),
        'align_offset': int(off), 'align_window': int(win),
        'tair_echo_max_abs_diff': echo,
        'max_abs_diff_vs_corpus_forcing': float(np.nanmax(np.abs(ld_clmu - N['ld'][:win]))),
    }
    return out


# ------------------------------------------------- (b) model-side core15 -----
def model_side(PSD, draws):
    S = sorted(PSD)
    clus = {s: PSD[s].get('cluster', s) for s in S}
    obs = np.array([PSD[s]['dtsa_std'] for s in S])
    teb = np.array([PSD[s]['teb_dtsa'] for s in S])
    clm = np.array([PSD[s]['clmu_dtsa'] for s in S])

    no_mpls = np.array([not s.startswith('US-Minneapolis') for s in S])
    MASK = {
        'all19': np.ones(len(S), bool),
        'excl_mpls17': no_mpls,
        'core16': no_mpls & np.array([s != 'PL-Lipowa' for s in S]),
        'core15': no_mpls & np.array([s not in ('PL-Lipowa', 'PL-Narutowicza')
                                      for s in S]),
    }
    out = {}
    keys = ['r', 'rho', 'sd_ratio', 'beta']
    for vk, mk in MASK.items():
        o, tb, cm = obs[mk], teb[mk], clm[mk]
        cl = np.array([clus[s] for s in np.array(S)[mk]])
        uc = sorted(set(cl.tolist()))
        idx_by_c = {u: np.where(cl == u)[0] for u in uc}
        pt_t, pt_c = model_stats(o, tb), model_stats(o, cm)

        rng = np.random.default_rng(SEED)
        bt = {k: [] for k in keys}; bc = {k: [] for k in keys}
        dif = {k: [] for k in keys}
        for _ in range(draws):
            pick = rng.choice(len(uc), size=len(uc), replace=True)
            ii = np.concatenate([idx_by_c[uc[p]] for p in pick])
            if len(np.unique(o[ii])) < 3:
                continue
            st, sc = model_stats(o[ii], tb[ii]), model_stats(o[ii], cm[ii])
            for k in keys:
                bt[k].append(st[k]); bc[k].append(sc[k]); dif[k].append(st[k] - sc[k])
        n_used = len(dif['r'])
        out[vk] = {
            'sites': [s for s in np.array(S)[mk]],
            'n': int(mk.sum()), 'n_clusters': len(uc),
            'draws_used': n_used, 'draws_dropped': draws - n_used,
            'TEB': {**pt_t, **{f'{k}_ci95': ci(bt[k]) for k in keys}},
            'CLMU5': {**pt_c, **{f'{k}_ci95': ci(bc[k]) for k in keys}},
            'paired_difference_TEB_minus_CLMU5': {
                k: {'point': pt_t[k] - pt_c[k], 'ci95': ci(dif[k]),
                    'P_le_0': float(np.mean(np.asarray(dif[k]) <= 0))} for k in keys},
            'one_sided': {
                'TEB_P_r_le_0': float(np.mean(np.asarray(bt['r']) <= 0)),
                'CLMU5_P_r_le_0': float(np.mean(np.asarray(bc['r']) <= 0)),
                'TEB_P_beta_ge_1': float(np.mean(np.asarray(bt['beta']) >= 1.0)),
                'CLMU5_P_beta_ge_1': float(np.mean(np.asarray(bc['beta']) >= 1.0)),
                'TEB_P_sdratio_ge_1': float(np.mean(np.asarray(bt['sd_ratio']) >= 1.0)),
                'CLMU5_P_sdratio_ge_1': float(np.mean(np.asarray(bc['sd_ratio']) >= 1.0))},
        }
    return out


# ------------------------------------------- (c) observed side + benchmark ---
def recompute_obs_from_netcdf(sites):
    """Independent dTsa per record straight out of the corpus NetCDF."""
    vals = {}
    for s in sites:
        ds = xr.open_dataset(CORPUS / f'{s}.nc')
        night = (np.asarray(ds.night_mask.values, bool)
                 & ~np.asarray(ds.pre_spinup_flag.values, bool))
        lwup = np.asarray(ds.obs_LWup.values, float)
        ld = np.asarray(ds.forcing_LWdown.values, float)
        ta = np.asarray(ds.forcing_Tair.values, float)
        ds.close()
        m = night & np.isfinite(lwup) & np.isfinite(ld) & np.isfinite(ta)
        vals[s] = float(np.mean(inv_ts(lwup[m], ld[m]) - ta[m]))
    return vals


def cluster_permutation(PSD, ss, nperm=10000, seed=20260904):
    """paper_stats.py convention: aggregate to city clusters (mean albedo, mean
    dTsa), permute the albedo labels, two-sided on |r|."""
    by_c = {}
    for s in ss:
        by_c.setdefault(PSD[s].get('cluster', s), []).append(s)
    cl = list(by_c)
    cx = np.array([np.mean([PSD[s]['albedo'] for s in by_c[c]]) for c in cl])
    cy = np.array([np.mean([PSD[s]['dtsa_std'] for s in by_c[c]]) for c in cl])
    r_obs = abs(pearsonr(cx, cy)[0])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(nperm):
        if abs(pearsonr(rng.permutation(cx), cy)[0]) >= r_obs - 1e-12:
            cnt += 1
    return dict(n_clusters=len(cl), r_cluster=float(pearsonr(cx, cy)[0]),
                abs_r=float(r_obs), p_perm=(cnt + 1) / (nperm + 1), nperm=nperm)


def observed_side(PSD, draws):
    S = sorted(PSD)
    no_mpls = [s for s in S if not s.startswith('US-Minneapolis')]
    SETS = {
        'all19': S,
        'excl_mpls17': no_mpls,
        'core16': [s for s in no_mpls if s != 'PL-Lipowa'],
        'core15': [s for s in no_mpls if s not in ('PL-Lipowa', 'PL-Narutowicza')],
    }
    nc_dtsa = recompute_obs_from_netcdf(S)
    out = {'netcdf_recomputation_max_abs_diff_vs_per_site': float(
        max(abs(nc_dtsa[s] - PSD[s]['dtsa_std']) for s in S))}
    for vk, ss in SETS.items():
        alb = np.array([PSD[s]['albedo'] for s in ss])
        dt = np.array([PSD[s]['dtsa_std'] for s in ss])
        dt_nc = np.array([nc_dtsa[s] for s in ss])
        pr, pp = pearsonr(alb, dt)
        sr, sp = spearmanr(alb, dt)
        slope = float(np.polyfit(alb, dt, 1)[0] * 0.1)
        pr2, pp2 = pearsonr(alb, dt_nc)
        by_c = {}
        for s in ss:
            by_c.setdefault(PSD[s].get('cluster', s), []).append(s)
        cl = list(by_c)
        rng = np.random.default_rng(SEED_OBS)
        br, bs = [], []
        for _ in range(draws):
            pick = rng.choice(len(cl), size=len(cl), replace=True)
            xs, ys = [], []
            for i in pick:
                for s in by_c[cl[i]]:
                    xs.append(PSD[s]['albedo']); ys.append(PSD[s]['dtsa_std'])
            if len(set(xs)) < 2:
                continue
            bs.append(np.polyfit(xs, ys, 1)[0] * 0.1)
            if len(set(ys)) > 1:
                br.append(np.corrcoef(xs, ys)[0, 1])
        out[vk] = {
            'n_sites': len(ss), 'n_clusters': len(cl),
            'pearson_r': float(pr), 'pearson_p': float(pp),
            'spearman_r': float(sr), 'spearman_p': float(sp),
            'slope_K_per_0p1_albedo': slope,
            'slope_ci95_cluster_bootstrap': ci(bs),
            'r_ci95_cluster_bootstrap': ci(br),
            'P_r_ge_0': float(np.mean(np.asarray(br) >= 0.0)),
            'bootstrap_draws_used': len(bs),
            'pearson_r_from_netcdf_recomputation': float(pr2),
            'pearson_p_from_netcdf_recomputation': float(pp2),
            'cluster_permutation': cluster_permutation(PSD, ss),
        }
    return out


def benchmark_core15(bm):
    """Restrict review5 benchmark win counts to each record set."""
    per = bm['benchmark']['per_record_mae']
    dec = bm['benchmark']['bias_decomposition']
    S = sorted(per['TEB'])
    no_mpls = [s for s in S if not s.startswith('US-Minneapolis')]
    SETS = {
        'all19': S,
        'excl_mpls17': no_mpls,
        'core16': [s for s in no_mpls if s != 'PL-Lipowa'],
        'core15': [s for s in no_mpls if s not in ('PL-Lipowa', 'PL-Narutowicza')],
    }
    flags = ('beats_KM3', 'beats_REG1', 'beats_REG2', 'beats_all_three',
             'beats_meaningful_two_REG2_KM3')
    out = {}
    for vk, ss in SETS.items():
        blk = {'n': len(ss)}
        for sch in ('TEB', 'CLMU5'):
            blk[sch] = {f: int(sum(bool(per[sch][s][f]) for s in ss)) for f in flags}
            blk[sch]['median_scheme_MAE_Wm2'] = float(
                np.median([per[sch][s]['scheme_MAE'] for s in ss]))
            pr_d = dec['per_record'][sch]
            blk[sch]['debiased_beats_both'] = int(sum(
                bool(pr_d[s]['debiased_beats_both']) for s in ss))
            blk[sch]['raw_beats_both'] = int(sum(
                bool(pr_d[s]['raw_beats_both']) for s in ss))
            blk[sch]['median_absMBE_over_MAE'] = float(np.median(
                [pr_d[s]['abs_MBE_over_MAE'] for s in ss]))
            blk[sch]['median_abs_MBE_Wm2'] = float(np.median(
                [abs(pr_d[s]['MBE']) for s in ss]))
        out[vk] = blk
    return out


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=NDRAW)
    ap.add_argument('--obs-draws', type=int, default=5000)
    args = ap.parse_args()

    sp = json.load(open(STATS, encoding='utf-8'))     # READ ONLY
    PSD = sp['per_site']
    bm = sp['review5_additions']['nocturnal_benchmark_skill']

    print('(a) forcing step ...', flush=True)
    fs = forcing_step()
    for k, v in fs['levels'].items():
        if 'error' in v:
            print(f'   {k:58s} {v["error"]}'); continue
        print(f'   {k:58s} raw {v["raw_pre"]:7.3f} -> {v["raw_post"]:7.3f} '
              f'({v["raw_step"]:+7.3f})   month-matched '
              f'{v.get("month_matched_pre", float("nan")):7.3f} -> '
              f'{v.get("month_matched_post", float("nan")):7.3f} '
              f'({v.get("month_matched_step", float("nan")):+7.3f})')
    for k, v in fs['what_the_runs_read'].items():
        print(f'   {k:58s} raw {v["raw_pre"]:7.3f} -> {v["raw_post"]:7.3f} '
              f'({v["raw_step"]:+7.3f})  max|diff vs corpus| = '
              f'{v["max_abs_diff_vs_corpus_forcing"]:.3e}')

    print('\n(b) model side ...', flush=True)
    ms = model_side(PSD, args.draws)
    for vk, b in ms.items():
        d = b['paired_difference_TEB_minus_CLMU5']
        print(f'  [{vk}] n={b["n"]} clusters={b["n_clusters"]} used={b["draws_used"]}')
        for sch in ('TEB', 'CLMU5'):
            x = b[sch]
            print(f'     {sch:6s} sdr={x["sd_ratio"]:.4f} {x["sd_ratio_ci95"]}  '
                  f'r={x["r"]:+.4f} {x["r_ci95"]}  rho={x["rho"]:+.4f}  '
                  f'beta={x["beta"]:.4f} {x["beta_ci95"]}')
        print(f'     d_beta={d["beta"]["point"]:+.4f} {d["beta"]["ci95"]} '
              f'P(d<=0)={d["beta"]["P_le_0"]:.3f}   '
              f'P(r<=0) {b["one_sided"]["TEB_P_r_le_0"]:.4f}/'
              f'{b["one_sided"]["CLMU5_P_r_le_0"]:.4f}   '
              f'P(beta>=1) {b["one_sided"]["TEB_P_beta_ge_1"]:.5f}/'
              f'{b["one_sided"]["CLMU5_P_beta_ge_1"]:.5f}')

    print('\n(c) observed side ...', flush=True)
    ob = observed_side(PSD, args.obs_draws)
    print(f'   netcdf recomputation max|diff| vs per_site = '
          f'{ob["netcdf_recomputation_max_abs_diff_vs_per_site"]:.2e} K')
    for vk in ('all19', 'excl_mpls17', 'core16', 'core15'):
        b = ob[vk]
        print(f'  [{vk}] n={b["n_sites"]} r={b["pearson_r"]:+.4f} (p={b["pearson_p"]:.2e}) '
              f'rho={b["spearman_r"]:+.4f} slope={b["slope_K_per_0p1_albedo"]:+.4f} '
              f'K/0.1alb  r CI {b["r_ci95_cluster_bootstrap"]}  P(r>=0)={b["P_r_ge_0"]:.4f}'
              f'  cluster-perm r={b["cluster_permutation"]["r_cluster"]:+.4f} '
              f'p={b["cluster_permutation"]["p_perm"]:.5f} '
              f'(k={b["cluster_permutation"]["n_clusters"]})')

    bc = benchmark_core15(bm)
    print('\n   benchmark win counts:')
    for vk, b in bc.items():
        print(f'  [{vk}] n={b["n"]}  TEB KM3/REG2/both = '
              f'{b["TEB"]["beats_KM3"]}/{b["TEB"]["beats_REG2"]}/'
              f'{b["TEB"]["beats_meaningful_two_REG2_KM3"]}   '
              f'CLMU5 = {b["CLMU5"]["beats_KM3"]}/{b["CLMU5"]["beats_REG2"]}/'
              f'{b["CLMU5"]["beats_meaningful_two_REG2_KM3"]}')

    payload = {
        '_merge_target': 'review5_additions.core15_sensitivity',
        'meta': {
            'script': 'review5_core15.py',
            'purpose': ('carry the PL-Narutowicza longwave step through the MODEL '
                        'evaluation and the benchmark, not only the observed '
                        'albedo-offset correlation'),
            'record_sets': {
                'all19': 'all 19 evaluable records',
                'excl_mpls17': 'drop US-Minneapolis1, US-Minneapolis2',
                'core16': 'also drop PL-Lipowa (conservative core)',
                'core15': 'also drop PL-Narutowicza (NEW: the stepped record)'},
            'draws': args.draws, 'seed': SEED,
            'obs_bootstrap_draws': args.obs_draws, 'obs_seed': SEED_OBS,
            'resampling_unit': 'city cluster',
            'clmu_source': 'per_site.clmu_dtsa (alignment-corrected, clmu_post_v11)',
            'conventions': ('model side identical to review5_scheme_ordering_paired.py; '
                            'observed side identical to paper_stats.py headline '
                            '(OLS slope in K per 0.1 albedo, city-cluster bootstrap) '
                            'except that the RNG is a FRESH default_rng(20260904) here '
                            'rather than paper_stats.py mid-stream state, so the '
                            'bootstrap interval endpoints differ in the last digits '
                            'from headline_obs while every point estimate is identical'),
            'constants': {'SIG': SIG, 'EPS': EPS,
                          'note': 'SIG is the project constant 5.67e-8, NOT exact Stefan-Boltzmann'},
            'reads_only': ['results/paper_stats_v1.json',
                           'data/urban-plumber/corpus/*.nc',
                           'data/urban-plumber/FullCollection/PL-*/timeseries/*.nc',
                           'external/teb_runs/PL-Narutowicza/input/Forc_LW.txt',
                           'external/clmu_baseline19/PL-Narutowicza_base.nc'],
        },
        'forcing_step': fs,
        'model_side': ms,
        'observed_side': ob,
        'benchmark_win_counts': bc,
    }
    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print('\nwrote', OUT)


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()
