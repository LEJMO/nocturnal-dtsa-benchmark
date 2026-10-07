# -*- coding: utf-8 -*-
"""review5_temporal_seasonal.py -- TEMPORAL / SEASONAL SAMPLING package (review round 5).

Closes the reviewer's "seasonal and temporal sampling uncertainty" gap, which two
prior rounds left unresolved. Table 1 reported only the nocturnal sample count n:
no analysis window, no monthly coverage, and no temporal sampling uncertainty of
each record mean. The city-cluster bootstrap resamples RECORDS; it says nothing
about how well each record mean is determined by the nights that happen to be in
that record. This script supplies that.

Deliverables (all written to results/review5_temporal-seasonal.json):
  1. per-record analysis window, per-calendar-month nocturnal counts, months
     covered, and the per-month valid-data rate (valid / astronomically possible)
  2. headline albedo-offset r and slope under month-equal and season-equal
     weighting of each record mean (3 record sets each)
  3. temporal sampling uncertainty of each record mean from a NIGHT-level and a
     MULTI-NIGHT-BLOCK bootstrap, block length estimated from the site's own
     nocturnal autocorrelation (integral timescale), not assumed
  4. propagation of that per-record temporal uncertainty into r and slope, and
     into the TEB/obs across-site spread ratios (TEB only -- see CLMU note)
  5. independent reproduction of the DJF-exclusion / single-season / epoch tests
  6. quantitative characterisation of the PL-Narutowicza 2010/2011 regime step
     (both-regime sensitivity values and the drop-the-record variant)

INDEPENDENCE. This is a deliberately separate code path from paper_stats.py: the
masks are read straight out of the corpus NetCDF (night_mask & ~pre_spinup_flag)
instead of through src.training.corpus_loader, the RNG seed base is different
(20260916 vs 20260904), and every reproduced number is recomputed from the
NetCDF, never read from results/paper_stats_v1.json.

FROZEN CONVENTIONS (identical to paper_stats.py -- do not change):
  SIG = 5.67e-8 (NOT the exact Stefan-Boltzmann constant), EPS = 0.95
  night mask   = night_mask & ~pre_spinup_flag
  Ts inversion = ((LWup - (1-eps)*LWdown) / (eps*sigma))**0.25
  dTsa         = nocturnal mean(Ts - forcing_Tair)
  19 evaluable records (MX-Escandon lacks usable obs_LWup)
  TEB          = external/teb_runs/<S>/output/LWU_base.txt, row i -> forcing step i+1
  record sets  : all19 / excl_mpls17 (drop US-Minneapolis1,2) / core16 (also drop PL-Lipowa)

CONCURRENCY. results/paper_stats_v1.json is READ-ONLY here and is never opened.
Output goes to results/review5_temporal-seasonal.json for the lead to merge under
review5_additions.temporal_sampling.

MODEL-SIDE. CLM-Urban statistics are deliberately NOT computed: the CLMU output
alignment is being corrected concurrently (GR-HECKOR, UK-KingsCollege). The
paired night-block machinery is model-agnostic; run
  review5_model_spread_propagation.py --model clmu
after the CLMU regeneration to obtain the CLMU half.

Usage:  python review5_temporal_seasonal.py [--draws 5000] [--prop-draws 20000]
"""
from __future__ import annotations

import argparse
import io
import csv
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / 'data/urban-plumber/corpus'
OUT = ROOT / 'results/review5_temporal-seasonal.json'

SIG, EPS = 5.67e-8, 0.95          # frozen: 5.67e-8, NOT 5.670374419e-8
HEIGHT_MISMATCH = ('US-Minneapolis1', 'US-Minneapolis2')
CLUSTER = {
    'US-Minneapolis1': 'Minneapolis', 'US-Minneapolis2': 'Minneapolis',
    'PL-Lipowa': 'Lodz', 'PL-Narutowicza': 'Lodz',
    'FI-Kumpula': 'Helsinki', 'FI-Torni': 'Helsinki',
}
SEASON_OF_MONTH = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                   6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
SEASONS = ('DJF', 'MAM', 'JJA', 'SON')
SEED_BASE = 20260916              # distinct from paper_stats.py (20260904)
MIN_N = 100                       # a record enters a subsample only with n >= 100
MIN_NIGHTS_PER_MONTH = 3          # a calendar month counts as "sampled" at >= 3 nights


# ------------------------------------------------------------------ helpers ---
def cluster_of(site):
    return CLUSTER.get(site, site)


def sitedata(site):
    p = ROOT / f'data/urban-plumber/FullCollection/{site}/{site}_sitedata_v1.csv'
    vals = {}
    for row in csv.DictReader(p.open(encoding='utf-8')):
        try:
            vals[row['parameter']] = float(row['value'])
        except Exception:
            pass
    return vals


def inv_ts(lwup, lwdown):
    """Frozen radiometric surface-temperature inversion."""
    return ((lwup - (1.0 - EPS) * lwdown) / (EPS * SIG)) ** 0.25


def ols_slope_per01(x, y):
    """OLS slope of y on x, expressed per 0.1 of x (manuscript convention)."""
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0] * 0.1)


def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3:
        return float('nan'), float('nan')
    r, p = pearsonr(x, y)
    return float(r), float(p)


def partial_r(x, y, z):
    """Pearson r(x, y | z) by residualising both on z (OLS, one covariate)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    zx = np.polyval(np.polyfit(z, x, 1), z)
    zy = np.polyval(np.polyfit(z, y, 1), z)
    return pear(x - zx, y - zy)


def runs_of_true(mask):
    """Contiguous True runs -> (start, stop) index pairs. One run == one night."""
    d = np.diff(np.concatenate(([0], mask.view(np.int8) if mask.dtype == bool
                                else mask.astype(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


# ----------------------------------------------------------- record loading ---
def load_record(site):
    """Nocturnal dTsa series + night grouping + coverage bookkeeping for one site."""
    ds = xr.open_dataset(CORPUS / f'{site}.nc')       # xarray decodes _FillValue -> NaN
    if 'obs_LWup' not in ds:
        ds.close()
        return None
    olw = np.asarray(ds['obs_LWup'].values, float)
    ta = np.asarray(ds['forcing_Tair'].values, float)
    ld = np.asarray(ds['forcing_LWdown'].values, float)
    nm = np.asarray(ds['night_mask'].values, bool)
    ps = np.asarray(ds['pre_spinup_flag'].values, bool)
    t = np.asarray(ds['time'].values)
    attrs = dict(native_timestep_s=int(ds.attrs.get('native_timestep_s', 1800)),
                 timestep_s=int(ds.attrs.get('timestep_s', 1800)),
                 koppen=str(ds.attrs.get('koppen_zone', 'NA')),
                 latitude=float(ds.attrs.get('latitude', np.nan)))
    ds.close()

    # masked-aware sanity: any sentinel that survived decoding would show up here
    if np.isfinite(olw).any():
        assert np.nanmin(olw[np.isfinite(olw)]) > 0.0, f'{site}: non-physical obs_LWup survived'

    night = nm & (~ps)                                 # astronomically possible, post-spin-up
    valid = night & np.isfinite(olw) & np.isfinite(ta) & np.isfinite(ld)
    if valid.sum() < MIN_N:
        return None

    d_all = np.full(olw.shape, np.nan)
    d_all[valid] = inv_ts(olw[valid], ld[valid]) - ta[valid]

    # TEB, aligned row i -> forcing step i+1 (frozen convention)
    teb_all = np.full(olw.shape, np.nan)
    p = ROOT / f'external/teb_runs/{site}/output/LWU_base.txt'
    teb_lw = np.full(olw.shape, np.nan)
    if p.exists():
        raw = np.loadtxt(p)
        nmod = min(raw.size, olw.size - 1)
        teb_lw[1:nmod + 1] = raw[:nmod]
        okt = valid & np.isfinite(teb_lw)
        teb_all[okt] = inv_ts(teb_lw[okt], ld[okt]) - ta[okt]

    # night grouping: contiguous runs of the OPPORTUNITY mask (handles midnight)
    st, en = runs_of_true(night)
    night_id = np.full(olw.shape, -1, np.int64)
    for k, (a, b) in enumerate(zip(st, en)):
        night_id[a:b] = k

    step = int(np.median(np.diff(t).astype('timedelta64[s]').astype(np.int64)))
    gap = int(np.max(np.diff(t).astype('timedelta64[s]').astype(np.int64)))

    return dict(site=site, t=t, valid=valid, night=night, d=d_all, teb=teb_all,
                night_id=night_id, n_night_runs=len(st), step_s=step, max_gap_s=gap,
                **attrs)


# ------------------------------------------------------- coverage bookkeeping ---
def coverage(rec):
    t, valid, night = rec['t'], rec['valid'], rec['night']
    tv = t[valid]
    ym_all = t.astype('datetime64[M]')
    moy_all = (ym_all.astype(int) % 12) + 1

    ym_keys = np.unique(ym_all[night])                 # months with any opportunity
    per_ym = {}
    for u in ym_keys:
        k = (ym_all == u)
        nposs = int((night & k).sum())
        nval = int((valid & k).sum())
        if nposs == 0:
            continue
        per_ym[str(u)] = [nval, nposs]

    first_ym, last_ym = str(np.min(ym_all[valid])), str(np.max(ym_all[valid]))
    rates_interior, rates_all = [], []
    for key, (nv, npo) in per_ym.items():
        r = nv / npo
        rates_all.append(r)
        if key not in (first_ym, last_ym):
            rates_interior.append(r)
    rates_use = rates_interior if len(rates_interior) >= 3 else rates_all

    moy_counts, moy_nights = {}, {}
    for mm in range(1, 13):
        k = valid & (moy_all == mm)
        moy_counts[str(mm)] = int(k.sum())
        moy_nights[str(mm)] = int(len(np.unique(rec['night_id'][k]))) if k.any() else 0
    moy_sampled = [mm for mm in range(1, 13) if moy_nights[str(mm)] >= MIN_NIGHTS_PER_MONTH]

    season_counts = {s: 0 for s in SEASONS}
    season_months_sampled = {s: 0 for s in SEASONS}
    for mm in range(1, 13):
        season_counts[SEASON_OF_MONTH[mm]] += moy_counts[str(mm)]
        if mm in moy_sampled:
            season_months_sampled[SEASON_OF_MONTH[mm]] += 1

    # per-calendar-year nocturnal means: the interannual half of temporal sampling
    per_year_mean, per_year_n = {}, {}
    yr_all = t.astype('datetime64[Y]')
    for u in np.unique(yr_all[valid]):
        k = valid & (yr_all == u)
        if k.sum() >= MIN_N:
            per_year_mean[str(u)] = float(np.mean(rec['d'][k]))
            per_year_n[str(u)] = int(k.sum())

    y0 = tv[0].astype('datetime64[Y]').astype(int) + 1970
    doy0 = (tv[0].astype('datetime64[D]') - tv[0].astype('datetime64[Y]')).astype(int)
    y1 = tv[-1].astype('datetime64[Y]').astype(int) + 1970
    doy1 = (tv[-1].astype('datetime64[D]') - tv[-1].astype('datetime64[Y]')).astype(int)
    mid_year = ((y0 + doy0 / 365.25) + (y1 + doy1 / 365.25)) / 2.0

    return dict(
        analysis_start=str(np.datetime_as_string(tv[0], unit='D')),
        analysis_end=str(np.datetime_as_string(tv[-1], unit='D')),
        analysis_start_month=first_ym, analysis_end_month=last_ym,
        n_nocturnal_steps=int(valid.sum()),
        n_nocturnal_possible=int(night.sum()),
        n_nights_with_data=int(len(np.unique(rec['night_id'][valid]))),
        span_days=int((tv[-1] - tv[0]).astype('timedelta64[D]').astype(int)) + 1,
        mid_year=float(mid_year),
        months_of_year_covered=int(sum(1 for mm in range(1, 13) if moy_counts[str(mm)] > 0)),
        months_of_year_sampled=len(moy_sampled),
        moy_sampled_list=moy_sampled,
        n_calendar_months_with_data=int(sum(1 for v in per_ym.values() if v[0] > 0)),
        n_calendar_months_with_opportunity=len(per_ym),
        monthly_valid_rate_median=float(np.median(rates_use)),
        monthly_valid_rate_min=float(np.min(rates_all)),
        monthly_valid_rate_p25=float(np.percentile(rates_use, 25)),
        monthly_valid_rate_mean=float(np.mean(rates_all)),
        n_months_rate_below_0p5=int(sum(1 for r in rates_all if r < 0.5)),
        edge_months_excluded_from_median=[first_ym, last_ym],
        moy_step_counts=moy_counts, moy_night_counts=moy_nights,
        season_step_counts=season_counts,
        season_months_sampled=season_months_sampled,
        four_season_record=bool(all(season_months_sampled[s] >= 1 for s in SEASONS)),
        per_calendar_month_valid_possible=per_ym,
        per_year_nocturnal_mean=per_year_mean, per_year_n=per_year_n,
        interannual_std_of_annual_means=(float(np.std(list(per_year_mean.values()), ddof=1))
                                         if len(per_year_mean) > 1 else None),
        interannual_range_of_annual_means=(float(np.ptp(list(per_year_mean.values())))
                                           if len(per_year_mean) > 1 else None),
        native_timestep_s=rec['native_timestep_s'],
        corpus_timestep_s=rec['step_s'],
        ffill_duplicated_from_hourly=bool(rec['native_timestep_s'] == 3600),
        corpus_axis_max_gap_s=rec['max_gap_s'],
    )


# ------------------------------------------------------ weighted record means ---
def weighted_means(rec, cov):
    """Record means under pooled / month-equal / season-equal weighting."""
    valid, d, t = rec['valid'], rec['d'], rec['t']
    moy = (t.astype('datetime64[M]').astype(int) % 12) + 1
    pooled = float(np.mean(d[valid]))

    moy_mean, moy_n = {}, {}
    for mm in cov['moy_sampled_list']:
        k = valid & (moy == mm)
        moy_mean[mm] = float(np.mean(d[k]))
        moy_n[mm] = int(k.sum())
    month_equal = float(np.mean([moy_mean[mm] for mm in sorted(moy_mean)])) if moy_mean else float('nan')

    seas_raw, seas_from_month = {}, {}
    for s in SEASONS:
        mms = [mm for mm in cov['moy_sampled_list'] if SEASON_OF_MONTH[mm] == s]
        if not mms:
            continue
        k = valid & np.isin(moy, mms)
        seas_raw[s] = float(np.mean(d[k]))
        seas_from_month[s] = float(np.mean([moy_mean[mm] for mm in mms]))
    season_equal = (float(np.mean([seas_raw[s] for s in SEASONS if s in seas_raw]))
                    if len(seas_raw) == 4 else float('nan'))
    season_equal_mb = (float(np.mean([seas_from_month[s] for s in SEASONS if s in seas_from_month]))
                       if len(seas_from_month) == 4 else float('nan'))
    return dict(pooled=pooled, month_equal=month_equal,
                season_equal=season_equal, season_equal_monthbalanced=season_equal_mb,
                moy_mean={str(k): v for k, v in moy_mean.items()},
                moy_n={str(k): v for k, v in moy_n.items()},
                season_mean=seas_raw)


# ------------------------------------------------- autocorrelation / block len ---
def night_series(rec, field='d'):
    """Per-night (mean, count, calendar-day index, month-of-year) for valid nights."""
    valid, arr, nid, t = rec['valid'], rec[field], rec['night_id'], rec['t']
    ok = valid & np.isfinite(arr)
    ids = nid[ok]
    vals = arr[ok]
    uniq, inv = np.unique(ids, return_inverse=True)
    ssum = np.bincount(inv, weights=vals)
    scnt = np.bincount(inv).astype(float)
    # global index of each night's FIRST valid timestep (stable sort keeps time order)
    order = np.argsort(ids, kind='stable')
    pos = np.where(ok)[0][order]
    first_idx = pos[np.searchsorted(ids[order], uniq)]
    day = t[first_idx].astype('datetime64[D]')
    day_index = (day - day[0]).astype('timedelta64[D]').astype(np.int64)
    moy = (t[first_idx].astype('datetime64[M]').astype(int) % 12) + 1
    return dict(mean=ssum / scnt, sum=ssum, cnt=scnt, day_index=day_index,
                moy=moy, n_nights=int(uniq.size))


def acf_and_timescale(nmean, day_index, moy_index=None, max_lag=40):
    """Lag-k ACF in units of CALENDAR NIGHTS (gaps respected) + integral timescale.

    Returns the raw ACF (seasonal cycle retained) and, if moy_index is given, the
    ACF of month-of-year anomalies (seasonal cycle removed = synoptic persistence).
    Integral timescale tau = 1 + 2*sum rho_k over the initial positive sequence,
    i.e. Sokal's automatic windowing. tau is the number of consecutive nights that
    carry the information of one independent night.
    """
    out = {}
    anom = None
    if moy_index is not None:
        clim = {int(m): float(np.mean(nmean[moy_index == m])) for m in np.unique(moy_index)}
        anom = nmean - np.array([clim[int(m)] for m in moy_index])
    for tag, series in (('raw', nmean), ('anom', anom)):
        if series is None:
            continue
        a = series - series.mean()
        v = float(np.mean(a * a))
        lookup = {int(k): i for i, k in enumerate(day_index)}
        rho, npairs = [], []
        for lag in range(1, max_lag + 1):
            prod = [a[i] * a[lookup[int(k) + lag]] for i, k in enumerate(day_index)
                    if int(k) + lag in lookup]
            if len(prod) < 20:
                rho.append(float('nan')); npairs.append(len(prod)); continue
            rho.append(float(np.mean(prod) / v) if v > 0 else float('nan'))
            npairs.append(len(prod))
        tau, kcut = 1.0, 0
        for lag, r in enumerate(rho, start=1):
            if not np.isfinite(r) or r <= 0:
                break
            tau += 2.0 * r
            kcut = lag
        out[tag] = dict(rho=[None if not np.isfinite(r) else round(r, 5) for r in rho],
                        n_pairs=npairs, tau_nights=float(tau), window_lag=int(kcut),
                        variance=v)
    return out


def step_lag1(rec):
    """Lag-1 Pearson autocorrelation of the 30-min nocturnal series WITHIN nights.

    Pairs are consecutive timesteps belonging to the same night only, so the
    day-time break never enters. Normalised as a Pearson r over the pair set (not
    by the whole-record variance), so it cannot exceed 1.
    """
    valid, d, nid = rec['valid'], rec['d'], rec['night_id']
    idx = np.where(valid)[0]
    v = d[valid]
    same = (nid[idx[1:]] == nid[idx[:-1]]) & (idx[1:] == idx[:-1] + 1)
    if same.sum() < 50:
        return float('nan')
    return pear(v[:-1][same], v[1:][same])[0]


# ---------------------------------------------------------------- bootstraps ---
def boot_night(ns, rng, draws):
    """i.i.d. night bootstrap of the record mean (count-weighted, as the estimator is)."""
    K = ns['n_nights']
    idx = rng.integers(0, K, size=(draws, K))
    return ns['sum'][idx].sum(1) / ns['cnt'][idx].sum(1)


def boot_block(ns, rng, draws, L):
    """Circular moving-block bootstrap over the night sequence, block length L nights."""
    K = ns['n_nights']
    L = int(max(1, min(L, K)))
    nb = int(np.ceil(K / L))
    starts = rng.integers(0, K, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % K
    idx = idx.reshape(draws, nb * L)[:, :K]
    return ns['sum'][idx].sum(1) / ns['cnt'][idx].sum(1)


def boot_block_paired(ns_a, ns_b, rng, draws, L):
    """Same block resample applied to two night-aligned series (obs and a model)."""
    K = ns_a['n_nights']
    L = int(max(1, min(L, K)))
    nb = int(np.ceil(K / L))
    starts = rng.integers(0, K, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % K
    idx = idx.reshape(draws, nb * L)[:, :K]
    return (ns_a['sum'][idx].sum(1) / ns_a['cnt'][idx].sum(1),
            ns_b['sum'][idx].sum(1) / ns_b['cnt'][idx].sum(1))


def paired_night_series(rec):
    """Night sums/counts for obs and TEB restricted to nights where BOTH are valid."""
    ok = rec['valid'] & np.isfinite(rec['d']) & np.isfinite(rec['teb'])
    if ok.sum() < MIN_N:
        return None
    ids = rec['night_id'][ok]
    uniq, inv = np.unique(ids, return_inverse=True)
    out = {}
    for tag, arr in (('obs', rec['d'][ok]), ('teb', rec['teb'][ok])):
        out[tag] = dict(sum=np.bincount(inv, weights=arr),
                        cnt=np.bincount(inv).astype(float), n_nights=int(uniq.size))
    out['obs']['day_index'] = out['teb']['day_index'] = None
    out['n_steps'] = int(ok.sum())
    return out


# -------------------------------------------------------------------- driver ---
def variant_sets(sites):
    return {
        'all19': list(sites),
        'excl_mpls17': [s for s in sites if s not in HEIGHT_MISMATCH],
        'core16': [s for s in sites if s not in HEIGHT_MISMATCH and s != 'PL-Lipowa'],
    }


def relation(alb, off):
    r, p = pear(alb, off)
    return dict(n=int(len(alb)), pearson_r=r, pearson_p=p,
                slope_K_per_0p1_albedo=ols_slope_per01(alb, off))


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=5000)
    ap.add_argument('--prop-draws', type=int, default=20000)
    args = ap.parse_args()

    all_sites = sorted(p.stem for p in CORPUS.glob('*.nc'))
    recs, covs, wms, albedo = {}, {}, {}, {}
    print('Loading records ...', flush=True)
    for s in all_sites:
        r = load_record(s)
        if r is None:
            print(f'  {s}: skipped (no usable obs_LWup / n<{MIN_N})')
            continue
        recs[s] = r
        covs[s] = coverage(r)
        wms[s] = weighted_means(r, covs[s])
        albedo[s] = sitedata(s).get('average_albedo_at_midday', float('nan'))
        print(f'  {s:18s} n={covs[s]["n_nocturnal_steps"]:6d} '
              f'nights={covs[s]["n_nights_with_data"]:5d} '
              f'{covs[s]["analysis_start"]}..{covs[s]["analysis_end"]} '
              f'moy={covs[s]["months_of_year_covered"]:2d} '
              f'rate={covs[s]["monthly_valid_rate_median"]:.3f} '
              f'pooled={wms[s]["pooled"]:+.3f} monthEq={wms[s]["month_equal"]:+.3f}', flush=True)
    sites = sorted(recs)
    SETS = variant_sets(sites)
    assert len(sites) == 19, f'expected 19 evaluable records, got {len(sites)}'

    # ---------------- (2) weighting variants ---------------------------------
    four_season = [s for s in sites if covs[s]['four_season_record']]
    weighting = {'record_sets': {k: v for k, v in SETS.items()},
                 'four_season_records': four_season,
                 'four_season_excluded': {
                     s: {'season_months_sampled': covs[s]['season_months_sampled'],
                         'analysis_window': [covs[s]['analysis_start'], covs[s]['analysis_end']],
                         'reason': 'no calendar month with >=%d sampled nights in season(s) %s'
                                   % (MIN_NIGHTS_PER_MONTH,
                                      ','.join(sn for sn in SEASONS
                                               if covs[s]['season_months_sampled'][sn] == 0))}
                     for s in sites if s not in four_season},
                 'min_nights_per_month_for_sampled': MIN_NIGHTS_PER_MONTH,
                 'variants': {}}
    for scheme in ('pooled', 'month_equal', 'season_equal', 'season_equal_monthbalanced',
                   'pooled_on_four_season_subset', 'month_equal_on_four_season_subset'):
        field = scheme.replace('_on_four_season_subset', '')
        pool = (sites if scheme in ('pooled', 'month_equal') else four_season)
        block = {}
        for vname, ss in SETS.items():
            use = [s for s in ss if s in pool and np.isfinite(wms[s][field])]
            block[vname] = relation([albedo[s] for s in use], [wms[s][field] for s in use])
            block[vname]['sites'] = use
        weighting['variants'][scheme] = block
    weighting['isolating_weighting_from_subsample'] = {
        'why': 'the season-equal variants can only be computed on the 4-season records, '
               'so comparing them with the 19-record pooled relation mixes a weighting '
               'change with a subsample change. pooled_on_four_season_subset applies the '
               'ORIGINAL pooled weighting to exactly those records, isolating each effect.',
        'subsample_effect_r': {
            v: weighting['variants']['pooled_on_four_season_subset'][v]['pearson_r']
               - weighting['variants']['pooled'][v]['pearson_r'] for v in SETS},
        'weighting_effect_r': {
            v: weighting['variants']['season_equal'][v]['pearson_r']
               - weighting['variants']['pooled_on_four_season_subset'][v]['pearson_r']
            for v in SETS},
        'month_equal_weighting_effect_r_full_sample': {
            v: weighting['variants']['month_equal'][v]['pearson_r']
               - weighting['variants']['pooled'][v]['pearson_r'] for v in SETS},
    }
    weighting['per_record_means'] = {
        s: {k: wms[s][k] for k in ('pooled', 'month_equal', 'season_equal',
                                   'season_equal_monthbalanced')} for s in sites}
    weighting['month_equal_minus_pooled'] = {
        s: float(wms[s]['month_equal'] - wms[s]['pooled']) for s in sites}
    weighting['season_equal_minus_pooled'] = {
        s: (float(wms[s]['season_equal'] - wms[s]['pooled'])
            if np.isfinite(wms[s]['season_equal']) else None) for s in sites}

    # ---------------- (3) autocorrelation + temporal CI ----------------------
    print('\nAutocorrelation and temporal bootstrap ...', flush=True)
    acorr, tci, reps = {}, {}, {}
    for i, s in enumerate(sites):
        rec = recs[s]
        ns = night_series(rec, 'd')
        ac = acf_and_timescale(ns['mean'], ns['day_index'], ns['moy'])
        K = ns['n_nights']
        L_syn = int(max(1, min(np.ceil(ac['anom']['tau_nights']), max(1, K // 5))))
        L_raw = int(max(1, min(np.ceil(ac['raw']['tau_nights']), max(1, K // 5))))
        L_max = max(1, K // 5)      # widest block a K-night record can support
        rng = np.random.default_rng(SEED_BASE + 1000 * i)
        bn = boot_night(ns, rng, args.draws)
        bs = boot_block(ns, rng, args.draws, L_syn)
        br = boot_block(ns, rng, args.draws, L_raw)
        bx = boot_block(ns, rng, args.draws, L_max)
        acorr[s] = dict(
            lag1_within_night_30min=step_lag1(rec),
            acf_nights_raw=ac['raw']['rho'][:15], tau_nights_raw=ac['raw']['tau_nights'],
            window_lag_raw=ac['raw']['window_lag'],
            acf_nights_anom=ac['anom']['rho'][:15], tau_nights_anom=ac['anom']['tau_nights'],
            window_lag_anom=ac['anom']['window_lag'], max_lag_evaluated=40,
            tau_raw_is_lower_bound=bool(ac['raw']['window_lag'] >= 40),
            tau_anom_is_lower_bound=bool(ac['anom']['window_lag'] >= 40),
            block_length_synoptic_nights=L_syn, block_length_seasonal_nights=L_raw,
            block_cap_nights=max(1, K // 5),
            cap_binds_synoptic=bool(np.ceil(ac['anom']['tau_nights']) > max(1, K // 5)),
            cap_binds_seasonal=bool(np.ceil(ac['raw']['tau_nights']) > max(1, K // 5)),
            n_nights=K, n_blocks_synoptic=int(np.ceil(K / L_syn)),
            n_blocks_seasonal=int(np.ceil(K / L_raw)))
        tci[s] = dict(
            mean=float(np.mean(rec['d'][rec['valid']])),
            n_nights=K, n_steps=int(rec['valid'].sum()),
            night_iid=dict(se=float(bn.std(ddof=1)),
                           ci95=[float(np.percentile(bn, 2.5)), float(np.percentile(bn, 97.5))]),
            block_synoptic=dict(L=L_syn, se=float(bs.std(ddof=1)),
                                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]),
            block_seasonal=dict(L=L_raw, se=float(br.std(ddof=1)),
                                ci95=[float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]),
            block_maximal=dict(L=L_max, se=float(bx.std(ddof=1)),
                               ci95=[float(np.percentile(bx, 2.5)), float(np.percentile(bx, 97.5))],
                               note='L = K/5, the widest block this record supports; a '
                                    'deliberately over-conservative upper bound used because '
                                    'the seasonal-cycle integral timescale is truncated at the '
                                    'longest lag evaluated (40 nights) at 12 of 19 records, so '
                                    'the seasonal-block L is itself a lower bound'))
        reps[s] = dict(night=bn, syn=bs, seas=br, maxb=bx)
        print(f'  {s:18s} K={K:5d} rho1step={acorr[s]["lag1_within_night_30min"]:.3f} '
              f'tau_anom={ac["anom"]["tau_nights"]:6.2f} tau_raw={ac["raw"]["tau_nights"]:7.2f} '
              f'L={L_syn}/{L_raw} SE(night)={tci[s]["night_iid"]["se"]:.4f} '
              f'SE(syn)={tci[s]["block_synoptic"]["se"]:.4f} '
              f'SE(seas)={tci[s]["block_seasonal"]["se"]:.4f} '
              f'SE(max,L={L_max})={tci[s]["block_maximal"]["se"]:.4f}', flush=True)

    # ---------------- (4a) propagation into r and slope ----------------------
    print('\nPropagating per-record temporal uncertainty into r and slope ...', flush=True)
    prop = {'draws': args.prop_draws,
            'note': 'each draw replaces every record mean by one of its own temporal '
                    'bootstrap replicates, drawn independently across records; albedo '
                    'is held fixed. This is the record-mean sampling uncertainty the '
                    'city-cluster bootstrap does not represent.'}
    rng = np.random.default_rng(SEED_BASE + 77)
    for kind in ('night', 'syn', 'seas', 'maxb'):
        blk = {}
        for vname, ss in SETS.items():
            alb = np.array([albedo[s] for s in ss], float)
            base = np.array([tci[s]['mean'] for s in ss], float)
            M = np.empty((args.prop_draws, len(ss)))
            for j, s in enumerate(ss):
                pick = rng.integers(0, reps[s][kind].size, size=args.prop_draws)
                M[:, j] = reps[s][kind][pick]
            am = alb - alb.mean()
            sxx = float((am * am).sum())
            Mc = M - M.mean(1, keepdims=True)
            sxy = Mc @ am
            slope = sxy / sxx * 0.1
            rr = sxy / np.sqrt(sxx * (Mc * Mc).sum(1))
            b = relation(alb, base)
            blk[vname] = dict(
                baseline_r=b['pearson_r'], baseline_slope_K_per_0p1_albedo=b['slope_K_per_0p1_albedo'],
                r_mean=float(rr.mean()), r_median=float(np.median(rr)),
                r_ci95=[float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))],
                r_worst_2p5pct=float(np.percentile(rr, 97.5)),
                slope_median=float(np.median(slope)),
                slope_ci95=[float(np.percentile(slope, 2.5)), float(np.percentile(slope, 97.5))],
                frac_r_le_minus_0p5=float(np.mean(rr <= -0.5)),
                frac_r_ge_0=float(np.mean(rr >= 0.0)),
                frac_slope_ge_0=float(np.mean(slope >= 0.0)),
                attenuation_r_median_over_baseline=float(np.median(rr) / b['pearson_r']))
            print(f'  {kind:5s} {vname:12s} r {b["pearson_r"]:+.4f} -> med {np.median(rr):+.4f} '
                  f'[{np.percentile(rr, 2.5):+.4f},{np.percentile(rr, 97.5):+.4f}]  '
                  f'slope {b["slope_K_per_0p1_albedo"]:+.3f} -> '
                  f'[{np.percentile(slope, 2.5):+.3f},{np.percentile(slope, 97.5):+.3f}]  '
                  f'P(r>=0)={np.mean(rr >= 0):.4f}', flush=True)
        prop[f'propagated_{kind}'] = blk

    # variance decomposition: temporal noise vs across-site signal
    vdec = {}
    for vname, ss in SETS.items():
        for kind, key in (('night', 'night_iid'), ('syn', 'block_synoptic'),
                          ('seas', 'block_seasonal'), ('maxb', 'block_maximal')):
            se = np.array([tci[s][key]['se'] for s in ss])
            off = np.array([tci[s]['mean'] for s in ss])
            noise = float(np.mean(se ** 2))
            total = float(np.var(off, ddof=1))
            vdec.setdefault(vname, {})[kind] = dict(
                mean_temporal_variance=noise, across_record_variance=total,
                noise_to_signal=noise / total,
                reliability=1.0 - noise / total,
                expected_attenuation_sqrt_reliability=float(np.sqrt(max(0.0, 1.0 - noise / total))),
                median_record_se=float(np.median(se)), max_record_se=float(np.max(se)),
                max_record_se_site=ss[int(np.argmax(se))])
    prop['variance_decomposition'] = vdec

    # combined: city-cluster resample AND temporal perturbation
    comb = {}
    rng = np.random.default_rng(SEED_BASE + 991)
    for vname, ss in SETS.items():
        by_cl = {}
        for s in ss:
            by_cl.setdefault(cluster_of(s), []).append(s)
        cls = list(by_cl)
        rs, sl = [], []
        for _ in range(args.prop_draws // 4):
            pick = rng.integers(0, len(cls), size=len(cls))
            xs, ys = [], []
            for i in pick:
                for s in by_cl[cls[i]]:
                    xs.append(albedo[s])
                    ys.append(reps[s]['seas'][rng.integers(0, reps[s]['seas'].size)])
            xs = np.array(xs); ys = np.array(ys)
            if np.unique(xs).size < 2:
                continue
            rs.append(pearsonr(xs, ys)[0]); sl.append(np.polyfit(xs, ys, 1)[0] * 0.1)
        rs = np.array(rs); sl = np.array(sl)
        comb[vname] = dict(n_clusters=len(cls), draws=int(rs.size),
                           r_median=float(np.median(rs)),
                           r_ci95=[float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
                           slope_median=float(np.median(sl)),
                           slope_ci95=[float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))],
                           frac_r_ge_0=float(np.mean(rs >= 0.0)))
    prop['combined_cluster_and_temporal_seasonal'] = comb

    # ---------------- (4b) TEB spread-ratio propagation ----------------------
    print('\nTEB spread-ratio propagation (paired nights) ...', flush=True)
    teb = {'note': 'obs and TEB record means are resampled with the SAME night blocks '
                   '(paired), so the across-record spread ratio inherits the temporal '
                   'sampling uncertainty of both. sd/range/iqr ratio = model/obs.',
           'model': 'TEB', 'block_length_source': 'per-record seasonal-block L'}
    pair, pair_reps = {}, {}
    for i, s in enumerate(sites):
        pp = paired_night_series(recs[s])
        if pp is None:
            continue
        L = acorr[s]['block_length_seasonal_nights']
        rng2 = np.random.default_rng(SEED_BASE + 5000 + 11 * i)
        a, b = boot_block_paired(pp['obs'], pp['teb'], rng2, args.draws, L)
        pair[s] = dict(obs_mean_paired=float(pp['obs']['sum'].sum() / pp['obs']['cnt'].sum()),
                       teb_mean_paired=float(pp['teb']['sum'].sum() / pp['teb']['cnt'].sum()),
                       n_paired_steps=pp['n_steps'], n_paired_nights=pp['obs']['n_nights'],
                       block_L=L,
                       obs_se=float(a.std(ddof=1)), teb_se=float(b.std(ddof=1)))
        pair_reps[s] = (a, b)
    teb['per_record_paired'] = pair
    for vname, ss in SETS.items():
        use = [s for s in ss if s in pair_reps]
        o0 = np.array([pair[s]['obs_mean_paired'] for s in use])
        m0 = np.array([pair[s]['teb_mean_paired'] for s in use])
        base = dict(sd_ratio=float(np.std(m0, ddof=1) / np.std(o0, ddof=1)),
                    range_ratio=float(np.ptp(m0) / np.ptp(o0)),
                    iqr_ratio=float((np.percentile(m0, 75) - np.percentile(m0, 25)) /
                                    (np.percentile(o0, 75) - np.percentile(o0, 25))),
                    r_model_obs=pear(o0, m0)[0],
                    slope_model_on_obs=float(np.polyfit(o0, m0, 1)[0]))
        O = np.column_stack([pair_reps[s][0] for s in use])
        M = np.column_stack([pair_reps[s][1] for s in use])
        sd = M.std(1, ddof=1) / O.std(1, ddof=1)
        rg = (M.max(1) - M.min(1)) / (O.max(1) - O.min(1))
        iq = ((np.percentile(M, 75, axis=1) - np.percentile(M, 25, axis=1)) /
              (np.percentile(O, 75, axis=1) - np.percentile(O, 25, axis=1)))
        Oc = O - O.mean(1, keepdims=True); Mc = M - M.mean(1, keepdims=True)
        rr = (Oc * Mc).sum(1) / np.sqrt((Oc * Oc).sum(1) * (Mc * Mc).sum(1))
        bt = ((Oc * Mc).sum(1) / (Oc * Oc).sum(1))
        def ci(a):
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        teb[vname] = dict(sites=use, baseline=base, draws=args.draws,
                          sd_ratio_median=float(np.median(sd)), sd_ratio_ci95=ci(sd),
                          range_ratio_median=float(np.median(rg)), range_ratio_ci95=ci(rg),
                          iqr_ratio_median=float(np.median(iq)), iqr_ratio_ci95=ci(iq),
                          r_model_obs_median=float(np.median(rr)), r_model_obs_ci95=ci(rr),
                          slope_model_on_obs_median=float(np.median(bt)),
                          slope_model_on_obs_ci95=ci(bt),
                          frac_sd_ratio_ge_1=float(np.mean(sd >= 1.0)),
                          frac_slope_ge_1=float(np.mean(bt >= 1.0)))
        print(f'  {vname:12s} sd_ratio {base["sd_ratio"]:.3f} '
              f'[{ci(sd)[0]:.3f},{ci(sd)[1]:.3f}]  P(sd>=1)={np.mean(sd >= 1.0):.4f}', flush=True)

    # ---------------- (5) DJF / single-season / epoch reproduction -----------
    print('\nReproducing DJF / single-season / epoch tests ...', flush=True)
    def season_subset_mean(rec, months, exclude=False):
        moy = (rec['t'].astype('datetime64[M]').astype(int) % 12) + 1
        sel = np.isin(moy, months)
        k = rec['valid'] & (~sel if exclude else sel)
        return (float(np.mean(rec['d'][k])) if k.sum() >= MIN_N else float('nan'), int(k.sum()))

    repro = {'min_n_rule': MIN_N}
    for tag, months, excl in (('djf_excluded', [12, 1, 2], True),
                              ('djf_only', [12, 1, 2], False),
                              ('jja_only', [6, 7, 8], False),
                              ('mam_only', [3, 4, 5], False),
                              ('son_only', [9, 10, 11], False)):
        vals, ns = {}, {}
        for s in sites:
            v, n = season_subset_mean(recs[s], months, excl)
            vals[s] = v; ns[s] = n
        block = {'per_record_mean': vals, 'per_record_n': ns,
                 'records_dropped_below_min_n': [s for s in sites if not np.isfinite(vals[s])]}
        for vname, ss in SETS.items():
            use = [s for s in ss if np.isfinite(vals[s])]
            block[vname] = relation([albedo[s] for s in use], [vals[s] for s in use])
        repro[tag] = block
        print(f'  {tag:14s} all19 r={block["all19"]["pearson_r"]:+.4f} '
              f'slope={block["all19"]["slope_K_per_0p1_albedo"]:+.4f} n={block["all19"]["n"]} | '
              f'excl r={block["excl_mpls17"]["pearson_r"]:+.4f} | '
              f'core r={block["core16"]["pearson_r"]:+.4f}', flush=True)

    epoch = {'per_record_mid_year': {s: covs[s]['mid_year'] for s in sites},
             'mid_year_min': min(covs[s]['mid_year'] for s in sites),
             'mid_year_min_site': min(sites, key=lambda s: covs[s]['mid_year']),
             'mid_year_max': max(covs[s]['mid_year'] for s in sites),
             'mid_year_max_site': max(sites, key=lambda s: covs[s]['mid_year'])}
    for vname, ss in SETS.items():
        alb = [albedo[s] for s in ss]
        off = [tci[s]['mean'] for s in ss]
        my = [covs[s]['mid_year'] for s in ss]
        pr, pp = partial_r(alb, off, my)
        r0, p0 = pear(alb, off)
        rmy, pmy = pear(my, off)
        ramy, pamy = pear(my, alb)
        epoch[vname] = dict(raw_r=r0, raw_p=p0, partial_r_given_mid_year=pr, partial_p=pp,
                            r_offset_mid_year=rmy, p_offset_mid_year=pmy,
                            r_albedo_mid_year=ramy, p_albedo_mid_year=pamy)
    repro['epoch'] = epoch
    print(f'  epoch partial r = ' + ' / '.join(
        f'{epoch[v]["partial_r_given_mid_year"]:+.4f}' for v in SETS) +
        '  (raw ' + ' / '.join(f'{epoch[v]["raw_r"]:+.4f}' for v in SETS) + ')', flush=True)

    # ---------------- (6) PL-Narutowicza regime step ------------------------
    print('\nPL-Narutowicza regime step ...', flush=True)
    naru = narutowicza_step(recs, covs, albedo, SETS, tci)

    # ---------------- assemble ----------------------------------------------
    payload = {
        '_merge_target': 'review5_additions.temporal_sampling',
        '_supersedes': [
            'per_site.<SITE>.period_start',
            'per_site.<SITE>.period_end',
            'elimination.djf_exclusion',
        ],
        '_supersedes_notes': {
            'per_site.<SITE>.period_start/period_end':
                'paper_stats.py takes these from the FULL corpus time axis, which '
                'includes the 6-month pre-spin-up block that is masked out of every '
                'statistic. Table 1 should print the analysis window '
                '(coverage.<SITE>.analysis_start/analysis_end) instead. The two differ '
                'at every record.',
            'elimination.djf_exclusion':
                'that key holds a DJF-exclusion test of the MODEL LWup bias at three '
                'records only; reproduction_checks.djf_excluded here applies DJF '
                'exclusion to the OBSERVED albedo-offset relation at all 19, which is '
                'what the reviewer asked for. Keep both, but cite this one for the '
                'observational claim.',
        },
        'meta': {
            'script': 'analysis/revalidation_2026-08/scripts/review5_temporal_seasonal.py',
            'companion_script_for_clmu': 'review5_model_spread_propagation.py --model clmu',
            'generated_for': 'review round 5, reviewer issue 4 (seasonal/temporal sampling)',
            'SIG': SIG, 'EPS': EPS, 'seed_base': SEED_BASE,
            'bootstrap_draws_per_record': args.draws,
            'propagation_draws': args.prop_draws,
            'n_records': len(sites), 'records': sites,
            'record_sets': SETS,
            'independence': 'masks read directly from the corpus NetCDF (not via '
                            'src.training.corpus_loader), seed base 20260916 (paper_stats.py '
                            'uses 20260904), every number recomputed from NetCDF; '
                            'results/paper_stats_v1.json is never read or written',
            'clmu_excluded': 'CLM-Urban statistics are omitted on purpose: the CLMU output '
                             'alignment is being corrected at GR-HECKOR and UK-KingsCollege. '
                             'Run review5_model_spread_propagation.py --model clmu afterwards.',
            'hourly_native_records': [s for s in sites if covs[s]['ffill_duplicated_from_hourly']],
            'hourly_native_note': 'JP-Yoyogi, PL-Lipowa, PL-Narutowicza, US-Baltimore are '
                                  '60-min records (Lipson 2022 ESSD Table 1, asterisk) '
                                  'forward-filled to the 30-min corpus grid, so consecutive '
                                  '30-min samples are duplicates and n overstates the '
                                  'independent sample size by 2x. The night-level and block '
                                  'bootstraps are immune because the duplication is entirely '
                                  'within nights; a timestep-level bootstrap would not be.',
        },
        'coverage': covs,
        'weighting': weighting,
        'autocorrelation': acorr,
        'record_mean_temporal_ci': tci,
        'propagation': prop,
        'teb_spread_propagation': teb,
        'clmu_spread_propagation': {
            'status': 'DEFERRED -- not computed this round',
            'reason': 'CLMU output-alignment correction in flight (GR-HECKOR, UK-KingsCollege); '
                      'every CLMU-derived statistic would change.',
            'how_to_fill': 'python review5_model_spread_propagation.py --model clmu '
                           '--out results/review5_temporal-seasonal_clmu.json '
                           '(reads external/clmu_baseline19/<SITE>_base.nc with the corrected '
                           'alignment offsets; identical paired night-block machinery, '
                           'identical seeds, identical block lengths)',
        },
        'reproduction_checks': repro,
        'narutowicza_regime_step': naru,
    }
    # -------- diagnostics the reviewer's question makes worth stating ---------
    m1 = recs['US-Minneapolis1']; m2 = recs['US-Minneapolis2']
    same_mpls = bool(np.array_equal(m1['valid'], m2['valid']) and
                     np.allclose(m1['d'][m1['valid']], m2['d'][m2['valid']],
                                 rtol=0, atol=1e-12))
    order = sorted(sites, key=lambda s: -tci[s]['block_seasonal']['se'])
    payload['diagnostics'] = {
        'minneapolis_pair_shares_one_radiation_record': same_mpls,
        'minneapolis_pair_note':
            'US-Minneapolis1 and 2 are one radiation record split by wind sector '
            '(Lipson 2022 ESSD Sect. 4.5, Table 7), so their nocturnal dTsa series are '
            'identical' + ('' if same_mpls else ' -- BUT THEY ARE NOT, CHECK THIS') +
            '. Their temporal standard errors are therefore identical in expectation and '
            'differ here only by Monte-Carlo noise between per-record seeds.',
        'records_ranked_by_temporal_se_seasonal_block': [
            [s, round(tci[s]['block_seasonal']['se'], 4)] for s in order],
        'largest_temporal_se_site': order[0],
        'temporal_se_flags_narutowicza':
            'the seasonal-block temporal SE independently singles out PL-Narutowicza '
            '(%.3f K, %.1fx the median of %.3f K) without being told about the regime '
            'step: the step makes the nightly series persistently autocorrelated out to '
            '%d nights, which is exactly what a long block length and a wide temporal '
            'interval are meant to expose.'
            % (tci['PL-Narutowicza']['block_seasonal']['se'],
               tci['PL-Narutowicza']['block_seasonal']['se'] /
               float(np.median([tci[s]['block_seasonal']['se'] for s in sites])),
               float(np.median([tci[s]['block_seasonal']['se'] for s in sites])),
               acorr['PL-Narutowicza']['window_lag_raw']),
        'interannual_std_of_annual_means': {
            s: covs[s]['interannual_std_of_annual_means'] for s in sites},
        'interannual_excursions': {
            s: dict(record_mean=tci[s]['mean'],
                    worst_year=max(covs[s]['per_year_nocturnal_mean'],
                                   key=lambda y: abs(covs[s]['per_year_nocturnal_mean'][y]
                                                     - tci[s]['mean'])),
                    worst_year_mean=covs[s]['per_year_nocturnal_mean'][
                        max(covs[s]['per_year_nocturnal_mean'],
                            key=lambda y: abs(covs[s]['per_year_nocturnal_mean'][y]
                                              - tci[s]['mean']))],
                    per_year=covs[s]['per_year_nocturnal_mean'])
            for s in sites
            if len(covs[s]['per_year_nocturnal_mean']) > 1
            and max(abs(v - tci[s]['mean'])
                    for v in covs[s]['per_year_nocturnal_mean'].values()) > 0.5},
        'interannual_excursions_note':
            'records whose worst calendar-year nocturnal mean departs from the record '
            'mean by more than 0.5 K. PL-Narutowicza is the 2010/2011 regime step. '
            'CA-Sunset is a single-year excursion (2014) that returns to the earlier '
            'level in 2015, i.e. interannual variability rather than a level shift. '
            'The US-Minneapolis 2006 value covers December only and is a coverage '
            'artefact of a partial first year.',
    }
    payload['table1_new_columns'] = {
        s: {'start': covs[s]['analysis_start'], 'end': covs[s]['analysis_end'],
            'months_of_12': covs[s]['months_of_year_covered'],
            'monthly_valid_rate_median_pct': round(100 * covs[s]['monthly_valid_rate_median'], 1)}
        for s in sites}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1, ensure_ascii=False, default=float)

    def leaves(o):
        if isinstance(o, dict):
            return sum(leaves(v) for v in o.values())
        if isinstance(o, (list, tuple)):
            return sum(leaves(v) for v in o)
        return 1
    print(f'\nwrote {OUT}  leaves={leaves(payload)}')


# ------------------------------------------------------- Narutowicza module ---
def narutowicza_step(recs, covs, albedo, SETS, tci):
    """Locate, size and attribute the PL-Narutowicza regime step; build both-regime
    and drop-the-record sensitivity values for the headline relation."""
    S, P = 'PL-Narutowicza', 'PL-Lipowa'
    rn, rl = recs[S], recs[P]
    assert np.array_equal(rn['t'], rl['t']), 'Lodz pair time axes differ'

    # nightly series of the Narutowicza-minus-Lipowa LWup difference: removes the
    # seasonal cycle and the regional synoptic signal, isolating an instrument offset
    ds = xr.open_dataset(CORPUS / f'{S}.nc')
    olw_n = np.asarray(ds['obs_LWup'].values, float)
    ld_n = np.asarray(ds['forcing_LWdown'].values, float)
    ta_n = np.asarray(ds['forcing_Tair'].values, float)
    ds.close()
    ds = xr.open_dataset(CORPUS / f'{P}.nc')
    olw_l = np.asarray(ds['obs_LWup'].values, float)
    ld_l = np.asarray(ds['forcing_LWdown'].values, float)
    ta_l = np.asarray(ds['forcing_Tair'].values, float)
    ds.close()

    both = (rn['night'] & rl['night'] & np.isfinite(olw_n) & np.isfinite(olw_l)
            & np.isfinite(ld_n) & np.isfinite(ld_l) & np.isfinite(ta_n) & np.isfinite(ta_l))
    t = rn['t']
    day = t.astype('datetime64[D]')

    def daily(vals, mask):
        d, idx = np.unique(day[mask], return_inverse=True)
        s = np.bincount(idx, weights=vals[mask])
        c = np.bincount(idx).astype(float)
        return d, s / c, c

    dd, dif_lwup, _ = daily(olw_n - olw_l, both)
    _, dif_lwdown, _ = daily(ld_n - ld_l, both)
    _, dif_tair, _ = daily(ta_n - ta_l, both)

    # single-changepoint scan on the paired LWup difference (least squares)
    def scan(dates, y):
        n = y.size
        cs = np.concatenate(([0.0], np.cumsum(y)))
        cs2 = np.concatenate(([0.0], np.cumsum(y * y)))
        best, bi = np.inf, None
        for i in range(30, n - 30):
            s1, s2 = cs[i], cs[n] - cs[i]
            q1, q2 = cs2[i], cs2[n] - cs2[i]
            sse = (q1 - s1 * s1 / i) + (q2 - s2 * s2 / (n - i))
            if sse < best:
                best, bi = sse, i
        return bi, str(dates[bi]), float(y[:bi].mean()), float(y[bi:].mean())

    i_lw, date_lw, pre_lw, post_lw = scan(dd, dif_lwup)
    i_ld, date_ld, pre_ld, post_ld = scan(dd, dif_lwdown)
    i_ta, date_ta, pre_ta, post_ta = scan(dd, dif_tair)
    # the control: the SAME split date applied to the inter-tower Tair difference
    k_lw = dd >= np.datetime64(date_lw)
    ta_pre_at_lw = float(dif_tair[~k_lw].mean())
    ta_post_at_lw = float(dif_tair[k_lw].mean())

    # the record's own nocturnal dTsa, split at the LWup changepoint
    cut = np.datetime64(date_lw)
    vn = rn['valid']
    early = vn & (day < cut)
    late = vn & (day >= cut)
    dtsa_early = float(np.mean(rn['d'][early]))
    dtsa_late = float(np.mean(rn['d'][late]))

    # gap around the break
    dvals = np.unique(day[vn])
    j = int(np.searchsorted(dvals, cut))
    gap_days = [str(x) for x in dvals[max(0, j - 2):j + 2]]

    # Lipowa split at the same date: does the neighbour step too?
    vl = rl['valid']
    lip_early = float(np.mean(rl['d'][vl & (day < cut)]))
    lip_late = float(np.mean(rl['d'][vl & (day >= cut)]))

    # how much of the LWdown difference is exactly zero (i.e. gap-filled FROM Lipowa)?
    dz = (ld_n - ld_l)[both]
    frac_zero = float(np.mean(np.abs(dz) < 1e-9))
    nzd = both & (np.abs(ld_n - ld_l) >= 1e-9)
    pre_nz = float(np.mean((ld_n - ld_l)[nzd & (day < cut)]))
    post_nz = float(np.mean((ld_n - ld_l)[nzd & (day >= cut)]))

    # per-year nocturnal means, both records
    years = {}
    for y in np.unique(t.astype('datetime64[Y]')):
        ky = (t.astype('datetime64[Y]') == y)
        kn = vn & ky
        if kn.sum() == 0:
            continue
        kb = both & ky
        years[str(y)] = dict(
            n=int(kn.sum()), naru_dtsa=float(np.mean(rn['d'][kn])),
            naru_lwup=float(np.mean(olw_n[kn])), naru_tair=float(np.mean(ta_n[kn])),
            lipowa_dtsa=float(np.mean(rl['d'][vl & ky])) if (vl & ky).sum() else None,
            diff_lwup=float(np.mean((olw_n - olw_l)[kb])) if kb.sum() else None,
            diff_lwdown=float(np.mean((ld_n - ld_l)[kb])) if kb.sum() else None,
            diff_tair=float(np.mean((ta_n - ta_l)[kb])) if kb.sum() else None)

    # sensitivity: headline relation with Narutowicza represented three ways
    out_rel = {}
    for tag, repl in (('as_published_full_record', tci[S]['mean']),
                      ('early_regime_only', dtsa_early),
                      ('late_regime_only', dtsa_late),
                      ('dropped', None)):
        block = {}
        for vname, ss in SETS.items():
            use = [s for s in ss if not (s == S and repl is None)]
            off = [(repl if (s == S and repl is not None) else tci[s]['mean']) for s in use]
            r, p = pear([albedo[s] for s in use], off)
            block[vname] = dict(n=len(use), pearson_r=r, pearson_p=p,
                                slope_K_per_0p1_albedo=ols_slope_per01([albedo[s] for s in use], off))
        out_rel[tag] = block

    return dict(
        record=S, albedo=albedo[S],
        changepoint=dict(
            method='single-changepoint least-squares scan over the nightly '
                   'Narutowicza-minus-Lipowa paired difference (same city, 2 km apart, '
                   'identical time axis), which removes the seasonal cycle and the '
                   'regional synoptic signal and leaves an instrumental offset',
            lwup_break_date=date_lw, lwup_pre_Wm2=pre_lw, lwup_post_Wm2=post_lw,
            lwup_step_Wm2=post_lw - pre_lw,
            lwdown_break_date=date_ld, lwdown_pre_Wm2=pre_ld, lwdown_post_Wm2=post_ld,
            lwdown_step_Wm2=post_ld - pre_ld,
            tair_break_date=date_ta, tair_pre_K=pre_ta, tair_post_K=post_ta,
            tair_step_K=post_ta - pre_ta,
            tair_step_at_lwup_break_K=ta_post_at_lw - ta_pre_at_lw,
            tair_pre_at_lwup_break_K=ta_pre_at_lw, tair_post_at_lwup_break_K=ta_post_at_lw,
            control_note='the inter-tower Tair difference does not step at the radiation '
                         'break date (%.3f K), and its own best changepoint sits 21 months '
                         'later with an even smaller amplitude, so the step is confined to '
                         'the two longwave channels.' % (ta_post_at_lw - ta_pre_at_lw),
            nights_either_side=gap_days,
            n_paired_nights=int(dif_lwup.size)),
        magnitude=dict(
            dtsa_full_record=tci[S]['mean'],
            dtsa_early_regime=dtsa_early, dtsa_late_regime=dtsa_late,
            dtsa_step_K=dtsa_late - dtsa_early,
            n_early_steps=int(early.sum()), n_late_steps=int(late.sum()),
            lipowa_early=lip_early, lipowa_late=lip_late,
            lipowa_step_K=lip_late - lip_early,
            note='the published record mean (+1.19 K) is a weighted average of two '
                 'regimes (%.2f K and %.2f K); no night in the record sits near it'
                 % (dtsa_early, dtsa_late)),
        attribution=dict(
            variable='obs_LWup and forcing_LWdown step together by a similar amount; '
                     'Tair does not step',
            lwdown_frac_exactly_equal_to_lipowa=frac_zero,
            lwdown_diff_excluding_filled_pre_Wm2=pre_nz,
            lwdown_diff_excluding_filled_post_Wm2=post_nz,
            physical_argument='LWdown is a sky-viewing measurement: two towers 2 km '
                              'apart in the same city must see almost the same sky. A '
                              'pre-break nocturnal LWdown difference of %.1f W m-2 '
                              'between them is not atmospherically attainable, whereas '
                              'the post-break %.1f W m-2 is within pyrgeometer offset '
                              'tolerance. Both components of the same 4-component net '
                              'radiometer (CNR1 class at the Lodz sites) stepping '
                              'together is the signature of an instrument or '
                              'calibration change, not of a surface change.'
                              % (pre_ld, post_ld),
            per_year=years),
        documentation_search=dict(
            conclusion='NOT DOCUMENTED anywhere in the collection or its data descriptor',
            checked=[
                dict(source='PL-Narutowicza NetCDF global attributes (history, comment) in '
                            'data/urban-plumber/FullCollection/PL-Narutowicza/timeseries/',
                     found='history = "v0.9 (2021-09-08): beta issue; v1 (2022-09-15): with '
                           'publication in ESSD"; comment = "Missing forcing filled with '
                           'PL-Lipowa tower site where available. Precipitation from IMGW '
                           'Lodz Lublinek." No instrument change, offset or discontinuity.'),
                dict(source='the collection\'s own per-site report page '
                            'FullCollection/PL-Narutowicza/index.html',
                     found='site characteristics, QC protocol and bias-correction plots; no '
                           'note of an instrument change, and PL-Narutowicza is absent from '
                           'the excluded-wind-sector table.'),
                dict(source='Lipson et al. 2022, Earth Syst. Sci. Data 14, 5157-5178',
                     url='https://doi.org/10.5194/essd-14-5157-2022',
                     found='Table 1 lists PL-Narutowicza as Jan 2008-Dec 2012 with the '
                           'asterisk that denotes 60-min resolution. The five QC steps '
                           '(out-of-range, night, constant, outlier at +/-4 SD in a rolling '
                           '30-day window, visual) contain no test for a sustained level '
                           'shift, and the words "step change" and "recalibration" do not '
                           'appear in the paper. The only inhomogeneity discussed is '
                           'spatial (excluded wind sectors, Table 7), which does not list '
                           'this site. Table 5 defines '
                           'measurement_height_above_ground as the height of the EDDY '
                           'COVARIANCE equipment, and notes that parameter 19 '
                           '(average_albedo_at_midday) is the only one that is a function '
                           'of the radiometer field of view.'),
                dict(source='urban-plumber_pipeline issue tracker',
                     url='https://github.com/matlipson/urban-plumber_pipeline/issues',
                     found='one issue in total (#1, "Can I calculate Net Ecosystem Exchange '
                           'from the provided dataset"); nothing on Narutowicza, Lodz or '
                           'longwave offsets.'),
                dict(source='site processing script create_dataset_PL-Narutowicza.py',
                     url='https://github.com/matlipson/urban-plumber_pipeline/blob/main/'
                         'sites/PL-Narutowicza/create_dataset_PL-Narutowicza.py',
                     found='restricts the record ("USE SUBSET ONLY (a lot of missing data in '
                           '2001)", df.loc[\'2008\':\'2012\']) and gap-fills forcing from the '
                           'PL-Lipowa tower; no radiometer, offset or 2010/2011 note.'),
                dict(source='Fortuniak, Pawlak and Siedlecki 2013, Boundary-Layer Meteorol. '
                            '146, 257-276 (the observations_reference for this site)',
                     url='https://doi.org/10.1007/s10546-012-9762-1',
                     found='the site reference is a turbulence-statistics paper; the Lodz '
                           'radiation measurements are made with a Kipp & Zonen CNR1-class '
                           '4-component net radiometer, but no instrument change at either '
                           'Lodz site around 2010/2011 is reported in the literature we '
                           'could reach.'),
            ],
            why_qc_missed_it='the collection\'s outlier test is +/-4 SD within a rolling '
                             '30-day window, so a sustained level shift is an outlier for at '
                             'most the first few days and is thereafter its own norm; no '
                             'homogeneity or changepoint test is applied anywhere in the '
                             'Urban-PLUMBER v1 pipeline.'),
        headline_sensitivity=out_rel,
        decision='RETAIN the record at its published full-record mean and report both '
                 'regimes as a sensitivity (lead decision). The drop-it variant is '
                 'supplied for completeness.',
    )


if __name__ == '__main__':
    main()
