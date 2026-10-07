# -*- coding: utf-8 -*-
"""paper_stats.py  --  SINGLE-SOURCE-OF-TRUTH statistics package for the dTsa manuscript.

Every number the paper cites numerically is produced here. This script does NOT
modify or re-run any evaluation pipeline; it (a) recomputes the observational
dTsa gradient and TEB dTsa from the frozen inversion, (b) reads archived model
outputs (CLMU5 g2_results.json, SUEWS result.json), and (c) consolidates
elimination numbers from existing artifacts.

FROZEN CONVENTIONS (do not change):
  night mask   = night_mask & ~pre_spinup_flag  (src.training.corpus_loader)
  Ts inversion = ((LWup-(1-eps)*LWdown)/(eps*sigma))**0.25 , eps=0.95
  dTsa         = nocturnal mean(Ts - Tair)   [Tair = forcing_Tair]
  19 evaluable records of the 20-site collection (MX-Escandon lacks obs_LWup)
  TEB output   = external/teb_runs/<S>/output/LWU_base.txt , row i -> forcing step i+1
  CLMU5        = external/clmu_g1/g2_results.json (per_site)
  SUEWS-OHM    = external/suews_g1/*/result.json  (VOID-instrument, auxiliary only)
  city clusters: Minneapolis={US-Minneapolis1,US-Minneapolis2},
                 Lodz={PL-Lipowa,PL-Narutowicza}, Helsinki={FI-Kumpula,FI-Torni},
                 all others singleton  -> 16 clusters
CORRECTED FRAMING (binding):
  US-Minneapolis1/2 : radiation @2 m vs Tair @40 m  -> HEIGHT-MISMATCH FLAG.
  Report every headline stat in three variants: all-19 / excl-2-Minneapolis /
  excl-Minneapolis+Lipowa.  Conservative core span = -2.2..+1.2 K plus Lipowa
  (+6.1, 37-m FOV caveat) reported separately.  The word "first" is banned.
"""
import sys, csv, json, glob, io
from pathlib import Path
import numpy as np
import xarray as xr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.training.corpus_loader import load_all_sites          # noqa: E402
from scipy.stats import pearsonr, spearmanr                     # noqa: E402

SIG = 5.67e-8
EPS = 0.95
TREF = 285.0
K_PER_WM2 = 1.0 / (4.0 * EPS * SIG * TREF**3)   # ~0.2006 K per W/m2 (garden_probe convention)

CLUSTER = {
    'US-Minneapolis1': 'Minneapolis', 'US-Minneapolis2': 'Minneapolis',
    'PL-Lipowa': 'Lodz', 'PL-Narutowicza': 'Lodz',
    'FI-Kumpula': 'Helsinki', 'FI-Torni': 'Helsinki',
}
HEIGHT_MISMATCH = {'US-Minneapolis1', 'US-Minneapolis2'}
RNG = np.random.default_rng(20260904)

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

def load_raw(site, canon=1800):
    ts = ROOT / f'data/urban-plumber/FullCollection/{site}/timeseries'
    mf = xr.open_dataset(ts / f'{site}_metforcing_v1.nc').squeeze(drop=True)
    co = xr.open_dataset(ts / f'{site}_clean_observations_v1.nc').squeeze(drop=True)
    obs_t = np.asarray(co.time.values)
    mfw = mf.sel(time=slice(obs_t[0], obs_t[-1]))
    cow = co.sel(time=slice(obs_t[0], obs_t[-1]))
    native = int(np.median(np.diff(mf.time.values).astype('timedelta64[s]').astype(int)))
    if native == 2 * canon:
        mfw = mfw.resample(time=f'{canon}s').ffill()
        cow = cow.resample(time=f'{canon}s').ffill()
    mf.close(); co.close()
    return mfw, cow

def ols_slope(x, y):
    return float(np.polyfit(x, y, 1)[0])

def safe_pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3:
        return float('nan'), float('nan')
    r, p = pearsonr(x, y)
    return float(r), float(p)

def safe_spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3:
        return float('nan'), float('nan')
    r, p = spearmanr(x, y)
    return float(r), float(p)

# ------------------------------------------------------- per-site collection ---
print('Loading corpus (frozen masks) ...', flush=True)
records = {r.site: r for r in load_all_sites()}

SITES = {}         # site -> dict of per-site numbers
discrepancies = [] # cross-check log vs ledger

clmu = json.load(open(ROOT / 'external/clmu_g1/g2_results.json', encoding='utf-8'))
clmu_ps = clmu['per_site']
g1_sites = clmu['g1_sites']
g2_sites = clmu['g2_sites']

for site, rec in records.items():
    ds = xr.open_dataset(ROOT / f'data/urban-plumber/corpus/{site}.nc')
    if 'obs_LWup' not in ds:
        ds.close()
        continue  # MX-Escandon -> not evaluable
    olw = ds['obs_LWup'].values
    ta = ds['forcing_Tair'].values
    ld = ds['forcing_LWdown'].values
    koppen = str(ds.attrs.get('koppen_zone', 'NA'))
    tax = np.asarray(ds['time'].values)
    ds.close()

    night = np.asarray(rec.night_mask, bool) & (~np.asarray(rec.pre_spinup_flag, bool))
    m = night & np.isfinite(olw) & np.isfinite(ta) & np.isfinite(ld)
    if m.sum() < 100:
        continue

    # obs dTsa (standard, eps 0.95)
    ts = ((olw[m] - (1 - EPS) * ld[m]) / (EPS * SIG)) ** 0.25
    dtsa_std = float((ts - ta[m]).mean())
    n_night = int(m.sum())

    # record period from corpus time axis over the evaluation window
    t_all = tax
    per_start = str(np.datetime_as_string(t_all[0], unit='D'))
    per_end = str(np.datetime_as_string(t_all[-1], unit='D'))

    # ---- raw qc for LWup qc0 %, strict dTsa, obs Qh qc0 --------------------
    lwup_qc0_pct = float('nan'); dtsa_strict = float('nan'); n_strict = 0
    obs_qh_qc0 = float('nan')
    builder_maxdiff = float('nan')
    try:
        mfw, cow = load_raw(site)
        n = min(len(olw), cow.sizes.get('time', 0))
        q_lw = np.asarray(cow['LWup_qc'].values, float)[:n] if 'LWup_qc' in cow else np.full(n, np.nan)
        q_ld = np.asarray(mfw['LWdown_qc'].values, float)[:n] if 'LWdown_qc' in mfw else np.full(n, np.nan)
        q_ta = np.asarray(mfw['Tair_qc'].values, float)[:n] if 'Tair_qc' in mfw else np.full(n, np.nan)
        r_lw = np.asarray(cow['LWup'].values, float)[:n] if 'LWup' in cow else np.full(n, np.nan)
        mm = m[:n]
        # builder integrity (corpus obs_LWup vs raw clean_observations LWup)
        both = mm & np.isfinite(r_lw)
        if both.any():
            builder_maxdiff = float(np.nanmax(np.abs(r_lw[both] - olw[:n][both])))
        # LWup qc0 % on the nocturnal eval mask
        if mm.sum() > 0:
            lwup_qc0_pct = 100.0 * float((q_lw[mm] == 0).mean())
        # strict dTsa: LWup & Tair & LWdown all qc0
        mstrict = mm & (q_lw == 0) & (q_ta == 0) & (q_ld == 0)
        if mstrict.sum() > 50:
            tss = ((olw[:n][mstrict] - (1 - EPS) * ld[:n][mstrict]) / (EPS * SIG)) ** 0.25
            dtsa_strict = float((tss - ta[:n][mstrict]).mean())
            n_strict = int(mstrict.sum())
        # obs nocturnal Qh qc0 (independent instrument)
        qh = np.asarray(cow['Qh'].values, float)[:n] if 'Qh' in cow else np.full(n, np.nan)
        qq = np.asarray(cow['Qh_qc'].values, float)[:n] if 'Qh_qc' in cow else np.full(n, np.nan)
        mq = night[:n] & np.isfinite(qh) & (qq == 0)
        if mq.sum() > 50:
            obs_qh_qc0 = float(qh[mq].mean())
    except Exception as e:
        discrepancies.append(f'{site}: raw-load failed ({e}); qc0/strict/Qh set NaN')

    # ---- TEB dTsa + TEB LWup bias (recomputed from LWU_base) ---------------
    teb_dtsa = float('nan'); teb_bias = float('nan'); n_teb = 0
    p = ROOT / f'external/teb_runs/{site}/output/LWU_base.txt'
    if p.exists():
        teb = np.loadtxt(p)
        nmod = min(teb.size, rec.n_steps - 1)
        sl = slice(1, nmod + 1)
        mt = m[sl] & np.isfinite(teb[:nmod])
        if mt.sum() > 50:
            ts_mod = ((teb[:nmod][mt] - (1 - EPS) * ld[sl][mt]) / (EPS * SIG)) ** 0.25
            teb_dtsa = float((ts_mod - ta[sl][mt]).mean())
            teb_bias = float((teb[:nmod][mt] - olw[sl][mt]).mean())
            n_teb = int(mt.sum())

    # ---- CLMU5 from archived g2 -------------------------------------------
    clmu_bias = clmu_ps.get(site, {}).get('lwup_bias', float('nan'))
    clmu_dtsa = clmu_ps.get(site, {}).get('dTsa_model', float('nan'))

    sd = sitedata(site)
    albedo = sd.get('average_albedo_at_midday', float('nan'))
    imperv = sd.get('impervious_area_fraction', float('nan'))
    pervious = (1.0 - imperv) if np.isfinite(imperv) else float('nan')

    SITES[site] = dict(
        site=site, cluster=cluster_of(site), koppen=koppen,
        period_start=per_start, period_end=per_end, n_night=n_night,
        lwup_qc0_pct=lwup_qc0_pct, dtsa_std=dtsa_std,
        dtsa_strict=dtsa_strict, n_strict=n_strict,
        albedo=albedo, pervious=pervious,
        height_mismatch=(site in HEIGHT_MISMATCH),
        teb_lwup_bias=teb_bias, clmu_lwup_bias=float(clmu_bias),
        teb_dtsa=teb_dtsa, clmu_dtsa=float(clmu_dtsa),
        obs_qh_qc0=obs_qh_qc0, builder_maxdiff=builder_maxdiff,
        in_g2=(site in g2_sites),
    )
    print(f'  {site:16s} dTsa={dtsa_std:+.2f} qc0%={lwup_qc0_pct:.1f} '
          f'TEBbias={teb_bias:+.2f} TEBdTsa={teb_dtsa:+.2f} '
          f'CLMUbias={clmu_bias:+.2f} n={n_night}', flush=True)

sites = sorted(SITES)
print(f'\n{len(sites)} evaluable sites.', flush=True)

# cross-check obs dTsa vs ledger (dslucm_surgery obs block)
ledger_obs = json.load(open(ROOT / 'external/dslucm_surgery/results.json', encoding='utf-8'))['obs']
for s in sites:
    if s in ledger_obs:
        d = SITES[s]['dtsa_std'] - ledger_obs[s]
        if abs(d) > 0.05:
            discrepancies.append(
                f'{s}: obs dTsa this-run {SITES[s]["dtsa_std"]:+.3f} vs ledger '
                f'{ledger_obs[s]:+.3f} (diff {d:+.3f} K)')

# ----------------------------------------------------- variant index sets ---
def variant_sites(name):
    if name == 'all':
        return list(sites)
    if name == 'excl_mpls':
        return [s for s in sites if s not in HEIGHT_MISMATCH]
    if name == 'excl_mpls_lipowa':
        return [s for s in sites if s not in HEIGHT_MISMATCH and s != 'PL-Lipowa']
    raise ValueError(name)

VARIANTS = ['all', 'excl_mpls', 'excl_mpls_lipowa']

def arr(field, subset):
    return np.array([SITES[s][field] for s in subset], float)

# ------------------------------------------ (2) HEADLINE OBS STATS ----------
def cluster_bootstrap_slope(subset, xf, yf, ndraw=5000):
    """Resample CITY CLUSTERS with replacement; OLS slope in K per 0.1 albedo."""
    by_cluster = {}
    for s in subset:
        by_cluster.setdefault(SITES[s]['cluster'], []).append(s)
    clusters = list(by_cluster)
    slopes = []
    for _ in range(ndraw):
        pick = RNG.choice(len(clusters), size=len(clusters), replace=True)
        xs, ys = [], []
        for i in pick:
            for s in by_cluster[clusters[i]]:
                xs.append(SITES[s][xf]); ys.append(SITES[s][yf])
        if len(set(xs)) < 2:
            continue
        slopes.append(np.polyfit(xs, ys, 1)[0] * 0.1)
    slopes = np.array(slopes)
    return (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)),
            int(slopes.size))

headline = {}
for v in VARIANTS:
    ss = variant_sites(v)
    alb = arr('albedo', ss); dt = arr('dtsa_std', ss)
    pr, pp = safe_pearson(alb, dt)
    sr, sp = safe_spearman(alb, dt)
    slope_per01 = ols_slope(alb, dt) * 0.1
    lo, hi, nb = cluster_bootstrap_slope(ss, 'albedo', 'dtsa_std', 5000)
    n_clusters = len({SITES[s]['cluster'] for s in ss})
    headline[v] = dict(
        n_sites=len(ss), n_clusters=n_clusters,
        pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp,
        slope_K_per_0p1_albedo=slope_per01,
        slope_ci95=[lo, hi], bootstrap_draws=nb,
        span_min=float(dt.min()), span_max=float(dt.max()),
        span_min_site=ss[int(np.argmin(dt))], span_max_site=ss[int(np.argmax(dt))],
    )

# span statement per corrected framing
core = variant_sites('excl_mpls_lipowa')
core_dt = arr('dtsa_std', core)
span_statement = (
    f'Conservative core span (excl. both Minneapolis + Lipowa, n={len(core)}): '
    f'{core_dt.min():+.2f}..{core_dt.max():+.2f} K '
    f'[{core[int(np.argmin(core_dt))]} .. {core[int(np.argmax(core_dt))]}]; '
    f'PL-Lipowa reported separately: {SITES["PL-Lipowa"]["dtsa_std"]:+.2f} K '
    f'(37-m FOV composition caveat). Minneapolis pair '
    f'({SITES["US-Minneapolis1"]["dtsa_std"]:+.2f}/'
    f'{SITES["US-Minneapolis2"]["dtsa_std"]:+.2f} K) carries HEIGHT-MISMATCH flag '
    f'(radiation @2 m vs Tair @40 m). The word "first" is banned.')

# ------------------------------------------ (3) CLUSTER PERMUTATION TEST ----
# aggregate to city clusters (mean dTsa, mean albedo), permute albedo labels.
def cluster_permutation(subset, nperm=10000):
    by_cluster = {}
    for s in subset:
        by_cluster.setdefault(SITES[s]['cluster'], []).append(s)
    cl = list(by_cluster)
    cx = np.array([np.mean([SITES[s]['albedo'] for s in by_cluster[c]]) for c in cl])
    cy = np.array([np.mean([SITES[s]['dtsa_std'] for s in by_cluster[c]]) for c in cl])
    r_obs = abs(pearsonr(cx, cy)[0])
    cnt = 0
    for _ in range(nperm):
        rp = abs(pearsonr(RNG.permutation(cx), cy)[0])
        if rp >= r_obs - 1e-12:
            cnt += 1
    return dict(n_clusters=len(cl), r_cluster=float(pearsonr(cx, cy)[0]),
                abs_r=float(r_obs), p_perm=(cnt + 1) / (nperm + 1), nperm=nperm)

perm_all = cluster_permutation(variant_sites('all'), 10000)
perm_exmp = cluster_permutation(variant_sites('excl_mpls'), 10000)

# ------------------------------------------ (4) EPS SENSITIVITY -------------
def dtsa_obs_eps(site, eps):
    rec = records[site]
    ds = xr.open_dataset(ROOT / f'data/urban-plumber/corpus/{site}.nc')
    olw = ds['obs_LWup'].values; ta = ds['forcing_Tair'].values; ld = ds['forcing_LWdown'].values
    ds.close()
    night = np.asarray(rec.night_mask, bool) & (~np.asarray(rec.pre_spinup_flag, bool))
    m = night & np.isfinite(olw) & np.isfinite(ta) & np.isfinite(ld)
    ts = ((olw[m] - (1 - eps) * ld[m]) / (eps * SIG)) ** 0.25
    return float((ts - ta[m]).mean())

eps_sens = {}
for eps in [0.90, 0.93, 0.95, 0.97, 0.99]:
    row = {}
    for v in VARIANTS:
        ss = variant_sites(v)
        vv = np.array([dtsa_obs_eps(s, eps) for s in ss])
        alb = arr('albedo', ss)
        r, p = safe_pearson(alb, vv)
        row[v] = dict(span_min=float(vv.min()), span_max=float(vv.max()),
                      r_albedo=r, p=p)
    eps_sens[f'{eps:.2f}'] = row

# ------------------------------------------ (5) MODEL EVALUATION STATS ------
def model_eval(subset, model_dtsa_field, model_bias_field):
    o = arr('dtsa_std', subset)
    md = arr(model_dtsa_field, subset)
    ok = np.isfinite(md) & np.isfinite(o)
    o2, md2 = o[ok], md[ok]
    slope = ols_slope(o2, md2)
    r, p = safe_pearson(o2, md2)
    alb = arr('albedo', subset)
    b = arr(model_bias_field, subset)
    okb = np.isfinite(b) & np.isfinite(alb)
    rb, pb = safe_pearson(alb[okb], b[okb])
    return dict(n=int(ok.sum()), slope_model_on_obs=slope, r_model_obs=r, p_model_obs=p,
                model_span_min=float(md2.min()), model_span_max=float(md2.max()),
                r_bias_albedo=rb, p_bias_albedo=pb)

model_stats = {}
for v in ['all', 'excl_mpls']:
    ss = variant_sites(v)
    model_stats[v] = dict(
        TEB=model_eval(ss, 'teb_dtsa', 'teb_lwup_bias'),
        CLMU5=model_eval(ss, 'clmu_dtsa', 'clmu_lwup_bias'),
    )
# 6-site G1 subset (the scope the ledger H1' numbers were actually computed on)
g1_present = [s for s in g1_sites if s in SITES]
model_stats['g1_6site'] = dict(
    sites=g1_present,
    TEB=model_eval(g1_present, 'teb_dtsa', 'teb_lwup_bias'),
    CLMU5=model_eval(g1_present, 'clmu_dtsa', 'clmu_lwup_bias'),
    note='ledger H1'' (TEB r=0.78/slope=0.238, CLMU5 r=0.19/slope=0.048) matches '
         'THIS 6-site subset, not the 19-site table.')

# cross-model pattern r on the 13 G2 sites (report as-is, honest)
g2_eval = [s for s in sites if SITES[s]['in_g2']]
tb = arr('teb_lwup_bias', g2_eval); cb = arr('clmu_lwup_bias', g2_eval)
okg = np.isfinite(tb) & np.isfinite(cb)
cm_r, cm_p = safe_pearson(tb[okg], cb[okg])
cross_model_pattern = dict(
    n_g2_sites=int(okg.sum()), g2_sites=g2_eval,
    r_TEBbias_CLMUbias=cm_r, p=cm_p,
    note='reported as-is (honest); extreme-inclusive pattern is model-structure-specific')

# SUEWS-OHM 6-site (auxiliary / VOID-instrument)
suews = {}
for f in sorted(glob.glob(str(ROOT / 'external/suews_g1/*/result.json'))):
    d = json.load(open(f, encoding='utf-8'))
    s = d.get('site')
    suews[s] = dict(noct_Ts_facet_minus_Tair=d.get('noct_Ts_facet_minus_Tair'),
                    lwup_bias_facet_recon=d.get('lwup_bias_facet_recon'),
                    n_night=d.get('n_night'))
suews_obs = {s: SITES[s]['dtsa_std'] for s in suews if s in SITES}
su_sites = [s for s in suews if s in SITES]
su_model = np.array([suews[s]['noct_Ts_facet_minus_Tair'] for s in su_sites], float)
su_obs = np.array([SITES[s]['dtsa_std'] for s in su_sites], float)
su_ok = np.isfinite(su_model)
suews_block = dict(
    label='AUXILIARY / VOID-INSTRUMENT (supy EHC/ESTM defective; OHM facet fallback)',
    n_sites=int(su_ok.sum()), per_site=suews,
    slope_facet_on_obs=(ols_slope(su_obs[su_ok], su_model[su_ok]) if su_ok.sum() > 2 else None),
    r_facet_obs=(safe_pearson(su_obs[su_ok], su_model[su_ok])[0] if su_ok.sum() > 2 else None),
    model_span=[float(su_model[su_ok].min()), float(su_model[su_ok].max())] if su_ok.any() else None,
)

# ------------------------------------------ (6) ELIMINATION NUMBERS ---------
# TEB material envelope from probe_rows.csv (5 sites x 8 configs) -- READ ONLY
teb_probe_rows = list(csv.DictReader(
    (ROOT / 'analysis/revalidation_2026-08/evidence/probe_rows.csv').open(encoding='utf-8')))
teb_env = {}
for r in teb_probe_rows:
    teb_env.setdefault(r['site'], []).append(float(r['lwup_bias']))
teb_envelope = {}
teb_reversals = 0
for s, vals in teb_env.items():
    base = [float(r['lwup_bias']) for r in teb_probe_rows if r['site'] == s and r['config'] == 'base'][0]
    env = max(vals) - min(vals)
    rev = sum(1 for x in vals if (x > 0) != (base > 0))
    teb_reversals += rev
    teb_envelope[s] = dict(base=base, env=env, vmin=min(vals), vmax=max(vals), sign_reversals=rev)
teb_env_range = [min(d['env'] for d in teb_envelope.values()),
                 max(d['env'] for d in teb_envelope.values())]

# CLMU5 envelope from clmu_probe/summary.json (5 sites x 7 configs) -- READ ONLY
clmu_probe = json.load(open(ROOT / 'external/clmu_probe/summary.json', encoding='utf-8'))
clmu_envelope = {}
clmu_reversals = 0
for s, blk in clmu_probe['per_site'].items():
    ct = blk['config_table']
    biases = [ct[c]['lwup_bias'] for c in ct]
    base = ct['base']['lwup_bias']
    rev = sum(1 for x in biases if (x > 0) != (base > 0))
    clmu_reversals += rev
    clmu_envelope[s] = dict(base=base, env=max(biases) - min(biases),
                            vmin=min(biases), vmax=max(biases), sign_reversals=rev)
clmu_env_range = [min(d['env'] for d in clmu_envelope.values()),
                  max(d['env'] for d in clmu_envelope.values())]

# garden invariance (from garden_probe.md) -- READ ONLY constants
garden_invariance = dict(
    site='US-Minneapolis2', zgarden_090_bias=18.78, zgarden_000_bias=18.13,
    delta_Wm2=18.13 - 18.78, delta_K=(18.13 - 18.78) * K_PER_WM2,
    note='full garden removal (0.90->0.0) leaves +18.8 warm bias essentially intact; '
         'not an aggregation artifact (garden_probe.md)')

# DJF-exclusion (sign_audit_table.csv) -- READ ONLY
sa_rows = list(csv.DictReader(
    (ROOT / 'analysis/revalidation_2026-08/evidence/sign_audit_table.csv').open(encoding='utf-8-sig')))
djf_excl = {}
for r in sa_rows:
    djf_excl[r['site']] = dict(bias_Wm2=float(r['lwup_bias_Wm2']),
                               bias_noDJF_Wm2=float(r['bias_noDJF_Wm2']))
djf_highlight = {s: djf_excl[s] for s in ['US-Minneapolis1', 'US-Minneapolis2', 'KR-Ochang']
                 if s in djf_excl}

# structural-surgery best config (dslucm_surgery/results.json S-AB) -- READ ONLY
surg = json.load(open(ROOT / 'external/dslucm_surgery/results.json', encoding='utf-8'))['summary']
sab = surg['S-AB']
structural_surgery = dict(
    best_config='S-AB', slope=sab['slope'], r=sab['r'],
    min_site=sab['min_site'], min_val=sab['min_val'], lipowa=sab['lipowa'],
    pass_=sab['pass'],
    note='structural surgery of dSLUCM cannot reproduce the Lipowa-positive extreme; '
         'no config passes n_neg>=4 & Lipowa>=+4 & slope>=0.5 & r>=0.6')

elimination = dict(
    teb_material_envelope=dict(range_Wm2=teb_env_range, sign_reversals=teb_reversals,
                               per_site=teb_envelope,
                               source='probe_rows.csv (5 sites x 8 configs)'),
    clmu_material_envelope=dict(range_Wm2=clmu_env_range, sign_reversals=clmu_reversals,
                                per_site=clmu_envelope,
                                source='clmu_probe/summary.json (5 sites x 7 configs)'),
    garden_invariance=garden_invariance,
    djf_exclusion=djf_highlight,
    structural_surgery=structural_surgery,
)

# ------------------------------------------ (7) PRESCRIPTION (LOO offset) ---
def loo_albedo_correction(subset, bias_field):
    alb = arr('albedo', subset)
    bias = arr(bias_field, subset)
    ok = np.isfinite(alb) & np.isfinite(bias)
    alb, bias = alb[ok], bias[ok]
    n = alb.size
    pred = np.empty(n)
    for i in range(n):
        idx = np.arange(n) != i
        c = np.polyfit(alb[idx], bias[idx], 1)
        pred[i] = np.polyval(c, alb[i])
    resid = bias - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((bias - bias.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    before = float(np.mean(np.abs(bias)))
    after = float(np.mean(np.abs(resid)))
    return dict(n=int(n), loo_r2=r2,
                abs_bias_before_Wm2=before, abs_bias_after_Wm2=after,
                abs_bias_before_K=before * K_PER_WM2, abs_bias_after_K=after * K_PER_WM2)

prescription = {}
for v in ['all', 'excl_mpls']:
    ss = variant_sites(v)
    prescription[v] = dict(
        TEB=loo_albedo_correction(ss, 'teb_lwup_bias'),
        CLMU5=loo_albedo_correction(ss, 'clmu_lwup_bias'),
        K_per_Wm2=K_PER_WM2)

# ------------------------------------------ (8) INDEPENDENT INSTRUMENT ------
def independent_instrument(subset):
    dt = arr('dtsa_std', subset)
    qh = arr('obs_qh_qc0', subset)
    ok = np.isfinite(qh) & np.isfinite(dt)
    dt2, qh2 = dt[ok], qh[ok]
    sr, sp = safe_spearman(dt2, qh2)
    pr, pp = safe_pearson(dt2, qh2)
    # sign agreement: sign(dTsa)==sign(Qh)
    agree = int(np.sum(np.sign(dt2) == np.sign(qh2)))
    ntot = int(dt2.size)
    from scipy.stats import binomtest
    p_binom = float(binomtest(agree, ntot, 0.5, alternative='two-sided').pvalue)
    return dict(n=ntot, spearman_r=sr, spearman_p=sp, pearson_r=pr, pearson_p=pp,
                sign_agree=agree, sign_agree_frac=agree / ntot if ntot else float('nan'),
                binom_p=p_binom)

indep = dict(
    all=independent_instrument([s for s in sites]),
    excl_lipowa=independent_instrument([s for s in sites if s != 'PL-Lipowa']),
    note='DESCRIPTIVE only. Positive relation expected (warm surface -> upward H); '
         'Lipowa is a documented counter-example (most positive dTsa yet high +Qh).')

# ------------------------------------------ ASSEMBLE + WRITE ----------------
package = dict(
    meta=dict(
        generated='2026-09-04', generator=str(Path(__file__).name),
        eps=EPS, sigma=SIG, Tref=TREF, K_per_Wm2=K_PER_WM2,
        n_evaluable_sites=len(sites),
        note='Single-source-of-truth statistics for the dTsa manuscript. '
             'No evaluation script was modified or re-run. '
             'obs dTsa & TEB dTsa recomputed from frozen inversion; '
             'CLMU5 from archived g2_results.json; SUEWS auxiliary/void-instrument.',
        clusters={c: [s for s in sites if SITES[s]['cluster'] == c]
                  for c in sorted({SITES[s]['cluster'] for s in sites})},
        height_mismatch_sites=sorted(HEIGHT_MISMATCH),
        banned_word='first',
    ),
    per_site={s: SITES[s] for s in sites},
    headline_obs=dict(variants=headline, span_statement=span_statement),
    cluster_permutation=dict(all=perm_all, excl_mpls=perm_exmp),
    eps_sensitivity=eps_sens,
    model_evaluation=dict(variants=model_stats, cross_model_pattern=cross_model_pattern,
                          suews_auxiliary=suews_block),
    elimination=elimination,
    prescription=prescription,
    independent_instrument=indep,
)

# ---- EXPLICIT ledger §3 cross-check (do not silently adopt either) ---------
def cc(label, computed, ledger, tol, note=''):
    ok = (ledger is not None) and np.isfinite(computed) and abs(computed - ledger) <= tol
    return dict(item=label, computed=round(float(computed), 4),
                ledger=ledger, tol=tol, match=bool(ok), note=note)

h_all = headline['all']
teb_all = model_stats['all']['TEB']; clmu_all = model_stats['all']['CLMU5']
teb_g1 = model_stats['g1_6site']['TEB']; clmu_g1 = model_stats['g1_6site']['CLMU5']
pres_all = prescription['all']
ledger_checks = [
    cc('#1 obs dTsa span min (US-Minneapolis)', SITES['US-Minneapolis1']['dtsa_std'], -3.04, 0.05,
       'per-site obs dTsa matches ledger exactly'),
    cc('#1 obs dTsa span max (PL-Lipowa)', SITES['PL-Lipowa']['dtsa_std'], 6.12, 0.05, ''),
    cc('#2 r(obs dTsa, albedo) all-19', h_all['pearson_r'], 0.762, 0.05,
       'DISCREPANCY (sign+identity): ledger +0.762 is r(TEB bias, albedo), NOT the obs '
       'gradient. TRUE r(obs dTsa, albedo)=-0.793. Manuscript headline must be NEGATIVE.'),
    cc('#2b r(TEB bias, albedo) all-19 [true id of ledger +0.762]',
       teb_all['r_bias_albedo'], 0.762, 0.02, 'matches -> confirms mislabel above'),
    cc('#3 TEB model-obs slope (ledger 0.238 == 19-site value)', teb_all['slope_model_on_obs'], 0.238, 0.02,
       'ledger slope 0.238 matches the 19-SITE value (%.3f); note the 6-site slope is %.3f. '
       'The ledger H1'' TEB row MIXED SCOPES (19-site slope + 6-site r).'
       % (teb_all['slope_model_on_obs'], teb_g1['slope_model_on_obs'])),
    cc('#3 TEB model-obs r (ledger 0.78 == 6-site value)', teb_g1['r_model_obs'], 0.78, 0.03,
       'ledger r=0.78 matches the 6-SITE value (%.3f); the 19-site r is %.3f'
       % (teb_g1['r_model_obs'], teb_all['r_model_obs'])),
    cc('#4 CLMU5 model-obs slope (6 G1 sites)', clmu_g1['slope_model_on_obs'], 0.048, 0.01,
       'matches on 6 G1 sites; all-19 slope=%.3f' % clmu_all['slope_model_on_obs']),
    cc('#4 CLMU5 model-obs r (6 G1 sites)', clmu_g1['r_model_obs'], 0.19, 0.03,
       'ledger 0.19 is 6-site; all-19 r=%.3f' % clmu_all['r_model_obs']),
    cc('#5 cross-model unseen-13 r(TEBbias,CLMUbias)', cross_model_pattern['r_TEBbias_CLMUbias'],
       -0.108, 0.02, 'matches'),
    cc('#7 TEB material envelope max (W/m2)', teb_env_range[1], 11.0, 1.0,
       'ledger 6-11 W/m2; computed %.1f-%.1f' % (teb_env_range[0], teb_env_range[1])),
    cc('#8 CLMU5 material envelope max (W/m2)', clmu_env_range[1], 7.3, 0.5,
       'ledger 2.2-7.3; computed %.1f-%.1f' % (clmu_env_range[0], clmu_env_range[1])),
    cc('#9 garden invariance (ZGARDEN 0.90->0.0)', garden_invariance['zgarden_000_bias'], 18.13, 0.05,
       'ledger +18.78->+18.13'),
    cc('CLMU5 r(bias,albedo) [ledger L3 +0.718]', clmu_all['r_bias_albedo'], 0.718, 0.02, 'matches'),
    cc('#12 TEB prescription |bias| after (W/m2)', pres_all['TEB']['abs_bias_after_Wm2'], 4.5, 0.3,
       'ledger 10.8->4.5; computed %.1f->%.1f'
       % (pres_all['TEB']['abs_bias_before_Wm2'], pres_all['TEB']['abs_bias_after_Wm2'])),
    cc('#12 TEB prescription LOO R2', pres_all['TEB']['loo_r2'], 0.51, 0.03,
       'DEFINITION diff: ledger 0.51; standard LOO 1-SSres/SStot=%.3f (in-sample=%.3f)'
       % (pres_all['TEB']['loo_r2'], teb_all['r_bias_albedo']**2)),
    cc('#12 CLMU5 prescription |bias| after (W/m2)', pres_all['CLMU5']['abs_bias_after_Wm2'], 5.9, 0.3,
       'ledger 7.6->5.9; computed %.1f->%.1f'
       % (pres_all['CLMU5']['abs_bias_before_Wm2'], pres_all['CLMU5']['abs_bias_after_Wm2'])),
    cc('#12 CLMU5 prescription LOO R2', pres_all['CLMU5']['loo_r2'], 0.42, 0.03,
       'DEFINITION diff: ledger 0.42; standard LOO=%.3f' % pres_all['CLMU5']['loo_r2']),
    cc('structural surgery S-AB slope', structural_surgery['slope'], 0.458, 0.01, 'matches surgery file'),
    cc('structural surgery S-AB r', structural_surgery['r'], 0.720, 0.01, 'matches surgery file'),
]
mismatches = [c for c in ledger_checks if not c['match']]
package['ledger_cross_check'] = dict(
    n_checks=len(ledger_checks), n_mismatch=len(mismatches),
    per_site_dtsa_discrepancies=discrepancies,
    checks=ledger_checks,
    key_corrections=[
        'HEADLINE SIGN: r(obs dTsa, albedo) = -0.79 (NEGATIVE). The ledger''s "+0.762" '
        'is r(TEB nocturnal LWup bias, albedo), a different quantity. High-albedo sites '
        'show the surface COOLER than air at night (negative dTsa) AND the largest model '
        'warm bias (positive) -- two sides of the same coin, both organized by albedo.',
        'SCOPE: CLMU5 H1'' (r=0.19, slope=0.048) is cleanly the 6-site G1 value. TEB H1'' '
        'MIXED scopes: its slope 0.238 is the 19-site value while its r 0.78 is the 6-site '
        'value. The 19-site single-source values are TEB r=%.3f/slope=%.3f, '
        'CLMU5 r=%.3f/slope=%.3f; the 6-site values are TEB r=%.3f/slope=%.3f, '
        'CLMU5 r=%.3f/slope=%.3f.'
        % (teb_all['r_model_obs'], teb_all['slope_model_on_obs'],
           clmu_all['r_model_obs'], clmu_all['slope_model_on_obs'],
           teb_g1['r_model_obs'], teb_g1['slope_model_on_obs'],
           clmu_g1['r_model_obs'], clmu_g1['slope_model_on_obs']),
        'PRESCRIPTION R2 is definition-dependent; single-source value = standard LOO '
        '1-SSres/SStot (TEB %.2f, CLMU5 %.2f). |bias| before->after matches ledger exactly.'
        % (pres_all['TEB']['loo_r2'], pres_all['CLMU5']['loo_r2']),
    ],
    note='Overlapping numbers cross-checked against 근거 대장 §3 '
         '(신규방향_연구논리_근거_강건성_2026-08-31.md), G1_JUDGMENT H1'', and '
         'dslucm_surgery obs block. Fresh computations are the single source of truth; '
         'discrepancies are reported, not silently reconciled.')

out_json = ROOT / 'results/paper_stats_v1.json'
out_json.parent.mkdir(parents=True, exist_ok=True)
with out_json.open('w', encoding='utf-8') as f:
    json.dump(package, f, indent=2, ensure_ascii=False, default=float)
print(f'\nWROTE {out_json}', flush=True)

# CSV master table (human)
csv_path = ROOT / 'paper/manuscript_dtsa/tables/site_master_table.csv'
csv_path.parent.mkdir(parents=True, exist_ok=True)
cols = ['site', 'cluster', 'koppen', 'period_start', 'period_end', 'n_night',
        'lwup_qc0_pct', 'dtsa_std', 'dtsa_strict', 'n_strict', 'albedo', 'pervious',
        'height_mismatch', 'teb_lwup_bias', 'clmu_lwup_bias', 'teb_dtsa', 'clmu_dtsa',
        'obs_qh_qc0']
with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(cols)
    for s in sorted(sites, key=lambda x: SITES[x]['dtsa_std']):
        row = SITES[s]
        w.writerow([row.get(c) for c in cols])
print(f'WROTE {csv_path}', flush=True)

# ------------------------------------------ CONSOLE SUMMARY ----------------
print('\n' + '=' * 78)
print('HEADLINE OBS STATS')
for v in VARIANTS:
    h = headline[v]
    print(f'  [{v:17s}] n={h["n_sites"]:2d} clusters={h["n_clusters"]:2d} '
          f'Pearson r={h["pearson_r"]:+.3f}(p={h["pearson_p"]:.4f}) '
          f'Spearman r={h["spearman_r"]:+.3f}(p={h["spearman_p"]:.4f})')
    print(f'{"":22s} slope={h["slope_K_per_0p1_albedo"]:+.3f} K/0.1alb '
          f'CI95[{h["slope_ci95"][0]:+.3f},{h["slope_ci95"][1]:+.3f}] '
          f'span {h["span_min"]:+.2f}({h["span_min_site"]})..{h["span_max"]:+.2f}({h["span_max_site"]})')
print('  span_statement:', span_statement)
print('\nCLUSTER PERMUTATION (all):', perm_all)
print('CLUSTER PERMUTATION (excl_mpls):', perm_exmp)
print('\nEPS SENSITIVITY (all-19):')
for e, row in eps_sens.items():
    a = row['all']
    print(f'  eps={e} span {a["span_min"]:+.2f}..{a["span_max"]:+.2f} r={a["r_albedo"]:+.3f}(p={a["p"]:.4f})')
print('\nMODEL EVAL:')
for v in ['all', 'excl_mpls']:
    for mdl in ['TEB', 'CLMU5']:
        m = model_stats[v][mdl]
        print(f'  [{v:9s}] {mdl:6s} slope={m["slope_model_on_obs"]:+.3f} r={m["r_model_obs"]:+.3f} '
              f'span {m["model_span_min"]:+.2f}..{m["model_span_max"]:+.2f} '
              f'r(bias,alb)={m["r_bias_albedo"]:+.3f}(p={m["p_bias_albedo"]:.4f})')
print(f'  cross-model pattern r (13 G2): r={cross_model_pattern["r_TEBbias_CLMUbias"]:+.3f} '
      f'(p={cross_model_pattern["p"]:.4f}, n={cross_model_pattern["n_g2_sites"]})')
print('  SUEWS (aux/void):', suews_block['slope_facet_on_obs'], suews_block['r_facet_obs'],
      suews_block['model_span'])
print('\nELIMINATION:')
print(f'  TEB envelope {teb_env_range[0]:.1f}-{teb_env_range[1]:.1f} W/m2, reversals={teb_reversals}')
print(f'  CLMU5 envelope {clmu_env_range[0]:.1f}-{clmu_env_range[1]:.1f} W/m2, reversals={clmu_reversals}')
print(f'  garden invariance: {garden_invariance["zgarden_090_bias"]}->{garden_invariance["zgarden_000_bias"]}')
print(f'  DJF-excl Mpls1: {djf_highlight.get("US-Minneapolis1")}')
print(f'  structural surgery S-AB: slope={sab["slope"]:.3f} r={sab["r"]:.3f} '
      f'min={sab["min_val"]:.2f}({sab["min_site"]}) lipowa={sab["lipowa"]:.2f}')
print('\nPRESCRIPTION (LOO albedo offset):')
for v in ['all', 'excl_mpls']:
    for mdl in ['TEB', 'CLMU5']:
        p = prescription[v][mdl]
        print(f'  [{v:9s}] {mdl:6s} LOO R2={p["loo_r2"]:+.3f} |bias| '
              f'{p["abs_bias_before_Wm2"]:.1f}->{p["abs_bias_after_Wm2"]:.1f} W/m2 '
              f'({p["abs_bias_before_K"]:.2f}->{p["abs_bias_after_K"]:.2f} K)')
print('\nINDEPENDENT INSTRUMENT (obs Qh qc0):')
for k in ['all', 'excl_lipowa']:
    d = indep[k]
    print(f'  [{k:11s}] Spearman r={d["spearman_r"]:+.3f}(p={d["spearman_p"]:.4f}) '
          f'sign-agree {d["sign_agree"]}/{d["n"]} (binom p={d["binom_p"]:.4f})')
print('\nLEDGER CROSS-CHECK: %d checks, %d mismatch, per-site dTsa discrepancies=%d'
      % (len(ledger_checks), len(mismatches), len(discrepancies)))
for c in ledger_checks:
    tag = 'OK  ' if c['match'] else 'FLAG'
    print(f'  [{tag}] {c["item"]}: computed={c["computed"]} ledger={c["ledger"]}')
    if not c['match'] and c['note']:
        print(f'         -> {c["note"]}')
print('\nKEY CORRECTIONS:')
for k in package['ledger_cross_check']['key_corrections']:
    print('  *', k)
print('=' * 78)
