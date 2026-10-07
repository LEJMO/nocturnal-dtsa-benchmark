"""DOUBLE-CHECK AUDIT of the dTsa observational finding (user-mandated).
Attacks tested here (the ones NOT yet covered by KG1/sign-audit):
 [1] GAP-FILL CONTAMINATION: % of nocturnal LWup samples that are qc=0 observed
     vs qc=1 (obs-interp) vs qc=2 (ERA5-derived); recompute dTsa with qc0-only
     (LWup qc0; and strictest: LWup&LWdown&Tair all qc0).
 [2] CORPUS-BUILDER INTEGRITY: replicate the builder alignment from raw files and
     compare against corpus obs_LWup (max abs diff).
 [3] MEASUREMENT-HEIGHT ARTIFACT: correlate dTsa with sensor height; partial vs albedo.
 [4] INDEPENDENT-INSTRUMENT PHYSICS: obs nocturnal Qh (sonic, qc0) vs dTsa across
     sites (expect positive relation: warm surface -> upward H).
 [5] RANGE SANITY + Lipowa seasonal stability (obs qc0).
"""
import sys, csv, io
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.training.corpus_loader import load_all_sites
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[3]
SIG, EPS = 5.67e-8, 0.95
CANON = 1800

def sitedata(site):
    p = ROOT / f'data/urban-plumber/FullCollection/{site}/{site}_sitedata_v1.csv'
    vals = {}
    for row in csv.DictReader(p.open(encoding='utf-8')):
        try: vals[row['parameter']] = float(row['value'])
        except Exception: pass
    return vals

def load_raw(site):
    ts = ROOT / f'data/urban-plumber/FullCollection/{site}/timeseries'
    mf = xr.open_dataset(ts / f'{site}_metforcing_v1.nc').squeeze(drop=True)
    co = xr.open_dataset(ts / f'{site}_clean_observations_v1.nc').squeeze(drop=True)
    obs_t = pd.to_datetime(co.time.values)
    mfw = mf.sel(time=slice(obs_t[0], obs_t[-1]))
    cow = co.sel(time=slice(obs_t[0], obs_t[-1]))
    native = int(np.median(np.diff(mf.time.values).astype('timedelta64[s]').astype(int)))
    if native == 2*CANON:
        mfw = mfw.resample(time=f'{CANON}s').ffill()
        cow = cow.resample(time=f'{CANON}s').ffill()
    return mfw, cow

rows = []
for rec in load_all_sites():
    ds = xr.open_dataset(ROOT / f'data/urban-plumber/corpus/{rec.site}.nc')
    if 'obs_LWup' not in ds: ds.close(); continue
    c_olw = ds['obs_LWup'].values; c_ta = ds['forcing_Tair'].values; c_ld = ds['forcing_LWdown'].values
    ds.close()
    night = np.asarray(rec.night_mask, bool) & (~np.asarray(rec.pre_spinup_flag, bool))
    try:
        mfw, cow = load_raw(rec.site)
    except Exception as e:
        print(f'{rec.site}: RAW LOAD FAIL {e}'); continue
    n = min(len(c_olw), cow.sizes.get('time', 0))
    r_lw = np.asarray(cow['LWup'].values, float)[:n] if 'LWup' in cow else np.full(n, np.nan)
    q_lw = np.asarray(cow['LWup_qc'].values, float)[:n] if 'LWup_qc' in cow else np.full(n, np.nan)
    q_ld = np.asarray(mfw['LWdown_qc'].values, float)[:n] if 'LWdown_qc' in mfw else np.full(n, np.nan)
    q_ta = np.asarray(mfw['Tair_qc'].values, float)[:n] if 'Tair_qc' in mfw else np.full(n, np.nan)
    m = night[:n] & np.isfinite(c_olw[:n]) & np.isfinite(c_ta[:n]) & np.isfinite(c_ld[:n])
    # [2] builder integrity
    both = m & np.isfinite(r_lw)
    maxdiff = float(np.nanmax(np.abs(r_lw[both] - c_olw[:n][both]))) if both.any() else np.nan
    # [1] qc composition on the evaluation mask
    tot = m.sum()
    f0 = float((q_lw[m] == 0).mean()); f1 = float((q_lw[m] == 1).mean()); f2 = float((q_lw[m] == 2).mean())
    def dtsa(mask):
        ts = ((c_olw[:n][mask] - (1-EPS)*c_ld[:n][mask]) / (EPS*SIG))**0.25
        return float((ts - c_ta[:n][mask]).mean()), int(mask.sum())
    d_all, n_all = dtsa(m)
    m0 = m & (q_lw == 0)
    d_q0, n_q0 = dtsa(m0) if m0.sum() > 50 else (np.nan, int(m0.sum()))
    mstrict = m0 & (q_ta == 0) & (q_ld == 0)
    d_st, n_st = dtsa(mstrict) if mstrict.sum() > 50 else (np.nan, int(mstrict.sum()))
    # [4] obs Qh qc0 nocturnal mean
    qh = np.asarray(cow['Qh'].values, float)[:n] if 'Qh' in cow else np.full(n, np.nan)
    qq = np.asarray(cow['Qh_qc'].values, float)[:n] if 'Qh_qc' in cow else np.full(n, np.nan)
    mq = night[:n] & np.isfinite(qh) & (qq == 0)
    qh0 = float(qh[mq].mean()) if mq.sum() > 50 else np.nan
    # [5] range sanity
    lw_mean = float(c_olw[:n][m].mean())
    sd = sitedata(rec.site)
    rows.append(dict(site=rec.site, maxdiff=maxdiff, f0=f0, f1=f1, f2=f2,
                     d_all=d_all, d_q0=d_q0, d_st=d_st, n_all=n_all, n_q0=n_q0, n_st=n_st,
                     qh0=qh0, lw_mean=lw_mean,
                     zh=sd.get('measurement_height_above_ground', np.nan),
                     alb=sd.get('average_albedo_at_midday', np.nan)))

print('=== [1]+[2] GAP-FILL COMPOSITION & BUILDER INTEGRITY (nocturnal eval mask) ===')
print(f"{'site':16s} {'corpus=raw?':>11s} {'%obs':>5s} {'%interp':>7s} {'%ERA5':>6s} | {'dTsa_all':>8s} {'dTsa_qc0':>8s} {'dTsa_strict':>11s} {'n_qc0':>6s}")
for r in sorted(rows, key=lambda x: x['d_all']):
    print(f"{r['site']:16s} {r['maxdiff']:11.2e} {100*r['f0']:5.1f} {100*r['f1']:7.1f} {100*r['f2']:6.1f} | "
          f"{r['d_all']:+8.2f} {r['d_q0']:+8.2f} {r['d_st']:+11.2f} {r['n_q0']:6d}")

d_all = np.array([r['d_all'] for r in rows]); d_q0 = np.array([r['d_q0'] for r in rows])
alb = np.array([r['alb'] for r in rows]); zh = np.array([r['zh'] for r in rows])
ok = np.isfinite(d_q0)
print(f"\nspan all-samples: {d_all.min():+.2f}..{d_all.max():+.2f} | qc0-only: {d_q0[ok].min():+.2f}..{d_q0[ok].max():+.2f}")
r1,p1 = pearsonr(alb, d_all); r2,p2 = pearsonr(alb[ok], d_q0[ok])
print(f"r(dTsa, albedo): all {r1:+.3f} (p={p1:.4f})  qc0 {r2:+.3f} (p={p2:.4f})")
print(f"max |dTsa_all - dTsa_qc0| across sites: {np.nanmax(np.abs(d_all-d_q0)):.2f} K")

print('\n=== [3] MEASUREMENT-HEIGHT ARTIFACT CHECK ===')
okz = np.isfinite(zh)
r3,p3 = pearsonr(zh[okz], d_all[okz])
print(f"sensor height range: {np.nanmin(zh):.1f}..{np.nanmax(zh):.1f} m")
print(f"r(dTsa, sensor_height) = {r3:+.3f} (p={p3:.4f})")
X = np.column_stack([zh[okz], np.ones(okz.sum())])
res_d = d_all[okz] - X @ np.linalg.lstsq(X, d_all[okz], rcond=None)[0]
res_a = alb[okz] - X @ np.linalg.lstsq(X, alb[okz], rcond=None)[0]
r4,p4 = pearsonr(res_a, res_d)
print(f"partial r(dTsa, albedo | height) = {r4:+.3f} (p={p4:.4f})")

print('\n=== [4] INDEPENDENT-INSTRUMENT CONSISTENCY: obs nocturnal Qh (sonic, qc0) vs dTsa ===')
qh0 = np.array([r['qh0'] for r in rows]); okq = np.isfinite(qh0)
r5,p5 = pearsonr(d_all[okq], qh0[okq])
print(f"r(dTsa, obs_Qh_night) = {r5:+.3f} (p={p5:.4f})  n={okq.sum()}")
for r in sorted(rows, key=lambda x: x['d_all']):
    if np.isfinite(r['qh0']):
        print(f"  {r['site']:16s} dTsa {r['d_all']:+5.2f}  obsQh {r['qh0']:+7.1f} W/m2")

print('\n=== [5] RANGE SANITY + LIPOWA SEASONAL (qc0) ===')
for r in rows:
    flag = 'OK' if 250 < r['lw_mean'] < 470 else 'SUSPICIOUS'
    if flag != 'OK': print(f"  {r['site']}: nocturnal LWup mean {r['lw_mean']:.0f} W/m2 {flag}")
print('  (all sites within 250-470 W/m2 unless listed above)')
# Lipowa seasonal
rec = [x for x in load_all_sites() if x.site == 'PL-Lipowa'][0]
ds = xr.open_dataset(ROOT / 'data/urban-plumber/corpus/PL-Lipowa.nc')
olw = ds['obs_LWup'].values; ta = ds['forcing_Tair'].values; ld = ds['forcing_LWdown'].values
season = ds['season'].values if 'season' in ds else None
ds.close()
mfw, cow = load_raw('PL-Lipowa')
nn = min(len(olw), cow.sizes['time'])
qlw = np.asarray(cow['LWup_qc'].values, float)[:nn]
night = np.asarray(rec.night_mask, bool) & (~np.asarray(rec.pre_spinup_flag, bool))
m = night[:nn] & np.isfinite(olw[:nn]) & (qlw == 0)
ts = ((olw[:nn] - (1-EPS)*ld[:nn]) / (EPS*SIG))**0.25
d = ts - ta[:nn]
if season is not None:
    for s_id, s_name in enumerate(['DJF','MAM','JJA','SON']):
        ms = m & (season[:nn] == s_id)
        if ms.sum() > 50:
            print(f"  Lipowa {s_name}: dTsa(qc0) {d[ms].mean():+.2f} K (n={ms.sum()})")
else:
    print('  (season variable absent in corpus; skipped)')
