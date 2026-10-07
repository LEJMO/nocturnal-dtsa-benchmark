"""KG1 - Radiometric robustness of the dTsa gradient (run-now version).
(a) eps sweep 0.90..0.99 on OBS dTsa: span + albedo correlation must survive.
(b) ADVERSARIAL per-site eps (eps assigned as monotone function of albedo, both
    directions, range 0.90-0.98): can plausible eps heterogeneity destroy the
    gradient/correlation?
(c) Model dTsa via the IDENTICAL inversion (not bias/5.15 approximation):
    TEB from LWU_base, CLMU5 from archived dTsa (already inversion-based),
    SUEWS-OHM facet values (6 sites) -> slopes/r vs obs.
"""
import sys, csv, json, glob
from pathlib import Path
import numpy as np
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.training.corpus_loader import load_all_sites
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[3]
SIG = 5.67e-8

def sitedata_albedo(site):
    p = ROOT / f'data/urban-plumber/FullCollection/{site}/{site}_sitedata_v1.csv'
    for row in csv.DictReader(p.open(encoding='utf-8')):
        if row['parameter'] == 'average_albedo_at_midday':
            return float(row['value'])
    return np.nan

# ---- load per-site nocturnal series ------------------------------------------
S = {}
for rec in load_all_sites():
    ds = xr.open_dataset(ROOT / f'data/urban-plumber/corpus/{rec.site}.nc')
    if 'obs_LWup' not in ds:
        ds.close(); continue
    n = rec.n_steps
    olw = ds['obs_LWup'].values; ta = ds['forcing_Tair'].values; ld = ds['forcing_LWdown'].values
    ds.close()
    night = np.asarray(rec.night_mask, bool) & (~np.asarray(rec.pre_spinup_flag, bool))
    m = night & np.isfinite(olw) & np.isfinite(ta) & np.isfinite(ld)
    if m.sum() < 100: continue
    # TEB model LWup (baseline) aligned: row i -> step i+1
    p = ROOT / f'external/teb_runs/{rec.site}/output/LWU_base.txt'
    teb = np.loadtxt(p) if p.exists() else None
    S[rec.site] = dict(olw=olw, ta=ta, ld=ld, m=m, teb=teb, nsteps=n,
                       alb=sitedata_albedo(rec.site))

sites = sorted(S)
alb = np.array([S[s]['alb'] for s in sites])

def dtsa_obs(site, eps):
    d = S[site]
    ts = ((d['olw'][d['m']] - (1-eps)*d['ld'][d['m']]) / (eps*SIG))**0.25
    return float((ts - d['ta'][d['m']]).mean())

print('=== KG1(a) EPS SWEEP on OBS dTsa (19 sites) ===')
print(f"{'eps':>5s} {'span_min':>8s} {'span_max':>8s} {'r_albedo':>9s} {'p':>8s}")
base = None
for eps in [0.90, 0.93, 0.95, 0.97, 0.99]:
    v = np.array([dtsa_obs(s, eps) for s in sites])
    r, p = pearsonr(alb, v)
    print(f"{eps:5.2f} {v.min():+8.2f} {v.max():+8.2f} {r:+9.3f} {p:8.4f}")
    if eps == 0.95: base = v

print('\n=== KG1(b) ADVERSARIAL per-site eps (0.90..0.98 assigned by albedo rank) ===')
order = np.argsort(np.argsort(alb))  # rank 0..n-1
for direction, lab in [(order, 'eps INCREASES with albedo'), (order[::-1] if False else (len(sites)-1-order), 'eps DECREASES with albedo')]:
    eps_i = 0.90 + 0.08 * direction/ (len(sites)-1)
    v = np.array([dtsa_obs(s, e) for s, e in zip(sites, eps_i)])
    r, p = pearsonr(alb, v)
    print(f"  {lab:32s}: span {v.min():+.2f}..{v.max():+.2f}  r_albedo {r:+.3f} (p={p:.4f})")

print('\n=== KG1(c) MODEL dTsa via IDENTICAL inversion (eps=0.95) ===')
eps = 0.95
# TEB proper inversion
teb_v, obs_v, ok_sites = [], [], []
for s in sites:
    d = S[s]
    if d['teb'] is None: continue
    nmod = min(d['teb'].size, d['nsteps']-1)
    sl = slice(1, nmod+1)
    mm = d['m'][sl] & np.isfinite(d['teb'][:nmod])
    ts_mod = ((d['teb'][:nmod][mm] - (1-eps)*d['ld'][sl][mm]) / (eps*SIG))**0.25
    teb_v.append(float((ts_mod - d['ta'][sl][mm]).mean()))
    # obs on same mask for exact pairing
    ts_obs = ((d['olw'][sl][mm] - (1-eps)*d['ld'][sl][mm]) / (eps*SIG))**0.25
    obs_v.append(float((ts_obs - d['ta'][sl][mm]).mean()))
    ok_sites.append(s)
teb_v, obs_v = np.array(teb_v), np.array(obs_v)
sl_t = np.polyfit(obs_v, teb_v, 1)[0]; r_t, _ = pearsonr(obs_v, teb_v)
print(f"TEB   (n={len(ok_sites)}): slope={sl_t:+.3f} r={r_t:+.2f}  model span {teb_v.min():+.2f}..{teb_v.max():+.2f} vs obs {obs_v.min():+.2f}..{obs_v.max():+.2f}")

# CLMU5 from archived per_site (dTsa_model computed by inversion in-run)
g2 = json.load(open(ROOT/'external/clmu_g1/g2_results.json', encoding='utf-8'))['per_site']
c_sites = [s for s in sites if s in g2]
c_v = np.array([g2[s]['dTsa_model'] for s in c_sites])
o_v = np.array([base[sites.index(s)] for s in c_sites])
sl_c = np.polyfit(o_v, c_v, 1)[0]; r_c, _ = pearsonr(o_v, c_v)
print(f"CLMU5 (n={len(c_sites)}): slope={sl_c:+.3f} r={r_c:+.2f}  model span {c_v.min():+.2f}..{c_v.max():+.2f}")

# SUEWS-OHM 6 sites facet dTsa
su = {}
for f in glob.glob(str(ROOT/'external/suews_g1/*/result.json')):
    d = json.load(open(f, encoding='utf-8'))
    su[d.get('site', Path(f).parent.name)] = d.get('noct_Ts_facet_minus_Tair')
su_sites = [s for s in sites if s in su and su[s] is not None]
if su_sites:
    s_v = np.array([su[s] for s in su_sites])
    o_v2 = np.array([base[sites.index(s)] for s in su_sites])
    sl_s = np.polyfit(o_v2, s_v, 1)[0]; r_s, _ = pearsonr(o_v2, s_v)
    print(f"SUEWS-OHM (n={len(su_sites)}): slope={sl_s:+.3f} r={r_s:+.2f}  model span {s_v.min():+.2f}..{s_v.max():+.2f}")
else:
    print('SUEWS: per-site result.json noct_Ts_facet_minus_Tair NOT FOUND — check key names')
    ex = glob.glob(str(ROOT/'external/suews_g1/*/result.json'))[:1]
    if ex: print('available keys:', list(json.load(open(ex[0],encoding='utf-8')).keys()))
