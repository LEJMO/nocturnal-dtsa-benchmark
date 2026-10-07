# -*- coding: utf-8 -*-
"""ADVERSARIAL RE-VERIFICATION of review5_permutation_rebuild.py.

Independent re-implementation, own data load, own seeds, own vectorisation.
Reads results/paper_stats_v1.json READ-ONLY for a per-site cross-check and
writes NOTHING; every number goes to stdout only.

Schemes:
  CB     restricted equal-size city-block permutation (record level, PRIMARY)
  CB_ws  CB plus within-block record shuffling (sensitivity)
  CM     city-mean permutation (= paper_stats.py:290)
  SM     city-mean albedo smeared back to records (= review2_additions.py:60)
  FL-CB  textbook Freedman-Lane (permute reduced-model residuals in CB blocks)
  FL-SM  review4_additions.py scheme (residualise both, smear albedo residuals)

Vectorisation notes (why this is not a copy of the script under test):
  * albedo |r| statistic reduces to |alb[perm] . (dt - mean dt)| because a
    permutation leaves the albedo variance invariant, so no per-draw pearsonr;
  * Freedman-Lane t is evaluated in closed form from a = ((X'X)^-1 X')[2] and
    M = I - X (X'X)^-1 X', using M fit0 = 0 and a . fit0 = 0.

Usage:  review5_permutation_rebuild_verify.py [N_ALBEDO] [N_FL] [SEED]
Default N_ALBEDO=1e7, N_FL=4e6, SEED=424242.
Constants: SIG = 5.67e-8 (project convention, NOT 5.670374419e-8), EPS = 0.95.
No CLMU or TEB quantity is read anywhere.
"""
import sys, csv, json, io, math
from pathlib import Path
from collections import Counter
import numpy as np
import xarray as xr
from scipy.stats import pearsonr, t as tdist

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[3]
SIG, EPS = 5.67e-8, 0.95
CLUSTER = {'US-Minneapolis1': 'Minneapolis', 'US-Minneapolis2': 'Minneapolis',
           'PL-Lipowa': 'Lodz', 'PL-Narutowicza': 'Lodz',
           'FI-Kumpula': 'Helsinki', 'FI-Torni': 'Helsinki'}
N_ALB = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
N_FL = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 424242

sys.path.insert(0, str(ROOT))
from src.training.corpus_loader import load_all_sites

def sitedata(site):
    v = {}
    for row in csv.DictReader((ROOT / f'data/urban-plumber/FullCollection/{site}/{site}_sitedata_v1.csv').open(encoding='utf-8')):
        try: v[row['parameter']] = float(row['value'])
        except Exception: pass
    return v

S = {}
for site, rec in {r.site: r for r in load_all_sites()}.items():
    ds = xr.open_dataset(ROOT / f'data/urban-plumber/corpus/{site}.nc')
    if 'obs_LWup' not in ds:
        ds.close(); continue
    olw = np.ma.masked_values(np.ma.masked_invalid(np.asarray(ds['obs_LWup'].values, float)), -999.0)
    ta = np.ma.masked_values(np.ma.masked_invalid(np.asarray(ds['forcing_Tair'].values, float)), -999.0)
    ld = np.ma.masked_values(np.ma.masked_invalid(np.asarray(ds['forcing_LWdown'].values, float)), -999.0)
    ds.close()
    night = np.asarray(rec.night_mask, bool) & (~np.asarray(rec.pre_spinup_flag, bool))
    m = night & ~np.ma.getmaskarray(olw) & ~np.ma.getmaskarray(ta) & ~np.ma.getmaskarray(ld)
    if m.sum() < 100: continue
    o, T, L = np.asarray(olw)[m], np.asarray(ta)[m], np.asarray(ld)[m]
    ts = ((o - (1 - EPS) * L) / (EPS * SIG)) ** 0.25
    S[site] = dict(cluster=CLUSTER.get(site, site), dtsa=float((ts - T).mean()),
                   albedo=sitedata(site).get('average_albedo_at_midday', np.nan),
                   eps_sky=float((L / (SIG * T ** 4)).mean()))

sites = sorted(S)
VAR = {'all19': sites,
       'excl_mpls17': [s for s in sites if not s.startswith('US-Minneapolis')],
       'core16': [s for s in sites if not s.startswith('US-Minneapolis') and s != 'PL-Lipowa']}

led = json.load(open(ROOT / 'results/paper_stats_v1.json', encoding='utf-8'))
ps = led['per_site']
print('ledger max|d dtsa|=%.3e  max|d albedo|=%.3e' % (
    max(abs(S[s]['dtsa'] - ps[s]['dtsa_std']) for s in sites),
    max(abs(S[s]['albedo'] - ps[s]['albedo']) for s in sites)))
print('cluster mismatches:', [s for s in sites if S[s]['cluster'] != ps[s]['cluster']])
r3 = led.get('review3_additions', {})
print('eps_sky stored keys:', [k for k in r3 if 'eps' in k.lower()][:5])

def arrs(ss):
    return (np.array([S[s]['albedo'] for s in ss]), np.array([S[s]['dtsa'] for s in ss]),
            np.array([S[s]['eps_sky'] for s in ss]))

def blocks(ss):
    by = {}
    for i, s in enumerate(ss): by.setdefault(S[s]['cluster'], []).append(i)
    return by

def cb_chunks(ss, rng, nperm, within, chunk=250_000):
    by = blocks(ss); names = list(by)
    cls = {}
    for c in names: cls.setdefault(len(by[c]), []).append(c)
    n = len(ss)
    done = 0
    while done < nperm:
        k = min(chunk, nperm - done)
        IX = np.empty((k, n), np.int64)
        for sz, members in cls.items():
            M = len(members)
            P = np.argsort(rng.random((k, M)), axis=1)
            for a in range(M):
                dest = by[members[a]]
                srcs = np.array([by[c] for c in members])           # (M, sz)
                chosen = srcs[P[:, a]]                              # (k, sz)
                if within and sz > 1:
                    o = np.argsort(rng.random((k, sz)), axis=1)
                    chosen = np.take_along_axis(chosen, o, axis=1)
                IX[:, dest] = chosen
        yield IX
        done += k

def cb_albedo(ss, nperm, seed, within=False):
    alb, dt, _ = arrs(ss)
    dtc = dt - dt.mean()
    r_obs = pearsonr(alb, dt)[0]
    denom = math.sqrt(((alb - alb.mean()) ** 2).sum() * (dtc ** 2).sum())
    s_obs = abs(r_obs) * denom
    rng = np.random.default_rng(seed); b = 0
    for IX in cb_chunks(ss, rng, nperm, within):
        b += int((np.abs(alb[IX] @ dtc) >= s_obs - 1e-12 * denom).sum())
    p = (b + 1) / (nperm + 1)
    cc = Counter(len(v) for v in blocks(ss).values())
    arr = 1
    for _, c in cc.items(): arr *= math.factorial(c)
    return dict(r=float(r_obs), p=p, b=b, se=math.sqrt(p * (1 - p) / nperm),
                blocksizes=dict(cc), arrangements=float(arr))

def cm_albedo(ss, nperm, seed):
    by = blocks(ss); cl = list(by)
    alb, dt, _ = arrs(ss)
    cx = np.array([alb[by[c]].mean() for c in cl]); cy = np.array([dt[by[c]].mean() for c in cl])
    r_obs = pearsonr(cx, cy)[0]
    cxc, cyc = cx - cx.mean(), cy - cy.mean()
    s_obs = abs((cxc * cyc).sum())
    rng = np.random.default_rng(seed); b = 0; done = 0
    while done < nperm:
        k = min(250_000, nperm - done)
        P = np.argsort(rng.random((k, len(cl))), axis=1)
        b += int((np.abs(cxc[P] @ cyc) >= s_obs - 1e-12).sum()); done += k
    p = (b + 1) / (nperm + 1)
    return dict(m=len(cl), r_city=float(r_obs), p=p, b=b, se=math.sqrt(p * (1 - p) / nperm),
                arrangements=float(math.factorial(len(cl))))

def sm_albedo(ss, nperm, seed):
    """review2_additions.py:60 -- city-mean albedo smeared back to records."""
    by = blocks(ss); cl = list(by)
    alb, dt, _ = arrs(ss)
    means = np.array([alb[by[c]].mean() for c in cl])
    rep = np.zeros(len(ss), int)
    for j, c in enumerate(cl): rep[by[c]] = j
    xv = means[rep]
    r_obs = pearsonr(xv, dt)[0]
    dtc = dt - dt.mean()
    rng = np.random.default_rng(seed); b = 0; done = 0
    while done < nperm:
        k = min(250_000, nperm - done)
        P = np.argsort(rng.random((k, len(cl))), axis=1)
        XP = means[P][:, rep]                                  # (k, n)
        num = np.abs(XP @ dtc)
        sx = np.sqrt(((XP - XP.mean(1, keepdims=True)) ** 2).sum(1))
        rr = num / (sx * math.sqrt((dtc ** 2).sum()))
        b += int((rr >= abs(r_obs)).sum()); done += k
    return dict(r_smeared=float(r_obs), p=(b + 1) / (nperm + 1), b=b)

def fl(ss, nperm, seed, within=False, scheme='CB'):
    alb, dt, es = arrs(ss); n = len(ss)
    X0 = np.column_stack([np.ones(n), es])
    b0 = np.linalg.lstsq(X0, dt, rcond=None)[0]
    fit0 = X0 @ b0; r0 = dt - fit0
    X1 = np.column_stack([np.ones(n), es, alb])
    XtXi = np.linalg.inv(X1.T @ X1)
    A = XtXi @ X1.T; a = A[2]
    Mmat = np.eye(n) - X1 @ A
    C22 = XtXi[2, 2]
    def tstat(Y):
        num = Y @ a
        rss = ((Y @ Mmat) * Y).sum(-1)
        return num / np.sqrt(rss / (n - 3) * C22)
    t_obs = float(tstat(dt))
    ra = alb - X0 @ np.linalg.lstsq(X0, alb, rcond=None)[0]
    pr = float(pearsonr(ra, r0)[0])
    rng = np.random.default_rng(seed); b = 0
    for IX in cb_chunks(ss, rng, nperm, within):
        b += int((np.abs(tstat(fit0 + r0[IX])) >= abs(t_obs) - 1e-12).sum())
    p = (b + 1) / (nperm + 1)
    return dict(partial_r=pr, t=t_obs, p=p, b=b, se=math.sqrt(p * (1 - p) / nperm),
                p_param=float(2 * (1 - tdist.cdf(abs(t_obs), n - 3))))

def fl_sm(ss, nperm, seed):
    """review4_additions.py scheme: residualise both, smear albedo-residual city means."""
    by = blocks(ss); cl = list(by)
    alb, dt, es = arrs(ss); n = len(ss)
    Z = np.column_stack([es, np.ones(n)])
    ra = alb - Z @ np.linalg.lstsq(Z, alb, rcond=None)[0]
    rd = dt - Z @ np.linalg.lstsq(Z, dt, rcond=None)[0]
    r_obs = pearsonr(ra, rd)[0]
    means = np.array([ra[by[c]].mean() for c in cl])
    rep = np.zeros(n, int)
    for j, c in enumerate(cl): rep[by[c]] = j
    rdc = rd - rd.mean()
    rng = np.random.default_rng(seed); b = 0; done = 0
    while done < nperm:
        k = min(250_000, nperm - done)
        P = np.argsort(rng.random((k, len(cl))), axis=1)
        XP = means[P][:, rep]
        num = np.abs(XP @ rdc)
        sx = np.sqrt(((XP - XP.mean(1, keepdims=True)) ** 2).sum(1))
        b += int((num / (sx * math.sqrt((rdc ** 2).sum())) >= abs(r_obs)).sum()); done += k
    return dict(partial_r=float(r_obs), p=(b + 1) / (nperm + 1), b=b)

print(f'\n=== CB restricted city-block, N={N_ALB}, seed={SEED} ===')
for i, (v, ss) in enumerate(VAR.items()):
    r = cb_albedo(ss, N_ALB, SEED + i)
    print(f'{v:12s} r={r["r"]:+.5f} p={r["p"]:.4e} b={r["b"]} SE={r["se"]:.2e} '
          f'sizes={r["blocksizes"]} arr={r["arrangements"]:.4e}')
print('=== CB_ws ===')
for i, (v, ss) in enumerate(VAR.items()):
    r = cb_albedo(ss, N_ALB, SEED + 10 + i, within=True)
    print(f'{v:12s} p={r["p"]:.4e} b={r["b"]} SE={r["se"]:.2e}')
print('=== CM city-mean ===')
for i, (v, ss) in enumerate(VAR.items()):
    r = cm_albedo(ss, N_ALB, SEED + 20 + i)
    print(f'{v:12s} m={r["m"]} r_city={r["r_city"]:+.5f} p={r["p"]:.4e} b={r["b"]} SE={r["se"]:.2e} arr={r["arrangements"]:.4e}')
print('=== SM (review2 legacy) ===')
for i, (v, ss) in enumerate(VAR.items()):
    r = sm_albedo(ss, min(N_ALB, 2_000_000), SEED + 30 + i)
    print(f'{v:12s} r_smeared={r["r_smeared"]:+.5f} p={r["p"]:.4e} b={r["b"]}')
print(f'\n=== FL-CB textbook, N={N_FL} ===')
for i, (v, ss) in enumerate(VAR.items()):
    r = fl(ss, N_FL, SEED + 40 + i)
    print(f'{v:12s} pr={r["partial_r"]:+.5f} t={r["t"]:+.4f} p={r["p"]:.5f} b={r["b"]} SE={r["se"]:.1e} p_param={r["p_param"]:.5f}')
print('=== FL-CB_ws ===')
for i, (v, ss) in enumerate(VAR.items()):
    r = fl(ss, N_FL, SEED + 50 + i, within=True)
    print(f'{v:12s} p={r["p"]:.5f} SE={r["se"]:.1e}')
print('=== FL-SM (review4 legacy scheme) ===')
for i, (v, ss) in enumerate(VAR.items()):
    r = fl_sm(ss, min(N_FL, 2_000_000), SEED + 60 + i)
    print(f'{v:12s} pr={r["partial_r"]:+.5f} p={r["p"]:.5f} b={r["b"]}')
