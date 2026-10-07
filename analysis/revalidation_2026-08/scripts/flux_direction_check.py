"""Definitive data check (OUR community-TEB run; Lipson raw per-model data is
not publicly released): for upward longwave LWup (surface-temp proxy) and
sensible heat H, split DAY vs NIGHT and report bias direction, RMSE, and the
fraction of samples where the model OVER-predicts (model>obs). Answers: is it
consistently over/under, or mixed?
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.training.corpus_loader import load_all_sites

RUNS = Path(__file__).resolve().parents[3] / 'external' / 'teb_runs'
CORP = Path(__file__).resolve().parents[3] / 'data' / 'urban-plumber' / 'corpus'
sig, eps, Tref = 5.67e-8, 0.95, 285.0
K_per_Wm2 = 1.0 / (4 * eps * sig * Tref**3)


def load(site, name):
    p = RUNS / site / 'output' / name
    return np.loadtxt(p) if p.exists() else None


def summarize(model, obs, mask, label):
    m = np.isfinite(model) & np.isfinite(obs) & mask
    if m.sum() < 50:
        return None
    d = model[m] - obs[m]
    return {'label': label, 'n': int(m.sum()), 'bias': float(d.mean()),
            'rmse': float(np.sqrt((d**2).mean())),
            'frac_over': float((d > 0).mean())}


def run(var_model, var_obs, name, to_K=None):
    poolN = {'d': []}
    poolD = {'d': []}
    over_night_sites = 0
    site_rows = []
    for rec in load_all_sites():
        M = load(rec.site, f'{var_model}_base.txt')
        if M is None:
            M = load(rec.site, f'{var_model}.txt')
        if M is None:
            continue
        n = min(M.size, rec.n_steps - 1)
        M = M[:n]
        sl = slice(1, n + 1)
        night = (np.asarray(rec.night_mask, bool)[sl] & (~np.asarray(rec.pre_spinup_flag, bool)[sl]))
        day = (~np.asarray(rec.night_mask, bool)[sl]) & (~np.asarray(rec.pre_spinup_flag, bool)[sl])
        ds = xr.open_dataset(CORP / f'{rec.site}.nc')
        obs = ds[var_obs].values[sl] if var_obs in ds else np.full(n, np.nan)
        ds.close()
        sN = summarize(M, obs, night, 'night')
        sD = summarize(M, obs, day, 'day')
        if sN:
            site_rows.append((rec.site, sN, sD))
            mN = np.isfinite(M) & np.isfinite(obs) & night
            poolN['d'].extend((M[mN] - obs[mN]).tolist())
            over_night_sites += (sN['bias'] > 0)
        if sD:
            mD = np.isfinite(M) & np.isfinite(obs) & day
            poolD['d'].extend((M[mD] - obs[mD]).tolist())
    print(f'\n===== {name} (community TEB, uncorrected) =====')
    print(f"{'site':18s} {'NIGHT bias':>10s} {'over%':>6s} {'RMSE':>6s} | {'DAY bias':>9s} {'over%':>6s}")
    for s, sN, sD in sorted(site_rows, key=lambda x: x[1]['bias']):
        db = sD['bias'] if sD else float('nan')
        do = 100 * sD['frac_over'] if sD else float('nan')
        print(f"{s:18s} {sN['bias']:+10.2f} {100*sN['frac_over']:5.0f}% {sN['rmse']:6.1f} | {db:+9.2f} {do:5.0f}%")
    dN = np.array(poolN['d']); dD = np.array(poolD['d'])
    print(f"POOLED NIGHT: bias {dN.mean():+.2f} W/m2, over {100*(dN>0).mean():.0f}% of samples, RMSE {np.sqrt((dN**2).mean()):.1f}")
    print(f"POOLED DAY  : bias {dD.mean():+.2f} W/m2, over {100*(dD>0).mean():.0f}% of samples, RMSE {np.sqrt((dD**2).mean()):.1f}")
    print(f"night-bias-positive sites: {over_night_sites}/{len(site_rows)}")
    if to_K:
        print(f"NIGHT bias in K-equiv: {dN.mean()*to_K:+.2f} K ; DAY {dD.mean()*to_K:+.2f} K")


run('LWU', 'obs_LWup', 'UPWARD LONGWAVE LWup (surface-temp proxy)', to_K=K_per_Wm2)
run('H_TOWN', 'obs_Qh', 'SENSIBLE HEAT H')
