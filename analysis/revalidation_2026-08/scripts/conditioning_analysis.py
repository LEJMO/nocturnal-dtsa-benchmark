"""EXPLORATORY (pre-declared in PREREGISTRATION_G1.md, "Exploratory analyses"):
within-site conditioning of the nightly community-TEB upward-longwave (LWup) bias.

Goal: discriminate mechanism families for the organized per-site nocturnal LWup bias.
  - climate-axis  : radiative-cooling / storage misrepresentation -> the bias should
                    CONCENTRATE on clear (low eps_sky) and/or calm nights, when the
                    surface energy budget is dominated by radiative cooling.
  - cover-axis    : static cover / material-property misrepresentation -> the bias should
                    be roughly INVARIANT to sky clearness and wind.

Masking/alignment copied VERBATIM from scripts/flux_direction_check.py:
  model row i corresponds to forcing step i+1, so slice(1, n+1); n = min(M.size, n_steps-1);
  night = night_mask & ~pre_spinup_flag. Primary metric: mean nocturnal bias = LWU_model - obs_LWup (W/m2).

Outputs:
  - per-site table -> evidence/conditioning_table.csv
  - console: per-site contrasts, pooled group contrasts, block-bootstrap 95% CIs, honest synthesis.
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.training.corpus_loader import load_all_sites  # noqa: E402

RUNS = ROOT / 'external' / 'teb_runs'
CORP = ROOT / 'data' / 'urban-plumber' / 'corpus'
OUT_CSV = ROOT / 'analysis' / 'revalidation_2026-08' / 'evidence' / 'conditioning_table.csv'

SIGMA = 5.67e-8
EPS_SURF = 0.95
TREF = 285.0
K_PER_WM2 = 1.0 / (4 * EPS_SURF * SIGMA * TREF ** 3)  # ~1/5.15

N_BOOT = 2000
RNG = np.random.default_rng(20260828)

WARM_EXTREME = ['US-Minneapolis1', 'US-Minneapolis2', 'KR-Ochang', 'US-WestPhoenix']
DENSE_EURO = ['FR-Capitole', 'UK-KingsCollege', 'NL-Amsterdam']
LIPOWA = ['PL-Lipowa']

SEASON_OF_MONTH = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                   6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}


def load_lwu(site):
    p = RUNS / site / 'output' / 'LWU_base.txt'
    return np.loadtxt(p) if p.exists() else None


def segment_blocks(night_bool):
    """Contiguous runs of night_bool==True -> list of index arrays (one per night)."""
    idx = np.where(night_bool)[0]
    if idx.size == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)


def block_bootstrap_diff(blocks_clear, blocks_cloudy, blocks_bias, is_clear_blk, is_cloudy_blk):
    """Block bootstrap of (mean bias in clearest quartile) - (mean in cloudiest quartile).

    Resamples whole night-blocks with replacement; quartile membership is FIXED
    (thresholds computed once on the full sample). 95% percentile CI.
    """
    n_blk = len(blocks_bias)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.integers(0, n_blk, n_blk)
        bias_c = np.concatenate([blocks_bias[i][is_clear_blk[i]] for i in pick])
        bias_d = np.concatenate([blocks_bias[i][is_cloudy_blk[i]] for i in pick])
        if bias_c.size == 0 or bias_d.size == 0:
            diffs[b] = np.nan
        else:
            diffs[b] = bias_c.mean() - bias_d.mean()
    diffs = diffs[np.isfinite(diffs)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def analyse_site(rec):
    site = rec.site
    M = load_lwu(site)
    if M is None:
        return None
    n = min(M.size, rec.n_steps - 1)
    M = M[:n]
    sl = slice(1, n + 1)

    night = np.asarray(rec.night_mask, bool)[sl] & (~np.asarray(rec.pre_spinup_flag, bool)[sl])

    ds = xr.open_dataset(CORP / f'{site}.nc')
    obs = ds['obs_LWup'].values[sl]
    ds.close()

    lwdown = rec.forcing['LWdown'][sl]
    tair = rec.forcing['Tair'][sl]
    wind = np.hypot(rec.forcing['Wind_N'][sl], rec.forcing['Wind_E'][sl])
    qair = rec.forcing['Qair'][sl]
    times = pd.to_datetime(rec.time_axis[sl])
    months = times.month.to_numpy()
    seasons = np.array([SEASON_OF_MONTH[m] for m in months])

    eps_sky = lwdown / (SIGMA * tair ** 4)  # clear (dry/cloudless) night -> LOW eps_sky

    bias = M - obs
    valid = (night & np.isfinite(M) & np.isfinite(obs) & np.isfinite(eps_sky)
             & np.isfinite(wind) & np.isfinite(qair))
    if valid.sum() < 100:
        return None

    v = valid
    eps_v = eps_sky[v]
    wind_v = wind[v]
    bias_v = bias[v]

    # Fixed quartile thresholds over the valid nocturnal sample
    eps_q = np.percentile(eps_v, [25, 50, 75])
    wnd_q = np.percentile(wind_v, [25, 50, 75])

    clearest = eps_sky <= eps_q[0]
    cloudiest = eps_sky >= eps_q[2]
    calmest = wind <= wnd_q[0]
    windiest = wind >= wnd_q[2]
    clear_calm = (eps_sky <= eps_q[1]) & (wind <= wnd_q[1])       # both below median
    cloudy_windy = (eps_sky >= eps_q[1]) & (wind >= wnd_q[1])     # both above median

    def mb(sel):
        m = v & sel
        return (float(bias[m].mean()), int(m.sum())) if m.sum() > 0 else (np.nan, 0)

    b_clear, n_clear = mb(clearest)
    b_cloudy, n_cloudy = mb(cloudiest)
    b_calm, n_calm = mb(calmest)
    b_windy, n_windy = mb(windiest)
    b_cc, n_cc = mb(clear_calm)
    b_cw, n_cw = mb(cloudy_windy)
    b_all = float(bias_v.mean())

    # Block bootstrap of clear-minus-cloudy difference over NIGHTS
    blocks = segment_blocks(night)
    blk_bias, blk_isclear, blk_iscloudy = [], [], []
    for blk in blocks:
        keep = v[blk]
        bi = blk[keep]
        if bi.size == 0:
            continue
        blk_bias.append(bias[bi])
        blk_isclear.append(clearest[bi])
        blk_iscloudy.append(cloudiest[bi])
    ci_lo, ci_hi = (np.nan, np.nan)
    if len(blk_bias) >= 5:
        ci_lo, ci_hi = block_bootstrap_diff(None, None, blk_bias, blk_isclear, blk_iscloudy)

    seasonal = {}
    for sname in ['DJF', 'MAM', 'JJA', 'SON']:
        m = v & (seasons == sname)
        seasonal[sname] = float(bias[m].mean()) if m.sum() >= 30 else np.nan
        seasonal[sname + '_n'] = int(m.sum())

    return {
        'site': site,
        'koppen': rec.koppen_zone,
        'lat': float(rec.latitude),
        'n_night': int(v.sum()),
        'n_blocks': len(blk_bias),
        'bias_all': b_all,
        'bias_clear': b_clear, 'bias_cloudy': b_cloudy,
        'clear_minus_cloudy': b_clear - b_cloudy,
        'ci95_lo': ci_lo, 'ci95_hi': ci_hi,
        'bias_calm': b_calm, 'bias_windy': b_windy,
        'calm_minus_windy': b_calm - b_windy,
        'bias_clearcalm': b_cc, 'bias_cloudywindy': b_cw,
        'clearcalm_minus_cloudywindy': b_cc - b_cw,
        'n_clear': n_clear, 'n_cloudy': n_cloudy, 'n_calm': n_calm, 'n_windy': n_windy,
        **{f'DJF': seasonal['DJF'], 'MAM': seasonal['MAM'], 'JJA': seasonal['JJA'], 'SON': seasonal['SON']},
        'DJF_n': seasonal['DJF_n'], 'JJA_n': seasonal['JJA_n'],
    }


def sig_flag(lo, hi):
    if not np.isfinite(lo) or not np.isfinite(hi):
        return '  '
    return '**' if (lo > 0 or hi < 0) else '  '  # CI excludes 0


def main():
    rows = []
    for rec in load_all_sites():
        r = analyse_site(rec)
        if r is not None:
            rows.append(r)
    df = pd.DataFrame(rows).sort_values('bias_all').reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option('display.width', 200)
    print('===== PER-SITE NOCTURNAL LWup BIAS CONDITIONING (community TEB, uncorrected) =====')
    print(f'{"site":17s} {"kop":4s} {"biasAll":>8s} {"clear":>7s} {"cloudy":>7s} {"cl-cl":>7s} '
          f'{"[95% CI]":>16s} {"calm":>7s} {"windy":>7s} {"ca-wi":>7s} {"CC":>7s} {"CW":>7s}')
    for _, r in df.iterrows():
        flag = sig_flag(r['ci95_lo'], r['ci95_hi'])
        ci = f"[{r['ci95_lo']:+.1f},{r['ci95_hi']:+.1f}]"
        print(f"{r['site']:17s} {r['koppen']:4s} {r['bias_all']:+8.2f} {r['bias_clear']:+7.2f} "
              f"{r['bias_cloudy']:+7.2f} {r['clear_minus_cloudy']:+7.2f}{flag} {ci:>16s} "
              f"{r['bias_calm']:+7.2f} {r['bias_windy']:+7.2f} {r['calm_minus_windy']:+7.2f} "
              f"{r['bias_clearcalm']:+7.2f} {r['bias_cloudywindy']:+7.2f}")

    print('\n  ** = clear-minus-cloudy 95% block-bootstrap CI excludes 0 (bias is condition-DEPENDENT)')
    print('  K-equiv: divide W/m2 by 5.15.')

    print('\n===== SEASONAL MEAN BIAS (W/m2) =====')
    print(f'{"site":17s} {"kop":4s} {"DJF":>8s} {"MAM":>8s} {"JJA":>8s} {"SON":>8s}  {"DJF_n":>6s} {"JJA_n":>6s}')
    for _, r in df.iterrows():
        def f(x):
            return f'{x:+8.2f}' if np.isfinite(x) else f'{"--":>8s}'
        print(f"{r['site']:17s} {r['koppen']:4s} {f(r['DJF'])} {f(r['MAM'])} {f(r['JJA'])} {f(r['SON'])}  "
              f"{int(r['DJF_n']):6d} {int(r['JJA_n']):6d}")

    print('\n===== POOLED GROUP CONTRASTS (mean over sites of the per-site contrast) =====')
    dfi = df.set_index('site')
    groups = [('WARM-EXTREME', WARM_EXTREME), ('DENSE-EURO', DENSE_EURO), ('PL-Lipowa', LIPOWA)]
    print(f'{"group":14s} {"n":>2s} {"biasAll":>8s} {"clear-cloudy":>13s} {"calm-windy":>11s} {"CC-CW":>8s}')
    for gname, sites in groups:
        sub = dfi.loc[[s for s in sites if s in dfi.index]]
        print(f"{gname:14s} {len(sub):2d} {sub['bias_all'].mean():+8.2f} "
              f"{sub['clear_minus_cloudy'].mean():+13.2f} {sub['calm_minus_windy'].mean():+11.2f} "
              f"{sub['clearcalm_minus_cloudywindy'].mean():+8.2f}")

    df.to_csv(OUT_CSV, index=False)
    print(f'\n[saved] {OUT_CSV}')
    return df


if __name__ == '__main__':
    main()
