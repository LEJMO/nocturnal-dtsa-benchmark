"""EXPLORATORY (pre-declared, PREREGISTRATION_G1.md §"Exploratory analyses"):
full-corpus audit of the per-site nocturnal upward-longwave (LWup) bias SIGN
structure, with block-bootstrap CIs, snow (DJF) sensitivity, a QH companion
column (scope honesty), and the pre-registered + exploratory descriptor
correlation families with Benjamini-Hochberg FDR and influence checks.

Masking / alignment is copied VERBATIM from scripts/flux_direction_check.py
(the frozen convention): for a TEB output of length M, model row i == forcing
step i+1, so we slice the per-step masks/obs with slice(1, n+1) where
n = min(M.size, rec.n_steps - 1).

Primary metric: per-site mean nocturnal LWup bias = mean(model - obs_LWup), W/m2.
Secondary display: K-equivalent = bias / 5.15.

Honesty: failures reported as failures; no site substitution; no threshold
tuning; every number derived from the frozen files only.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

PROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROOT))
from src.training.corpus_loader import load_all_sites  # noqa: E402

RUNS = PROOT / 'external' / 'teb_runs'
CORP = PROOT / 'data' / 'urban-plumber' / 'corpus'
SITEDATA = PROOT / 'data' / 'urban-plumber' / 'FullCollection'
OUT_CSV = PROOT / 'analysis' / 'revalidation_2026-08' / 'evidence' / 'sign_audit_table.csv'

SIG, EPS, TREF = 5.67e-8, 0.95, 285.0
K_PER_WM2 = 1.0 / (4 * EPS * SIG * TREF ** 3)  # ~= 1/5.15

N_BOOT = 5000
SEED = 20260829
DJF = {12, 1, 2}

# Cold-climate sites for snow (DJF) sensitivity per pre-registration (Dfa/Dfb/Dwa/Dwb).
COLD_KOPPEN = {'Dfa', 'Dfb', 'Dwa', 'Dwb'}


def load_txt(site, name):
    p = RUNS / site / 'output' / name
    return np.loadtxt(p) if p.exists() else None


def read_sitedata(site):
    """parameter -> float value from FullCollection/<SITE>/<SITE>_sitedata_v1.csv."""
    p = SITEDATA / site / f'{site}_sitedata_v1.csv'
    d = {}
    with open(p, newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[1]:
                key = row[1].strip()
                try:
                    d[key] = float(row[2])
                except (ValueError, IndexError):
                    pass
    return d


def contiguous_blocks(idx):
    """Split a sorted 1-D array of selected sample positions into contiguous-night
    blocks (a gap of >1 between consecutive positions starts a new block)."""
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, breaks)


def block_bootstrap_ci(resid, blocks, n_boot=N_BOOT, seed=SEED):
    """95% CI of the mean residual via block bootstrap over contiguous nights.
    resid: per-sample residuals (model-obs) aligned to the block index space.
    blocks: list of arrays of positions (into the same space as resid indexing)."""
    if len(blocks) < 2:
        return (np.nan, np.nan)
    # Map global positions -> local index in resid vector.
    # resid is already the masked residual array; blocks index into masked space.
    bsum = np.array([resid[b].sum() for b in blocks])
    bcnt = np.array([resid[b].size for b in blocks], dtype=float)
    B = len(blocks)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for k in range(n_boot):
        j = rng.integers(0, B, B)
        means[k] = bsum[j].sum() / bcnt[j].sum()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def verdict(lo, hi):
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 'indeterminate'
    if hi < 0:
        return 'NEGATIVE'
    if lo > 0:
        return 'positive'
    return 'indeterminate'


def per_site_bias(site, n_steps, night_mask, pre_spinup, model_name, obs_name,
                  want_blocks=False, want_months=False):
    """Return dict with bias, n, (optional) block bootstrap CI + monthly split.
    Masking/alignment verbatim from flux_direction_check.py."""
    M = load_txt(site, model_name)
    if M is None:
        return None
    n = min(M.size, n_steps - 1)
    M = M[:n]
    sl = slice(1, n + 1)
    night = np.asarray(night_mask, bool)[sl] & (~np.asarray(pre_spinup, bool)[sl])
    ds = xr.open_dataset(CORP / f'{site}.nc')
    obs = ds[obs_name].values[sl] if obs_name in ds else np.full(n, np.nan)
    months = pd.to_datetime(ds['time'].values[sl]).month.to_numpy() if want_months else None
    ds.close()
    mask = np.isfinite(M) & np.isfinite(obs) & night
    if mask.sum() < 50:
        return None
    resid = M[mask] - obs[mask]
    out = {'bias': float(resid.mean()), 'n': int(mask.sum()),
           'frac_over': float((resid > 0).mean())}
    if want_months:
        m_sel = months[mask]
        keep = ~np.isin(m_sel, list(DJF))
        out['bias_noDJF'] = float(resid[keep].mean()) if keep.sum() >= 50 else np.nan
        out['n_noDJF'] = int(keep.sum())
    if want_blocks:
        pos = np.where(mask)[0]                     # positions in 0..n-1
        local = np.arange(pos.size)                 # index into resid
        # blocks are contiguous runs of pos; translate to local indices
        breaks = np.where(np.diff(pos) > 1)[0] + 1
        blocks = np.split(local, breaks)
        out['ci'] = block_bootstrap_ci(resid, blocks)
        out['n_nights'] = len(blocks)
    return out


def bh_fdr(pvals, q=0.1):
    """Benjamini-Hochberg: return boolean survive-flags aligned to pvals order."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    survive = np.zeros(m, bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        survive[order[:kmax + 1]] = True
    return survive


def corr_block(x, y, labels, mask=None):
    """Pearson + Spearman with p for each column of a design; returns list of dicts."""
    x = np.asarray(x, float)
    rows = []
    for j, lab in enumerate(labels):
        xj = x[:, j]
        m = np.isfinite(xj) & np.isfinite(y)
        if mask is not None:
            m = m & mask
        r_p, p_p = stats.pearsonr(xj[m], y[m])
        r_s, p_s = stats.spearmanr(xj[m], y[m])
        rows.append({'descriptor': lab, 'n': int(m.sum()),
                     'pearson_r': r_p, 'pearson_p': p_p,
                     'spearman_r': r_s, 'spearman_p': p_s})
    return rows


def main():
    records = {r.site: r for r in load_all_sites()}
    sites = sorted(records)

    rows = []
    for s in sites:
        rec = records[s]
        lw = per_site_bias(s, rec.n_steps, rec.night_mask, rec.pre_spinup_flag,
                           'LWU_base.txt', 'obs_LWup', want_blocks=True, want_months=True)
        if lw is None:
            print(f'{s}: LWup MISSING (reported, not substituted)')
            continue
        qh = per_site_bias(s, rec.n_steps, rec.night_mask, rec.pre_spinup_flag,
                           'H_TOWN_base.txt', 'obs_Qh')
        sd = read_sitedata(s)
        lo, hi = lw['ci']
        rows.append({
            'site': s,
            'koppen': rec.koppen_zone,
            'lat': rec.latitude,
            'n_night': lw['n'],
            'n_nights': lw['n_nights'],
            'lwup_bias_Wm2': round(lw['bias'], 3),
            'ci_lo': round(lo, 3),
            'ci_hi': round(hi, 3),
            'bias_K': round(lw['bias'] * K_PER_WM2, 4),
            'sign_verdict': verdict(lo, hi),
            'bias_noDJF_Wm2': (round(lw['bias_noDJF'], 3)
                               if np.isfinite(lw['bias_noDJF']) else np.nan),
            'n_noDJF': lw['n_noDJF'],
            'is_cold': rec.koppen_zone in COLD_KOPPEN,
            'qh_bias_Wm2': round(qh['bias'], 3) if qh else np.nan,
            'albedo': sd['average_albedo_at_midday'],
            'pervious': round(1.0 - sd['impervious_area_fraction'], 4),
            'HW': sd['canyon_height_width_ratio'],
            'bldg_h': sd['building_mean_height'],
            'wall_plan': sd['wall_to_plan_area_ratio'],
            'tree': sd['tree_area_fraction'],
            'grass': sd['grass_area_fraction'],
            'abs_lat': abs(rec.latitude),
            'continental': 1 if rec.koppen_zone.startswith('D') else 0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('lwup_bias_Wm2').reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    # ---- Table print ----
    print('\n===== PER-SITE NOCTURNAL LWup BIAS SIGN AUDIT (block-bootstrap 95% CI) =====')
    hdr = (f"{'site':17s} {'kop':4s} {'bias':>8s} {'95% CI':>17s} {'verdict':>12s} "
           f"{'noDJF':>8s} {'QHbias':>8s} {'nights':>6s}")
    print(hdr)
    for _, r in df.iterrows():
        ci = f"[{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}]"
        nod = f"{r['bias_noDJF_Wm2']:+.1f}" if pd.notna(r['bias_noDJF_Wm2']) else '   -'
        cold = '*' if r['is_cold'] else ' '
        print(f"{r['site']:17s} {r['koppen']:4s} {r['lwup_bias_Wm2']:+8.2f} {ci:>17s} "
              f"{r['sign_verdict']:>12s} {nod:>7s}{cold} {r['qh_bias_Wm2']:+8.2f} {r['n_nights']:6d}")

    neg = df[df['sign_verdict'] == 'NEGATIVE']['site'].tolist()
    pos = df[df['sign_verdict'] == 'positive']['site'].tolist()
    ind = df[df['sign_verdict'] == 'indeterminate']['site'].tolist()
    print(f"\nrobust NEGATIVE (CI<0): {neg}")
    print(f"robust positive (CI>0): {pos}")
    print(f"indeterminate         : {ind}")

    # QH sign structure (scope honesty)
    qpos = int((df['qh_bias_Wm2'] > 0).sum())
    print(f"\nQH nocturnal bias: {qpos}/{len(df)} sites positive "
          f"(range {df['qh_bias_Wm2'].min():+.1f} .. {df['qh_bias_Wm2'].max():+.1f} W/m2) "
          f"-- LWup axis organization is NOT mirrored by QH if QH is mixed-sign.")

    # ---- Snow sensitivity for cold sites ----
    print('\n===== SNOW (DJF) SENSITIVITY -- cold sites (Dfa/Dfb/Dwa/Dwb) =====')
    cold = df[df['is_cold']].sort_values('lwup_bias_Wm2')
    print(f"{'site':17s} {'kop':4s} {'bias_all':>9s} {'bias_noDJF':>11s} {'delta':>7s}")
    for _, r in cold.iterrows():
        if pd.notna(r['bias_noDJF_Wm2']):
            dlt = r['bias_noDJF_Wm2'] - r['lwup_bias_Wm2']
            print(f"{r['site']:17s} {r['koppen']:4s} {r['lwup_bias_Wm2']:+9.2f} "
                  f"{r['bias_noDJF_Wm2']:+11.2f} {dlt:+7.2f}")

    y = df['lwup_bias_Wm2'].values.astype(float)

    # ---- PRIMARY descriptor family (pre-registered; no FDR) ----
    print('\n===== PRIMARY descriptor family (pre-registered: albedo, pervious) =====')
    prim_labels = ['average_albedo_at_midday', 'pervious_fraction']
    Xp = np.column_stack([df['albedo'].values, df['pervious'].values]).astype(float)
    prim = corr_block(Xp, y, prim_labels)
    for r in prim:
        print(f"  {r['descriptor']:26s} Pearson r={r['pearson_r']:+.3f} (p={r['pearson_p']:.4f})  "
              f"Spearman rho={r['spearman_r']:+.3f} (p={r['spearman_p']:.4f})  n={r['n']}")

    # ---- EXPLORATORY family with BH-FDR q=0.1 ----
    print('\n===== EXPLORATORY descriptor family (BH-FDR q=0.1) =====')
    exp_labels = ['HW_ratio', 'building_height', 'wall_to_plan', 'tree_frac',
                  'grass_frac', 'continental_flag', 'abs_lat']
    Xe = np.column_stack([df['HW'].values, df['bldg_h'].values, df['wall_plan'].values,
                          df['tree'].values, df['grass'].values,
                          df['continental'].values, df['abs_lat'].values]).astype(float)
    expl = corr_block(Xe, y, exp_labels)
    p_pear = [r['pearson_p'] for r in expl]
    p_spear = [r['spearman_p'] for r in expl]
    surv_p = bh_fdr(p_pear, 0.1)
    surv_s = bh_fdr(p_spear, 0.1)
    print(f"  {'descriptor':18s} {'PearsonR':>9s} {'p':>8s} {'FDR':>4s} | "
          f"{'SpearR':>8s} {'p':>8s} {'FDR':>4s}")
    for i, r in enumerate(expl):
        print(f"  {r['descriptor']:18s} {r['pearson_r']:+9.3f} {r['pearson_p']:8.4f} "
              f"{'Y' if surv_p[i] else '.':>4s} | {r['spearman_r']:+8.3f} "
              f"{r['spearman_p']:8.4f} {'Y' if surv_s[i] else '.':>4s}")
    surv_names_p = [exp_labels[i] for i in range(len(exp_labels)) if surv_p[i]]
    surv_names_s = [exp_labels[i] for i in range(len(exp_labels)) if surv_s[i]]
    print(f"  survive FDR (Pearson):  {surv_names_p}")
    print(f"  survive FDR (Spearman): {surv_names_s}")

    # ---- Minneapolis shared-Tair + Lipowa influence checks on albedo corr ----
    print('\n===== ALBEDO correlation robustness (shared-Tair + influence) =====')
    alb = df['albedo'].values.astype(float)
    full_mask = np.ones(len(df), bool)
    r_full = stats.pearsonr(alb, y)
    rs_full = stats.spearmanr(alb, y)
    m_noM2 = df['site'].values != 'US-Minneapolis2'   # drop one of the shared-Tair pair
    r_noM2 = stats.pearsonr(alb[m_noM2], y[m_noM2])
    rs_noM2 = stats.spearmanr(alb[m_noM2], y[m_noM2])
    m_noLip = df['site'].values != 'PL-Lipowa'         # drop the negative anchor
    r_noLip = stats.pearsonr(alb[m_noLip], y[m_noLip])
    rs_noLip = stats.spearmanr(alb[m_noLip], y[m_noLip])
    print(f"  full (n={len(df)}):            Pearson r={r_full[0]:+.3f} (p={r_full[1]:.4f})  "
          f"Spearman={rs_full[0]:+.3f} (p={rs_full[1]:.4f})")
    print(f"  excl US-Minneapolis2 (n={m_noM2.sum()}): Pearson r={r_noM2[0]:+.3f} (p={r_noM2[1]:.4f})  "
          f"Spearman={rs_noM2[0]:+.3f} (p={rs_noM2[1]:.4f})")
    print(f"  excl PL-Lipowa (n={m_noLip.sum()}):       Pearson r={r_noLip[0]:+.3f} (p={r_noLip[1]:.4f})  "
          f"Spearman={rs_noLip[0]:+.3f} (p={rs_noLip[1]:.4f})")

    print(f"\nCSV written: {OUT_CSV}")
    return df, prim, expl, (r_full, r_noM2, r_noLip)


if __name__ == '__main__':
    main()
