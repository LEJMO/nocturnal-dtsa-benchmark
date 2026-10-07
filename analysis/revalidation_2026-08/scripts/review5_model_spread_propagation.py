# -*- coding: utf-8 -*-
"""review5_model_spread_propagation.py -- paired night-block propagation of temporal
sampling uncertainty into the MODEL/OBS across-site spread ratios.

WHY THIS EXISTS SEPARATELY. review5_temporal_seasonal.py computes the TEB half
inline. The CLM-Urban half could not be computed in the same round because the
CLMU output alignment was being corrected (GR-HECKOR, UK-KingsCollege), which
changes per_site.clmu_dtsa and every CLMU-derived statistic. This script is the
parameterised runner for the CLMU half; it uses the identical machinery, block
lengths and seeds, so the TEB and CLMU columns remain comparable.

RUN AFTER THE CLMU REGENERATION:

    python review5_model_spread_propagation.py --model clmu \
        --out ../../../results/review5_temporal-seasonal_clmu.json

and, if the corrected alignment offsets are not the ones this script's Tair-echo
search finds, pass them explicitly (they are printed by clmu_post.py as
`align_offset`):

    python review5_model_spread_propagation.py --model clmu \
        --offsets '{"GR-HECKOR": 175320, "UK-KingsCollege": 175320}'

To reproduce the TEB half that is already inside the main script (regression test):

    python review5_model_spread_propagation.py --model teb

Model series conventions (frozen):
  teb  : external/teb_runs/<SITE>/output/LWU_base.txt, row i -> forcing step i+1
  clmu : external/clmu_baseline19/<SITE>_base.nc, FIRE = upward LW, FLDS = downward
         LW, Tair echo used to locate the spin-up offset exactly as in clmu_post.py
Both are converted to a model dTsa series through the SAME frozen inversion
(SIG = 5.67e-8, EPS = 0.95) on the SAME nocturnal mask as the observations, so the
obs and model record means can be resampled with the same night blocks.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review5_temporal_seasonal import (           # noqa: E402  (same-folder import)
    ROOT, CORPUS, SIG, EPS, MIN_N, SEED_BASE, HEIGHT_MISMATCH,
    load_record, coverage, night_series, acf_and_timescale,
    boot_block_paired, variant_sets, inv_ts, pear,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def model_lwup_series(site, model, n, offsets):
    """Model upward-LW on the corpus grid (NaN where the model has no value)."""
    out = np.full(n, np.nan)
    if model == 'teb':
        p = ROOT / f'external/teb_runs/{site}/output/LWU_base.txt'
        if not p.exists():
            return out, None, {}
        raw = np.loadtxt(p)
        nmod = min(raw.size, n - 1)
        out[1:nmod + 1] = raw[:nmod]          # row i -> forcing step i+1
        return out, None, dict(alignment='row i -> forcing step i+1', n_model=int(raw.size))

    # ---- CLM-Urban -------------------------------------------------------
    p = ROOT / f'external/clmu_baseline19/{site}_base.nc'
    if not p.exists():
        return out, None, {}
    h = xr.open_dataset(p)
    fire = np.asarray(h['FIRE'].values).reshape(-1)
    flds = np.asarray(h['FLDS'].values).reshape(-1)
    tair_m = np.asarray((h['Tair'] if 'Tair' in h else h['TBOT']).values).reshape(-1)
    h.close()

    ds = xr.open_dataset(CORPUS / f'{site}.nc')
    ta = np.asarray(ds['forcing_Tair'].values, float)
    ds.close()

    if site in offsets:
        off = int(offsets[site])
        rmse = float(np.sqrt(np.nanmean((tair_m[off:off + n] - ta) ** 2)))
        src = 'supplied'
    else:
        off, rmse, src = None, 1e9, 'tair_echo_search'
        for cand in range(len(fire) - n, -1, -1):
            r = float(np.sqrt(np.nanmean((tair_m[cand:cand + n] - ta) ** 2)))
            if r < rmse:
                rmse, off = r, cand
            if r < 1e-3:
                break
    out[:] = fire[off:off + n]
    dwn = flds[off:off + n]
    return out, dwn, dict(alignment=f'spin-up offset {off} ({src})',
                          align_rmse_Tair=rmse, n_model=int(fire.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['teb', 'clmu'], required=True)
    ap.add_argument('--draws', type=int, default=5000)
    ap.add_argument('--offsets', default='{}',
                    help='JSON dict site -> CLMU spin-up offset, overriding the echo search')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    offsets = json.loads(args.offsets)

    recs, pairs, pair_reps, align = {}, {}, {}, {}
    for p in sorted(CORPUS.glob('*.nc')):
        rec = load_record(p.stem)
        if rec is not None:
            recs[p.stem] = rec
    # seed index MUST be the position in the 19 evaluable records (as the main
    # script enumerates them), not in the 20 corpus files, or the seeds diverge
    eval_sites = sorted(recs)

    for i, s in enumerate(eval_sites):
        rec = recs[s]
        ds = xr.open_dataset(CORPUS / f'{s}.nc')
        ld_corpus = np.asarray(ds['forcing_LWdown'].values, float)
        ta = np.asarray(ds['forcing_Tair'].values, float)
        ds.close()
        n = ld_corpus.size

        mlw, mld, meta = model_lwup_series(s, args.model, n, offsets)
        align[s] = meta
        if not np.isfinite(mlw).any():
            continue
        # the inversion uses the model's own downward LW when the model supplies it
        # (CLMU FLDS), otherwise the corpus forcing (TEB is driven by it directly)
        ld_use = mld if mld is not None else ld_corpus
        ok = rec['valid'] & np.isfinite(mlw) & np.isfinite(ld_use)
        if ok.sum() < MIN_N:
            continue
        mdt = np.full(n, np.nan)
        mdt[ok] = inv_ts(mlw[ok], ld_use[ok]) - ta[ok]
        rec['model'] = mdt

        ids = rec['night_id'][ok]
        uniq, inv = np.unique(ids, return_inverse=True)
        ns_o = dict(sum=np.bincount(inv, weights=rec['d'][ok]),
                    cnt=np.bincount(inv).astype(float), n_nights=int(uniq.size))
        ns_m = dict(sum=np.bincount(inv, weights=mdt[ok]),
                    cnt=ns_o['cnt'].copy(), n_nights=ns_o['n_nights'])

        # identical block length as the main script: seasonal-block L from the
        # observed nightly ACF of this record, capped at K/5
        nsd = night_series(rec, 'd')
        ac = acf_and_timescale(nsd['mean'], nsd['day_index'], nsd['moy'])
        K = nsd['n_nights']
        L = int(max(1, min(np.ceil(ac['raw']['tau_nights']), max(1, K // 5))))

        rng = np.random.default_rng(SEED_BASE + 5000 + 11 * i)   # same seed rule
        a, b = boot_block_paired(ns_o, ns_m, rng, args.draws, L)
        pairs[s] = dict(obs_mean_paired=float(ns_o['sum'].sum() / ns_o['cnt'].sum()),
                        model_mean_paired=float(ns_m['sum'].sum() / ns_m['cnt'].sum()),
                        n_paired_steps=int(ok.sum()), n_paired_nights=ns_o['n_nights'],
                        block_L=L, obs_se=float(a.std(ddof=1)), model_se=float(b.std(ddof=1)))
        pair_reps[s] = (a, b)
        print(f'  {s:18s} obs={pairs[s]["obs_mean_paired"]:+7.3f} '
              f'{args.model}={pairs[s]["model_mean_paired"]:+7.3f} L={L:3d} '
              f'SEobs={pairs[s]["obs_se"]:.4f} SEmod={pairs[s]["model_se"]:.4f}', flush=True)

    SETS = variant_sets(eval_sites)
    out = {'_merge_target': f'review5_additions.temporal_sampling.'
                            f'{args.model}_spread_propagation',
           'model': args.model.upper(), 'draws': args.draws,
           'alignment': align, 'per_record_paired': pairs,
           'note': 'obs and model record means resampled with the SAME night blocks; '
                   'ratio = model/obs across-record spread. Block length from the '
                   'observed nightly integral timescale (seasonal-block), capped at K/5.'}

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    for vname, ss in SETS.items():
        use = [s for s in ss if s in pair_reps]
        o0 = np.array([pairs[s]['obs_mean_paired'] for s in use])
        m0 = np.array([pairs[s]['model_mean_paired'] for s in use])
        O = np.column_stack([pair_reps[s][0] for s in use])
        M = np.column_stack([pair_reps[s][1] for s in use])
        sd = M.std(1, ddof=1) / O.std(1, ddof=1)
        rg = (M.max(1) - M.min(1)) / (O.max(1) - O.min(1))
        iq = ((np.percentile(M, 75, axis=1) - np.percentile(M, 25, axis=1)) /
              (np.percentile(O, 75, axis=1) - np.percentile(O, 25, axis=1)))
        Oc = O - O.mean(1, keepdims=True); Mc = M - M.mean(1, keepdims=True)
        rr = (Oc * Mc).sum(1) / np.sqrt((Oc * Oc).sum(1) * (Mc * Mc).sum(1))
        bt = (Oc * Mc).sum(1) / (Oc * Oc).sum(1)
        out[vname] = dict(
            sites=use,
            baseline=dict(sd_ratio=float(np.std(m0, ddof=1) / np.std(o0, ddof=1)),
                          range_ratio=float(np.ptp(m0) / np.ptp(o0)),
                          iqr_ratio=float((np.percentile(m0, 75) - np.percentile(m0, 25)) /
                                          (np.percentile(o0, 75) - np.percentile(o0, 25))),
                          r_model_obs=pear(o0, m0)[0],
                          slope_model_on_obs=float(np.polyfit(o0, m0, 1)[0])),
            sd_ratio_median=float(np.median(sd)), sd_ratio_ci95=ci(sd),
            range_ratio_median=float(np.median(rg)), range_ratio_ci95=ci(rg),
            iqr_ratio_median=float(np.median(iq)), iqr_ratio_ci95=ci(iq),
            r_model_obs_median=float(np.median(rr)), r_model_obs_ci95=ci(rr),
            slope_model_on_obs_median=float(np.median(bt)), slope_model_on_obs_ci95=ci(bt),
            frac_sd_ratio_ge_1=float(np.mean(sd >= 1.0)), frac_slope_ge_1=float(np.mean(bt >= 1.0)))
        print(f'{vname:12s} sd_ratio {out[vname]["baseline"]["sd_ratio"]:.4f} '
              f'ci {ci(sd)}  P(sd>=1)={out[vname]["frac_sd_ratio_ge_1"]:.4f}')

    dest = Path(args.out) if args.out else (
        ROOT / f'results/review5_temporal-seasonal_{args.model}.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    print('wrote', dest)


if __name__ == '__main__':
    main()
