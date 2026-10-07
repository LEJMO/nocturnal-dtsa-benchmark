# independent verification of the two load-bearing new numbers
from pathlib import Path
import json, io, numpy as np
from scipy.optimize import minimize, linprog

P = json.load(io.open(str(Path(__file__).resolve().parents[3] / 'results' / 'paper_stats_v1.json'),
                      encoding='utf-8'))['per_site']
S = sorted(P)
sets = {'all19': S,
        'excl_mpls17': [s for s in S if not s.startswith('US-Minneapolis')],
        'core16': [s for s in S if not s.startswith('US-Minneapolis') and s != 'PL-Lipowa'],
        'core15': [s for s in S if not s.startswith('US-Minneapolis')
                   and s not in ('PL-Lipowa', 'PL-Narutowicza')]}

def beta_of(o, m):
    o = np.asarray(o); m = np.asarray(m)
    return float(np.cov(o, m, ddof=1)[0, 1] / np.var(o, ddof=1))

print('=== (A) aligned step, plain numpy, no weighting machinery ===')
for v in ('core16', 'core15'):
    ss = sets[v]
    a = np.array([P[s]['albedo'] for s in ss]); o = np.array([P[s]['dtsa_std'] for s in ss])
    for sch, key in (('TEB', 'teb_dtsa'), ('CLMU5', 'clmu_dtsa')):
        m = np.array([P[s][key] for s in ss])
        for B in (0.25, 0.5, 0.75, 1.0):
            o2 = o + B * np.sign(a - a.mean())
            sdr = m.std(ddof=1) / o2.std(ddof=1)
            r = np.corrcoef(o2, m)[0, 1]
            print(f'  {v:8s} {sch:6s} B={B:4.2f}  s_o={o2.std(ddof=1):.4f}  '
                  f'sdr={sdr:.4f}  r={r:+.4f}  beta={r*sdr:+.4f}  '
                  f'beta_cov/var={beta_of(o2, m):+.4f}')

print('\n=== (B) sup beta over the box: brute force vs the reported value ===')
rep = json.load(io.open(str(Path(__file__).resolve().parents[3] / 'results' / 'review5_obs_error_into_model.json'), encoding='utf-8'))
adv = rep['adversarial_bounds_aimed_at_beta']
rng = np.random.default_rng(7)
for v in ('core16', 'core15', 'all19'):
    ss = sets[v]; o = np.array([P[s]['dtsa_std'] for s in ss]); n = len(ss)
    for sch, key in (('TEB', 'teb_dtsa'), ('CLMU5', 'clmu_dtsa')):
        m = np.array([P[s][key] for s in ss])
        for B in (0.5, 1.0):
            claim = adv['variants'][v]['amplitudes'][f'B_{B:g}K'][sch]['sup_beta']
            # brute force 1: random uniform box sampling
            D = rng.uniform(-B, B, size=(400000, n))
            O = o[None, :] + D
            oc = O - O.mean(1)[:, None]; mc = m - m.mean()
            bb = (oc @ mc) / (oc * oc).sum(1)
            # brute force 2: local ascent from 200 random starts on beta directly
            def negbeta(x):
                z = o + x; zc = z - z.mean()
                return -float((zc @ mc) / (zc @ zc))
            best = -np.inf
            for _ in range(150):
                x0 = rng.uniform(-B, B, n)
                res = minimize(negbeta, x0, method='L-BFGS-B',
                               bounds=[(-B, B)] * n, options={'maxiter': 5000})
                best = max(best, -res.fun)
            # brute force 3: all 2^n vertices for the small sets
            vert = -np.inf
            if n <= 16:
                bits = np.arange(n)
                for lo in range(0, 1 << n, 1 << 14):
                    idx = np.arange(lo, min(lo + (1 << 14), 1 << n))
                    Sm = (((idx[:, None] >> bits) & 1) * 2.0 - 1.0) * B
                    OO = o[None, :] + Sm
                    ooc = OO - OO.mean(1)[:, None]
                    vert = max(vert, float(((ooc @ mc) / (ooc * ooc).sum(1)).max()))
            print(f'  {v:8s} {sch:6s} B={B:3.1f} reported_sup={claim:.6f}  '
                  f'random_max={bb.max():.6f}  ascent_max={best:.6f}  '
                  f'vertex_max={vert if np.isfinite(vert) else float("nan"):.6f}  '
                  f'OK_upper_bound={bool(claim >= max(bb.max(), best, vert if np.isfinite(vert) else -9) - 1e-6)}')

print('\n=== (C) B*(beta=1): is beta=1 attainable at B* and not below it? ===')
for v in ('core16', 'core15'):
    ss = sets[v]; o = np.array([P[s]['dtsa_std'] for s in ss]); n = len(ss)
    for sch, key in (('TEB', 'teb_dtsa'), ('CLMU5', 'clmu_dtsa')):
        m = np.array([P[s][key] for s in ss]); mc = m - m.mean()
        Bs = adv['beta_breakdown_amplitude'][v][sch]['B_star_K']
        def sup_at(B):
            def negbeta(x):
                z = o + x; zc = z - z.mean()
                return -float((zc @ mc) / (zc @ zc))
            best = -np.inf
            for _ in range(300):
                res = minimize(negbeta, rng.uniform(-B, B, n), method='L-BFGS-B',
                               bounds=[(-B, B)] * n, options={'maxiter': 5000})
                best = max(best, -res.fun)
            return best
        lo = sup_at(Bs * 0.98); at = sup_at(Bs * 1.0005)
        print(f'  {v:8s} {sch:6s} B*={Bs:.4f}  sup_beta(0.98 B*)={lo:.4f} (<1? {lo < 1})  '
              f'sup_beta(1.0005 B*)={at:.4f} (>=1? {at >= 1})')
