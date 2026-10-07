# -*- coding: utf-8 -*-
"""Paired city-cluster bootstrap for the TEB-vs-CLM-Urban difference in
cross-site skill, on the ALIGNMENT-CORRECTED CLMU values.

The manuscript claims a "robust gap in ordering" between the two schemes. That
claim needs an interval on the DIFFERENCE, which nothing in the statistics
package currently provides. This computes, per record set:
  - each scheme's spread ratio, Pearson r, Spearman rho, and observed-aligned
    slope beta = r*(s_m/s_o), with cluster-bootstrap CIs
  - the PAIRED differences d_r, d_rho, d_beta with cluster-bootstrap CIs and
    one-sided tail probabilities
  - one-sided P(r <= 0) per scheme, and P(beta >= 1), which is the quantity the
    paper's central claim actually rests on
Resampling unit = city cluster (the same 16/15 clusters used elsewhere).

Writes results/review5_scheme_ordering_paired.json (merged separately).
"""
import io, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NDRAW = 100_000
SEED = 20260916


def spearman(a, b):
    def rk(x):
        x = np.asarray(x, float); o = x.argsort(); r = np.empty(len(x)); r[o] = np.arange(len(x))
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r); return (s / cnt)[inv]
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


def stats(o, m):
    o = np.asarray(o, float); m = np.asarray(m, float)
    r = float(np.corrcoef(o, m)[0, 1])
    sdr = float(m.std(ddof=1) / o.std(ddof=1))
    return dict(r=r, rho=spearman(o, m), sd_ratio=sdr, beta=r * sdr)


def main():
    sp = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
    PSD = sp["per_site"]
    S = sorted(PSD)
    clus = {s: PSD[s].get("cluster", s) for s in S}
    obs = np.array([PSD[s]["dtsa_std"] for s in S])
    teb = np.array([PSD[s]["teb_dtsa"] for s in S])
    clm = np.array([PSD[s]["clmu_dtsa"] for s in S])

    MASK = {
        "all19": np.ones(len(S), bool),
        "excl_mpls17": np.array([not s.startswith("US-Minneapolis") for s in S]),
        "core16": np.array([(not s.startswith("US-Minneapolis")) and s != "PL-Lipowa" for s in S]),
    }
    out = {"_merge_target": "review5_additions.scheme_ordering_paired",
           "meta": {"script": "review5_scheme_ordering_paired.py", "draws": NDRAW,
                    "seed": SEED, "resampling_unit": "city cluster",
                    "clmu_source": "alignment-corrected (clmu_post_v11)",
                    "note": ("the manuscript's 'robust gap in ordering' between TEB and "
                             "CLM-Urban requires an interval on the DIFFERENCE, which this "
                             "supplies; beta = r*(s_m/s_o)")}}

    def ci(v, lo=2.5, hi=97.5):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]

    for vk, mk in MASK.items():
        o, t, c = obs[mk], teb[mk], clm[mk]
        cl = np.array([clus[s] for s in np.array(S)[mk]])
        uc = sorted(set(cl.tolist()))
        idx_by_c = {u: np.where(cl == u)[0] for u in uc}
        pt_t, pt_c = stats(o, t), stats(o, c)

        rng = np.random.default_rng(SEED)
        keys = ["r", "rho", "sd_ratio", "beta"]
        bt = {k: [] for k in keys}; bc = {k: [] for k in keys}
        dif = {k: [] for k in keys}
        for _ in range(NDRAW):
            pick = rng.choice(len(uc), size=len(uc), replace=True)
            ii = np.concatenate([idx_by_c[uc[p]] for p in pick])
            if len(np.unique(o[ii])) < 3:      # degenerate design
                continue
            st, sc = stats(o[ii], t[ii]), stats(o[ii], c[ii])
            for k in keys:
                bt[k].append(st[k]); bc[k].append(sc[k]); dif[k].append(st[k] - sc[k])

        n_used = len(dif["r"])
        blk = {"n": int(mk.sum()), "n_clusters": len(uc), "draws_used": n_used,
               "draws_dropped": NDRAW - n_used,
               "TEB": {**pt_t, **{f"{k}_ci95": ci(bt[k]) for k in keys}},
               "CLMU5": {**pt_c, **{f"{k}_ci95": ci(bc[k]) for k in keys}},
               "paired_difference_TEB_minus_CLMU5": {
                   k: {"point": pt_t[k] - pt_c[k], "ci95": ci(dif[k]),
                       "P_le_0": float(np.mean(np.asarray(dif[k]) <= 0))} for k in keys},
               "one_sided": {
                   "TEB_P_r_le_0": float(np.mean(np.asarray(bt["r"]) <= 0)),
                   "CLMU5_P_r_le_0": float(np.mean(np.asarray(bc["r"]) <= 0)),
                   "TEB_P_beta_ge_1": float(np.mean(np.asarray(bt["beta"]) >= 1.0)),
                   "CLMU5_P_beta_ge_1": float(np.mean(np.asarray(bc["beta"]) >= 1.0)),
                   "TEB_P_sdratio_ge_1": float(np.mean(np.asarray(bt["sd_ratio"]) >= 1.0)),
                   "CLMU5_P_sdratio_ge_1": float(np.mean(np.asarray(bc["sd_ratio"]) >= 1.0))}}
        out[vk] = blk
        d = blk["paired_difference_TEB_minus_CLMU5"]
        print(f"[{vk}] n={blk['n']} clusters={blk['n_clusters']} used={n_used}")
        print(f"   TEB   r={pt_t['r']:+.4f} {blk['TEB']['r_ci95']}  beta={pt_t['beta']:.4f} "
              f"{blk['TEB']['beta_ci95']}  sdr={pt_t['sd_ratio']:.4f}")
        print(f"   CLMU5 r={pt_c['r']:+.4f} {blk['CLMU5']['r_ci95']}  beta={pt_c['beta']:.4f} "
              f"{blk['CLMU5']['beta_ci95']}  sdr={pt_c['sd_ratio']:.4f}")
        print(f"   d_r   = {d['r']['point']:+.4f} {d['r']['ci95']}  P(d<=0)={d['r']['P_le_0']:.3f}")
        print(f"   d_beta= {d['beta']['point']:+.4f} {d['beta']['ci95']}  P(d<=0)={d['beta']['P_le_0']:.3f}")
        print(f"   P(r<=0): TEB {blk['one_sided']['TEB_P_r_le_0']:.3f}  "
              f"CLMU5 {blk['one_sided']['CLMU5_P_r_le_0']:.3f}   "
              f"P(beta>=1): {blk['one_sided']['TEB_P_beta_ge_1']:.4f} / "
              f"{blk['one_sided']['CLMU5_P_beta_ge_1']:.4f}")
        print()

    p = ROOT / "results/review5_scheme_ordering_paired.json"
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", p)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
