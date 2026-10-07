# -*- coding: utf-8 -*-
"""Decompose each scheme's nocturnal error budget into a MEAN-BIAS part and a
TEMPORAL-STRUCTURE part, and ask which one drives its performance against the
out-of-sample benchmarks.

Why this is needed: the benchmarks are scored on MAE, which penalises mean bias
directly, so "TEB fails the benchmarks" and "TEB has poor temporal structure"
are different claims and the manuscript must not conflate them. It also
establishes, analytically, why the cross-site axis can rank the schemes the
other way round: the observed-aligned slope beta = r*(s_m/s_o) is EXACTLY
invariant to adding any constant to every record's modelled offset, whereas MAE
is not.

Reads the merged statistics package only; writes
results/review5_benchmark_decomposition.json for the lead to merge.
"""
import io, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sp = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
B = sp["review5_additions"]["nocturnal_benchmark_skill"]
PR = B["per_record_skill"]
MAE = B["benchmark"]["per_record_mae"]
PS = sp["per_site"]
SITES = sorted(PR["TEB"])

out = {"_merge_target": "review5_additions.benchmark_error_decomposition",
       "meta": {"script": "review5_benchmark_decomposition.py",
                "question": ("is each scheme's out-of-sample benchmark result driven by its "
                             "mean bias or by its temporal structure?"),
                "identity": "RMSE^2 = MBE^2 + cRMSE^2 (verified per record)",
                "debiased_proxy": ("cRMSE is used as the error a mean-shift-corrected scheme "
                                   "would retain; it is a PROXY, since MAE and RMSE differ by "
                                   "a distribution-shape factor, and is labelled as such")},
       "schemes": {}}

for scheme in ("TEB", "CLMU5"):
    d = MAE[scheme]
    mbe = np.array([PR[scheme][s]["MBE"] for s in SITES], float)
    rmse = np.array([PR[scheme][s]["RMSE"] for s in SITES], float)
    crmse = np.array([PR[scheme][s]["cRMSE"] for s in SITES], float)
    r = np.array([PR[scheme][s]["r"] for s in SITES], float)
    mmae = np.array([d[s]["scheme_MAE"] for s in SITES], float)
    ident = float(np.max(np.abs(rmse ** 2 - (mbe ** 2 + crmse ** 2))))
    share = mbe ** 2 / (mbe ** 2 + crmse ** 2)

    blk = {
        "identity_max_abs_residual": ident,
        "abs_MBE": {"median": float(np.median(np.abs(mbe))),
                    "min": float(np.abs(mbe).min()), "max": float(np.abs(mbe).max())},
        "cRMSE": {"median": float(np.median(crmse)),
                  "min": float(crmse.min()), "max": float(crmse.max())},
        "r": {"median": float(np.median(r)), "min": float(r.min())},
        "bias_share_of_MSE": {"median": float(np.median(share)),
                              "min": float(share.min()), "max": float(share.max()),
                              "n_records_bias_dominated": int((share > 0.5).sum()),
                              "n_records": len(SITES)},
        "model_MAE": {"median": float(np.median(mmae))},
        "corr_MAE_with_absMBE": float(np.corrcoef(mmae, np.abs(mbe))[0, 1]),
        "corr_MAE_with_cRMSE": float(np.corrcoef(mmae, crmse)[0, 1]),
        "per_benchmark": {},
    }
    for bn in ("REG2", "KM3"):
        b = np.array([d[s][f"{bn}_MAE"] for s in SITES], float)
        marg = mmae - b
        blk["per_benchmark"][bn] = {
            "wins_actual": int((marg < 0).sum()),
            "wins_if_own_mean_bias_removed_cRMSE_proxy": int((crmse < b).sum()),
            "corr_margin_with_absMBE": float(np.corrcoef(marg, np.abs(mbe))[0, 1]),
            "corr_margin_with_cRMSE": float(np.corrcoef(marg, crmse)[0, 1]),
            "benchmark_MAE_median": float(np.median(b)),
        }
    out["schemes"][scheme] = blk

# ---- analytic invariance of the cross-site slope to a uniform bias ----
obs = np.array([PS[s]["dtsa_std"] for s in SITES], float)
chk = {}
for scheme, fld in (("TEB", "teb_dtsa"), ("CLMU5", "clmu_dtsa")):
    m = np.array([PS[s][fld] for s in SITES], float)
    def beta(x):
        rr = float(np.corrcoef(obs, x)[0, 1])
        return rr * (x.std(ddof=1) / obs.std(ddof=1)), rr
    b0, r0 = beta(m)
    worst = 0.0
    for c in (-5.0, -1.0, 1.0, 5.0, 20.0):
        b1, r1 = beta(m + c)
        worst = max(worst, abs(b1 - b0), abs(r1 - r0))
    chk[scheme] = {"beta": b0, "r": r0,
                   "max_abs_change_under_uniform_shift": worst,
                   "shifts_tested_K": [-5, -1, 1, 5, 20]}
out["cross_site_slope_invariance_to_uniform_bias"] = {
    "statement": ("beta = r*(s_m/s_o) and r are exactly invariant to adding a constant to "
                  "every record's modelled offset, whereas MAE and RMSE are not; this is why "
                  "the level and cross-site axes can rank the two schemes oppositely"),
    "numerical_check": chk}

p = ROOT / "results/review5_benchmark_decomposition.json"
with io.open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    for scheme, blk in out["schemes"].items():
        print(f"=== {scheme} ===")
        print(f"  identity residual {blk['identity_max_abs_residual']:.2e}")
        print(f"  |MBE| median {blk['abs_MBE']['median']:.2f}   "
              f"cRMSE median {blk['cRMSE']['median']:.2f}   r median {blk['r']['median']:.3f}")
        print(f"  bias share of MSE median {blk['bias_share_of_MSE']['median']*100:.1f}%  "
              f"({blk['bias_share_of_MSE']['n_records_bias_dominated']}/"
              f"{blk['bias_share_of_MSE']['n_records']} bias-dominated)")
        print(f"  corr(MAE,|MBE|) {blk['corr_MAE_with_absMBE']:+.3f}   "
              f"corr(MAE,cRMSE) {blk['corr_MAE_with_cRMSE']:+.3f}")
        for bn, v in blk["per_benchmark"].items():
            print(f"    {bn}: wins {v['wins_actual']}/19 -> debiased proxy "
                  f"{v['wins_if_own_mean_bias_removed_cRMSE_proxy']}/19")
    print("\ncross-site slope invariance to a uniform bias:")
    for s, v in out["cross_site_slope_invariance_to_uniform_bias"]["numerical_check"].items():
        print(f"  {s}: beta {v['beta']:.4f} r {v['r']:+.4f}  "
              f"max change under shifts {v['max_abs_change_under_uniform_shift']:.2e}")
    print(f"\nwrote {p}")
