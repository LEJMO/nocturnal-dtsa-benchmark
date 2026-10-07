# -*- coding: utf-8 -*-
"""Evaluate CLMU5 campaign histories (albedo / spin-up) with the validated
clmu_post metrics, compare to the paper's CLMU baseline (paper_stats
clmu_dtsa / clmu_lwup_bias), and summarize cross-site slopes.

Writes external/clmu_campaign_results.json. Does NOT modify paper_stats.
"""
import io, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clmu_post import metrics  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
P = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))["per_site"]
SITES = sorted(P)

out = {"meta": {"baseline_source": "paper_stats clmu_dtsa / clmu_lwup_bias",
                "post": "clmu_post.metrics (validated bit-exact on KR-Ochang)"}}

for tag, subdir, suffix in [("campaign_A_albedo", "clmu_albedo19", "_alb.nc"),
                            ("campaign_B_spinup", "clmu_spinup19", "_spin.nc")]:
    d = ROOT / f"external/{subdir}"
    if not d.exists():
        continue
    res = {}
    for s in SITES:
        f = d / f"{s}{suffix}"
        if not f.exists():
            res[s] = {"status": "MISSING"}; continue
        try:
            m = metrics(s, str(f))
            m["status"] = "OK"
            m["baseline_dtsa_K"] = P[s]["clmu_dtsa"]
            m["baseline_bias_Wm2"] = P[s]["clmu_lwup_bias"]
            m["obs_dtsa_K"] = P[s]["dtsa_std"]
            res[s] = m
        except Exception as e:
            res[s] = {"status": f"ERR:{e}"}
    ok = [s for s in SITES if res.get(s, {}).get("status") == "OK"]
    if ok:
        obs = np.array([P[s]["dtsa_std"] for s in ok])
        base = np.array([P[s]["clmu_dtsa"] for s in ok])
        camp = np.array([res[s]["dTsa_model"] for s in ok])
        out[tag + "_summary"] = {
            "n": len(ok),
            "baseline_slope_model_on_obs": float(np.polyfit(obs, base, 1)[0]),
            "baseline_r": float(pearsonr(obs, base)[0]),
            "campaign_slope_model_on_obs": float(np.polyfit(obs, camp, 1)[0]),
            "campaign_r": float(pearsonr(obs, camp)[0]),
            "baseline_span_K": [float(base.min()), float(base.max())],
            "campaign_span_K": [float(camp.min()), float(camp.max())],
            "obs_span_K": [float(obs.min()), float(obs.max())],
            "max_abs_dtsa_change_K": float(np.max(np.abs(camp - base))),
            "mean_abs_dtsa_change_K": float(np.mean(np.abs(camp - base))),
        }
        print(f"=== {tag} (n={len(ok)}) ===")
        cs = out[tag + "_summary"]
        print(f"  baseline slope {cs['baseline_slope_model_on_obs']:.3f} "
              f"span {np.round(cs['baseline_span_K'],2)}")
        print(f"  campaign slope {cs['campaign_slope_model_on_obs']:.3f} "
              f"span {np.round(cs['campaign_span_K'],2)}  "
              f"max|dtsa change|={cs['max_abs_dtsa_change_K']:.3f} K")
        for s in ok:
            print(f"    {s:16s} obs {P[s]['dtsa_std']:+.2f} | base {P[s]['clmu_dtsa']:+.2f} "
                  f"-> camp {res[s]['dTsa_model']:+.2f}")
    out[tag] = res

json.dump(out, open(ROOT / "external/clmu_campaign_results.json", "w",
                    encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nwrote external/clmu_campaign_results.json")
