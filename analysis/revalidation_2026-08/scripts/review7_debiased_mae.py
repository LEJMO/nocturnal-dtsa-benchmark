# -*- coding: utf-8 -*-
"""Exact 'own mean bias removed' benchmark count (v15 review).

The v15 manuscript said that removing each scheme's own mean bias would lift
TEB from 10 to 16 of 19 records against the KM3 benchmark, computed by
comparing the scheme's centred RMSE with the benchmark MAE. Because
mean|e - mean(e)| <= sqrt(mean((e - mean(e))^2)) for any residual series, the
centred-RMSE proxy undercounts: 16 is a LOWER BOUND, not the count. This
script computes the debiased MAE directly from the same aligned nocturnal
frames the published skill scores use (imported read-only from
review5_benchmark_skill.py; no metric code is modified) and compares it with
the leave-one-city-cluster-out benchmark MAEs of the published city-holdout
package. Output: results/review7_debiased_mae.json (merged into
paper_stats_v1.json by review5_merge.py).
"""
import io, sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review5_benchmark_skill as bs      # frames + ROOT only; main() is guarded

ROOT = bs.ROOT
S = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
A = S["review5_additions"]
PRS = A["nocturnal_benchmark_skill"]["per_record_skill"]
CITY = A["benchmark_city_holdout"]["city_holdout"]["per_record_mae"]
PUB = A["nocturnal_benchmark_skill"]["benchmark"]["per_record_mae"]
PROXY = A["benchmark_error_decomposition"]["schemes"]

out = {
    "_merge_target": "review5_additions.benchmark_debiased_mae",
    "meta": {
        "purpose": ("exact count of records at which a scheme with its own record-mean "
                    "bias removed would beat each forcing-based benchmark; replaces the "
                    "centred-RMSE proxy, which is a lower bound because "
                    "mean|e-mean(e)| <= sqrt(mean((e-mean(e))^2))"),
        "generator": "review7_debiased_mae.py (imports review5_benchmark_skill frames read-only)",
        "definition": "MAE_debiased = mean(|e - mean(e)|), e = model - obs on the published nocturnal frame",
        "benchmarks": "leave-one-city-cluster-out MAEs from review5_additions.benchmark_city_holdout (primary) and the first-published leave-one-record-out MAEs (secondary)",
    },
    "schemes": {},
}
for sch in ("TEB", "CLMU5"):
    per = {}
    for rec in PRS[sch]:
        f = bs.FRAME[sch](rec)
        e = f["model"] - f["obs"]
        mbe = float(np.mean(e)); ed = e - mbe
        mae = float(np.mean(np.abs(e)))
        mae_deb = float(np.mean(np.abs(ed)))
        crmse = float(np.sqrt(np.mean(ed ** 2)))
        # the frame must reproduce the published per-record scores exactly
        assert abs(crmse - PRS[sch][rec]["cRMSE"]) < 1e-6, (sch, rec, crmse, PRS[sch][rec]["cRMSE"])
        assert abs(mbe - PRS[sch][rec]["MBE"]) < 1e-6, (sch, rec)
        assert abs(mae - CITY[sch][rec]["scheme_MAE"]) < 1e-6, (sch, rec)
        c, p = CITY[sch][rec], PUB[sch][rec]
        per[rec] = {
            "n": int(e.size), "MBE": mbe, "MAE": mae, "cRMSE": crmse,
            "MAE_debiased": mae_deb, "MAE_debiased_le_cRMSE": bool(mae_deb <= crmse + 1e-12),
            "KM3_MAE_city": c["KM3_MAE"], "REG2_MAE_city": c["REG2_MAE"],
            "beats_KM3_debiased_city": bool(mae_deb < c["KM3_MAE"]),
            "beats_REG2_debiased_city": bool(mae_deb < c["REG2_MAE"]),
            "beats_KM3_cRMSEproxy_city": bool(crmse < c["KM3_MAE"]),
            "beats_REG2_cRMSEproxy_city": bool(crmse < c["REG2_MAE"]),
            "beats_KM3_debiased_recordLOO": bool(mae_deb < p["KM3_MAE"]),
            "beats_REG2_debiased_recordLOO": bool(mae_deb < p["REG2_MAE"]),
        }
    n = len(per)
    cnt = lambda k: int(sum(1 for r in per.values() if r[k]))
    counts = {
        "n_records": n,
        "actual_beats_KM3_city": int(sum(1 for r in CITY[sch].values() if r["beats_KM3"])),
        "actual_beats_REG2_city": int(sum(1 for r in CITY[sch].values() if r["beats_REG2"])),
        "debiased_beats_KM3_city": cnt("beats_KM3_debiased_city"),
        "debiased_beats_REG2_city": cnt("beats_REG2_debiased_city"),
        "debiased_beats_both_city": int(sum(1 for r in per.values() if r["beats_KM3_debiased_city"] and r["beats_REG2_debiased_city"])),
        "cRMSEproxy_beats_KM3_city": cnt("beats_KM3_cRMSEproxy_city"),
        "cRMSEproxy_beats_REG2_city": cnt("beats_REG2_cRMSEproxy_city"),
        "debiased_beats_KM3_recordLOO": cnt("beats_KM3_debiased_recordLOO"),
        "debiased_beats_REG2_debiased_recordLOO": cnt("beats_REG2_debiased_recordLOO"),
        "inequality_holds_all_records": all(r["MAE_debiased_le_cRMSE"] for r in per.values()),
        "published_proxy_KM3": PROXY[sch]["per_benchmark"]["KM3"]["wins_if_own_mean_bias_removed_cRMSE_proxy"],
        "published_proxy_REG2": PROXY[sch]["per_benchmark"]["REG2"]["wins_if_own_mean_bias_removed_cRMSE_proxy"],
    }
    assert counts["cRMSEproxy_beats_KM3_city"] == counts["published_proxy_KM3"]
    assert counts["debiased_beats_KM3_city"] >= counts["cRMSEproxy_beats_KM3_city"]
    out["schemes"][sch] = {"per_record": per, "counts": counts}

p = ROOT / "results/review7_debiased_mae.json"
io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1, ensure_ascii=False))
if __name__ == "__main__":
    # review5_benchmark_skill already re-wrapped sys.stdout at import; wrapping
    # it again closes the shared buffer, so print through the existing stream.
    print("wrote", p)
    for sch in ("TEB", "CLMU5"):
        print(sch, json.dumps(out["schemes"][sch]["counts"]))
