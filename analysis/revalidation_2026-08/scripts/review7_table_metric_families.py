# -*- coding: utf-8 -*-
"""Emit the metric-families table (tab:families): the three metric families on
the same output, side by side.

v15 (reviewer of v14): the benchmark facts were presented only in prose inside
the Discussion. This table puts per-record temporal skill, level, the
out-of-sample benchmark counts and the cross-site statistics side by side in
Results, so the disagreement between the families is visible at a glance.
v27 (supervisor, A-5): the conservative core (16 records) is the headline
record set, so the table carries a core column pair beside the all-19 pair.
Every number is read from results/paper_stats_v1.json: the all-19 values are
asserted identical to the v26 sources (review5/review7 blocks), the core
values come from review8_additions.core16_metric_families.
"""
import io, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
A = S["review5_additions"]
SK = A["nocturnal_benchmark_skill"]["skill_summary"]
PRS = A["nocturnal_benchmark_skill"]["per_record_skill"]
BIAS = S["review3_additions"]["bias_aggregates_Wm2"]
DEC = A["benchmark_error_decomposition"]["schemes"]
CH = A["benchmark_city_holdout"]["city_holdout"]
AOD = S["amplitude_ordering_decomposition"]
SOP = A["scheme_ordering_paired"]
MF = S["review8_additions"]["core16_metric_families"]

SCH = [("TEB", "TEB"), ("CLM-Urban", "CLMU5")]

def fmt_beta(a, lo, hi):
    return f"{a['slope']:.2f} [{lo:.2f}, {hi:.2f}]".replace("[-", "[$-$").replace(", -", ", $-$")

col = {}
for disp, k in SCH:
    # ---- all 19 records: the v26 sources, cross-checked against review8
    w = CH["win_counts_eval_only"][k]["all19"]; assert w["n"] == 19
    pm = CH["per_record_mae"][k]
    proxy_km3 = sum(1 for r in pm if PRS[k][r]["cRMSE"] < pm[r]["KM3_MAE"])
    assert proxy_km3 == DEC[k]["per_benchmark"]["KM3"]["wins_if_own_mean_bias_removed_cRMSE_proxy"]
    DEB = A["benchmark_debiased_mae"]["schemes"][k]["counts"]
    assert DEB["inequality_holds_all_records"] and DEB["cRMSEproxy_beats_KM3_city"] == proxy_km3
    m19 = MF["all19"][k]
    assert abs(m19["median_r"] - SK[k]["r"]["median"]) < 1e-9
    assert abs(m19["record_mean_bias_Wm2"] - BIAS["record_equal_mean"][k]) < 1e-6
    assert abs(m19["sample_weighted_bias_Wm2"] - BIAS["sample_weighted_mean"][k]) < 1e-6
    assert m19["beats_REG2"] == w["beats_REG2"] and m19["beats_KM3"] == w["beats_KM3"]
    assert m19["beats_both"] == w["beats_meaningful_two_REG2_KM3"]
    assert m19["debiased_beats_KM3"] == DEB["debiased_beats_KM3_city"]
    assert m19["debiased_beats_REG2"] == DEB["debiased_beats_REG2_city"]
    assert round(m19["median_bias_share_of_MSE"], 2) == round(DEC[k]["bias_share_of_MSE"]["median"], 2)
    for vk in ("core16", "all19"):
        m = MF[vk][k]; a = AOD[vk][k]; lo, hi = SOP[vk][k]["beta_ci95"]
        P = SOP[vk]["one_sided"][f"{k}_P_beta_ge_1"]
        col[(k, vk)] = {
            "r": f"{m['median_r']:.3f}", "d": f"{m['median_d']:.2f}", "crmse": f"{m['median_cRMSE_Wm2']:.1f}",
            "bias_rec": f"{m['record_mean_bias_Wm2']:+.1f}", "bias_wt": f"{m['sample_weighted_bias_Wm2']:+.1f}",
            "mae": f"{m['median_MAE_Wm2']:.1f}", "bias_share": f"{100*m['median_bias_share_of_MSE']:.0f}\\%",
            "reg2": str(m["beats_REG2"]), "km3": str(m["beats_KM3"]), "both": str(m["beats_both"]),
            "deb_reg2": str(m["debiased_beats_REG2"]), "deb_km3": str(m["debiased_beats_KM3"]),
            "sdr": f"{a['sd_ratio']:.2f}", "rmo": f"{a['r']:+.2f}", "beta": fmt_beta(a, lo, hi),
            "P": ("$<0.001$" if P < 0.001 else f"{P:.3f}"),
        }

ORDER = [("TEB", "core16"), ("CLMU5", "core16"), ("TEB", "all19"), ("CLMU5", "all19")]
def row(label, key): return f"{label} & " + " & ".join(col[o][key] for o in ORDER) + r" \\"
def head(txt): return r"\multicolumn{5}{@{}l}{\emph{" + txt + r"}} \\"

cap = r"\caption{Three metric families on the same nocturnal LW$_{\uparrow}$ output of the two runs, in the conservative core and on all records.}"
out = [r"\begin{table}[!tbp]", r"\centering\footnotesize", cap, r"\label{tab:families}",
       r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
       r" & \multicolumn{2}{c}{Conservative core (16)} & \multicolumn{2}{c}{All 19 records} \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
       r" & TEB & CLM-Urban & TEB & CLM-Urban \\", r"\midrule",
       head(r"Per-record temporal skill (median over records)"),
       row(r"Pearson $r$", "r"), row(r"Index of agreement $d$", "d"), row(r"Centred RMSE (W\,m$^{-2}$)", "crmse"),
       r"\addlinespace", head(r"Level"),
       row(r"Record-mean bias (W\,m$^{-2}$)", "bias_rec"), row(r"Sample-weighted bias (W\,m$^{-2}$)", "bias_wt"),
       row(r"Median MAE (W\,m$^{-2}$)", "mae"), row(r"Median share of MSE from mean bias", "bias_share"),
       r"\addlinespace", head(r"Out-of-sample benchmark wins (records)"),
       row(r"Beats REG2", "reg2"), row(r"Beats KM3", "km3"), row(r"Beats both", "both"),
       r"\addlinespace", head(r"Same, with own record-mean bias removed (diagnostic)"),
       row(r"Beats REG2", "deb_reg2"), row(r"Beats KM3", "deb_km3"),
       r"\addlinespace", head(r"Cross-site gradient (Table~\ref{tab:modelvar})"),
       row(r"Spread ratio $s_m/s_o$", "sdr"), row(r"Ordering $r_{m,o}$", "rmo"),
       row(r"Slope $\hat\beta_{m|o}$ [95\% CI]", "beta"), row(r"$P(\hat\beta_{m|o}\ge1)$", "P"),
       r"\bottomrule", r"\end{tabular}", r"\end{table}"]

p = ROOT / "paper/manuscript_dtsa/tables/table5_metric_families.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"wrote {p}")
    for o in ORDER: print(" ", o, json.dumps(col[o]))
