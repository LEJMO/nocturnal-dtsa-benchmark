# -*- coding: utf-8 -*-
"""Emit Supplementary Table S3 (tab:b1): the supervisor's B-1 checks —
albedo-only, sky-emissivity-only and two-descriptor models of the record-mean
offset under leave-one-city-cluster-out, and the albedo relation inside the
Cfb climate class. Numbers from review8_additions (paper_stats_v1.json)."""
import io, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
S = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
B = S["review8_additions"]["b1_albedo_skyemissivity"]
SETS = [("Conservative core (16)", "core16"), ("Excl.\\ Minneapolis (17)", "excl_mpls17"), ("All 19 records", "all19")]
MODELS = [("albedo only", "albedo_only"), ("sky emissivity only", "skyemis_only"), ("albedo + sky emissivity", "albedo_plus_skyemis")]
rows = []
for lab, vk in SETS:
    blk = B["sets"][vk]
    for i, (ml, mk) in enumerate(MODELS):
        m = blk[mk]; coef = m["coef_K_per_unit"]
        ctxt = f"{coef[0]:+.1f}" if len(coef) == 1 else f"{coef[0]:+.1f} / {coef[1]:+.1f}"
        rows.append(f"{lab if i == 0 else ''} & {ml} & {ctxt} & {m['R2_LOCO']:+.2f} & {m['LOCO_MAE_K']:.2f} \\\\")
    rows.append(r"\addlinespace")
rows.pop()
c = B["cfb_subset"]
cap = r"\caption{Albedo and mean nocturnal sky emissivity as descriptors of the record-mean offset under leave-one-city-cluster-out, and the albedo relation within the Cfb climate class.}"
NOTE = (r"\par\vspace{2pt}{\scriptsize\raggedright Coefficients in K per unit of the descriptor (albedo / sky emissivity); "
        r"$R^2_{\mathrm{LOCO}}$ and held-out MAE as in main-text Sect.~2.6. Cfb subset (" + ", ".join(c["sites"]) + f"): $r={c['r']:+.2f}$, "
        f"slope ${c['slope_K_per_0p1_albedo']:+.1f}$~K per 0.1 albedo, exact two-sided permutation $p={c['exact_permutation_two_sided_p']:.3f}$ "
        f"({c['n_arrangements']} arrangements).\\par}}")
out = [r"\begin{table}[!tbp]", r"\centering\footnotesize", cap, r"\label{tab:b1}", r"\setlength{\tabcolsep}{5pt}",
       r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
       r"Record set & Descriptors & Coefficients (K per unit) & $R^2_{\mathrm{LOCO}}$ & Held-out MAE (K) \\", r"\midrule",
       *rows, r"\bottomrule", r"\end{tabular}", NOTE, r"\end{table}"]
p = ROOT / "paper/manuscript_dtsa/tables/tableS3_b1.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"wrote {p}"); [print("  ", r) for r in rows]
