# -*- coding: utf-8 -*-
"""Emit Supplementary Table S4 (tab:materialK): the one-at-a-time material
perturbations of Sect. 3.4 expressed as the change of the nocturnal LW_up bias
and its kelvin equivalent, per probe site and configuration. Numbers from
review8_additions.material_sensitivity_K (paper_stats_v1.json)."""
import io, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
S = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
M = S["review8_additions"]["material_sensitivity_K"]
K = S["review8_additions"]["meta"]["K_per_Wm2"]
TEB_CFG = [("TI_half", "heat capacity $\\times0.5$"), ("TI_double", "heat capacity $\\times2$"),
           ("TC_half", "conductivity $\\times0.5$"), ("TC_double", "conductivity $\\times2$"),
           ("ALB_site", "observed albedo"), ("EMIS_low", "emissivity reduced"), ("ROOF_light", "lightweight roof")]
CLMU_CFG = [("CV_half", "heat capacity $\\times0.5$"), ("CV_double", "heat capacity $\\times2$"),
            ("TK_half", "conductivity $\\times0.5$"), ("TK_double", "conductivity $\\times2$"),
            ("ALB_site", "observed albedo"), ("EMIS_low", "emissivity reduced")]
def block(scheme, cfgs):
    sites = sorted(M[scheme], key=lambda s: M[scheme][s]["base"]["lwup_bias_Wm2"])
    head = r"\multicolumn{" + str(len(sites) + 1) + r"}{@{}l}{\emph{" + ("TEB" if scheme == "TEB" else "CLM-Urban") + r"}} \\"
    hdr = "Perturbation & " + " & ".join(sites) + r" \\"
    base = "default bias (W\\,m$^{-2}$) & " + " & ".join(f"{M[scheme][s]['base']['lwup_bias_Wm2']:+.1f}" for s in sites) + r" \\"
    lines = [head, hdr, r"\midrule", base, r"\addlinespace"]
    for key, lab in cfgs:
        cells = []
        for s in sites:
            v = M[scheme][s].get(key)
            cells.append(f"{v['delta_Wm2']:+.1f} ({v['delta_K']:+.2f})" if v else "--")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return lines
cap = r"\caption{One-at-a-time material perturbations: change of the record-mean nocturnal LW$_{\uparrow}$ bias relative to the default configuration, W\,m$^{-2}$ (kelvin equivalent in parentheses).}"
NOTE = (r"\par\vspace{2pt}{\scriptsize\raggedright Kelvin equivalent at " + f"{K:.3f}" + r"~K per W\,m$^{-2}$ (Sect.~2.6). A positive change warms the modelled surface. "
        r"No perturbation changes the sign of the default bias at any site (Sect.~3.4).\par}")
out = [r"\begin{table}[!tbp]", r"\centering\footnotesize", cap, r"\label{tab:materialK}", r"\setlength{\tabcolsep}{3pt}", r"\begingroup\scriptsize",
       r"\begin{tabular}{@{}l" + "c" * 5 + r"@{}}", r"\toprule", *block("TEB", TEB_CFG), r"\midrule", *block("CLMU5", CLMU_CFG),
       r"\bottomrule", r"\end{tabular}", r"\endgroup", NOTE, r"\end{table}"]
p = ROOT / "paper/manuscript_dtsa/tables/tableS4_material_K.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"wrote {p}")
