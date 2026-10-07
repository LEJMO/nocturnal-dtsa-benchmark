# -*- coding: utf-8 -*-
"""Emit Supplementary Table S2 (tab:cells): the composition of the 208
observation-error scenario cells evaluated on the model metrics, and the six
cells in which the city-cluster bootstrap 95% interval of the observed-aligned
slope reaches unity. Every entry is read from results/paper_stats_v1.json
(review5_additions.obs_error_propagated_to_model)."""
import io, sys, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
E = S["review5_additions"]["obs_error_propagated_to_model"]
cells = []
def walk(o, path):
    if isinstance(o, dict):
        if "P_beta_ge_1" in o and "beta_ci95" in o:
            cells.append((path, o["P_beta_ge_1"], o["beta_ci95"], o["beta_median"], o["n_draws"]))
        for k, v in o.items(): walk(v, path + [k])
walk(E["scenarios"], [])
assert len(cells) == 208 == E["headline"]["n_scenario_cells"]

FAM = {"random_independent": r"independent random LW$_{\uparrow}$ error ($\sigma_E$ = 2, 4, 5, 10 W\,m$^{-2}$)",
       "random_shared_minneapolis": r"random error shared by the Minneapolis pair (same four $\sigma_E$)",
       "adderley_placement": r"placement-calibrated error \citep{Adderley2015}: per-record $\sigma$ capped at 11.2 / uncapped, flat 11.2, flat 9.4 W\,m$^{-2}$",
       "common_mode_scale_error": r"per-cent-of-reading scale error: common-mode $\pm$1, 2, 3\%; random-sign 1, 2, 3\%",
       "aligned_step": r"albedo-aligned $\pm B$ step field ($B$ = 0.25, 0.5, 0.75, 1 K)",
       "box_obs_argmax": r"box-search vertex field (same four $B$)"}
SHORT = {"random_independent": r"independent random error", "random_shared_minneapolis": r"shared Minneapolis error",
         "adderley_placement": r"placement-calibrated error", "common_mode_scale_error": r"scale error",
         "aligned_step": r"albedo-aligned step field", "box_obs_argmax": r"box-search vertex field"}
DESIGN = {"random_independent": "joint, 50\\,000", "random_shared_minneapolis": "joint, 50\\,000",
          "adderley_placement": "joint, 50\\,000", "common_mode_scale_error": "joint, 50\\,000",
          "aligned_step": "deterministic, 100\\,000", "box_obs_argmax": "deterministic, 100\\,000"}
SETS = {"all19": "all 19", "excl_mpls17": "excl.\\ Minneapolis", "core16": "core", "core15": "core excl.\\ Narutowicza"}
SCH = {"TEB": "TEB", "CLMU5": "CLM-Urban"}

def level_of(path):
    """The level key is the path segment immediately before the scheme key."""
    i = next(k for k, seg in enumerate(path) if seg in SCH)
    return path[i - 1]

rows_a = []
for fam in FAM:
    fc = [c for c in cells if c[0][0] == fam]
    levels = sorted({level_of(c[0]) for c in fc})
    sets = sorted({seg for c in fc for seg in c[0] if seg in SETS}, key=list(SETS).index)
    n_inc = sum(1 for c in fc if c[2][1] >= 1.0)
    # the printed arithmetic must hold: cells = levels x record sets x schemes
    assert len(fc) == len(levels) * len(sets) * 2, (fam, len(fc), levels, sets)
    rows_a.append(f"{FAM[fam]} & {DESIGN[fam]} & {len(levels)} & {len(sets)} & {len(fc)} & "
                  f"{max(c[1] for c in fc):.4f} & {n_inc} \\\\")
assert sum(len([c for c in cells if c[0][0] == fam]) for fam in FAM) == 208
tot = f"\\midrule\nTotal & & & & {len(cells)} & {max(c[1] for c in cells):.4f} & {sum(1 for c in cells if c[2][1] >= 1.0)} \\\\"

exc = sorted([c for c in cells if c[2][1] >= 1.0], key=lambda c: -c[1])
assert len(exc) == 6
rows_b = []
for path, P, ci, med, n in exc:
    fam = path[0]; rs = next(seg for seg in path if seg in SETS); lev = level_of(path)
    sch = next(seg for seg in path if seg in SCH)
    lev_txt = lev.replace("B_", "$B=").replace("K", "$~K") if lev.startswith("B_") else lev
    rows_b.append(f"{SHORT[fam]} & {SETS[rs]} & {lev_txt} & {SCH[sch]} & {P:.4f} & {med:.2f} & "
                  f"[{ci[0]:.2f}, {ci[1]:.2f}] \\\\")
    assert sch == "CLMU5" and rs in ("core16", "core15")

# v26 (user): one-line caption; parts (a) and (b) are described in the
# Supplement text (Sect. S5) where the table is cited.
# v27 (supervisor A-1): the composition table moves to the main text (one
# table + one figure summarize the scenarios); the six exception cells stay
# in the Supplement.
cap_main = (r"\caption{Observation-error scenarios applied to the model metrics: composition "
            r"and the cells whose slope interval reaches unity.}")
out_main = [r"\begin{table}[!tbp]", r"\centering\footnotesize", cap_main, r"\label{tab:scenarios}",
            r"\setlength{\tabcolsep}{3pt}", r"\scriptsize",
            r"\begin{tabular}{@{}p{0.50\linewidth}lrrrrr@{}}", r"\toprule",
            r"Error family and levels & Design, draws & Levels & Sets & Cells & max $P$ & CI $\ni 1$ \\", r"\midrule",
            *rows_a, tot, r"\bottomrule", r"\end{tabular}",
            r"\par\vspace{2pt}{\scriptsize\raggedright Cells = levels $\times$ record sets $\times$ two schemes; "
            r"max $P$ = largest bootstrap frequency $P(\hat\beta_{m|o}\ge1)$ in the family; CI $\ni 1$ = cells whose "
            r"95\% interval includes unity (listed in Supplementary Table~S2).\par}",
            r"\end{table}"]
p_main = ROOT / "paper/manuscript_dtsa/tables/table6_scenarios.tex"
io.open(p_main, "w", encoding="utf-8", newline="\n").write("\n".join(out_main) + "\n")

cap = (r"\caption{The six scenario cells whose 95\% interval of the observed-aligned "
       r"slope includes unity.}")
out = [r"\begin{table}[!tbp]", r"\centering\footnotesize", cap, r"\label{tab:cells}",
       r"\setlength{\tabcolsep}{3pt}", r"\scriptsize",
       r"\begin{tabular}{@{}lllcrrc@{}}", r"\toprule",
       r"Error family & Record set & Level & Scheme & $P(\hat\beta\ge1)$ & median $\hat\beta$ & 95\% CI \\", r"\midrule",
       *rows_b, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
p = ROOT / "paper/manuscript_dtsa/tables/tableS2_scenario_cells.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"wrote {p_main} and {p}")
    for r in rows_a + [tot] + rows_b: print("  ", r)
