# -*- coding: utf-8 -*-
"""Regenerate the model-comparison table (printed Table 3, label tab:modelvar)
from the MERGED statistics package.

Differences from the v10 generator it replaces:
  * CLM-Urban values come from the alignment-corrected run.
  * Confidence intervals come from review5_additions.scheme_ordering_paired
    (100,000 city-cluster draws) rather than the 5,000-draw block, which was
    too few to order two cells stably.
  * The table now carries the PAIRED scheme difference with its own interval,
    because the manuscript's former "robust gap in ordering" claim needed one
    and the interval turns out to include zero in every record set.

Do NOT re-run v10_tables.py after the review-5 merge: it would resurrect the
superseded obs_uncertainty_sensitivity block.
"""
import io, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sp = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
AOD = sp["amplitude_ordering_decomposition"]
SOP = sp["review5_additions"]["scheme_ordering_paired"]


def _bound(x, draws=100_000):
    """Smallest 1-significant-figure-at-the-third-decimal value that is a TRUE
    upper bound on x, with a 3-sigma Monte Carlo margin. Printing round(x,3)
    would state a FALSE bound whenever x rounds down (0.00131 -> 0.001)."""
    import math
    se = math.sqrt(max(x, 1e-12) * (1 - max(x, 1e-12)) / draws)
    target = x + 3 * se
    for cand in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        if cand >= target:
            return f"{cand:g}".replace("0.", "0.")
    return f"{target:.3f}"

# v27 (supervisor, A-5): the conservative core is the headline record set and
# leads the table; the two larger sets follow as sensitivity.
VAR = [("Conservative core (16)", "core16"),
       ("Excl.\\ Minneapolis (17)", "excl_mpls17"),
       ("All 19 records", "all19")]
NAME = [("TEB", "TEB"), ("CLM-Urban", "CLMU5")]

rows, dlines = [], []
for lab, vk in VAR:
    for i, (disp, key) in enumerate(NAME):
        a = AOD[vk][key]
        s = SOP[vk][key]
        lo, hi = s["beta_ci95"]
        rows.append(f"{lab if i == 0 else ''} & {disp} & {a['sd_ratio']:.2f} & "
                    f"{a['r']:+.2f} & {a['rho']:+.2f} & "
                    f"{a['slope']:.2f} [{lo:.2f}, {hi:.2f}] \\\\")
    d = SOP[vk]["paired_difference_TEB_minus_CLMU5"]["r"]
    dlines.append(rf"{d['point']:+.2f}\,[{d['ci95'][0]:.2f},{d['ci95'][1]:.2f}]")

pb = min(SOP[vk][k]["P_beta_ge_1"] if False else SOP[vk]["one_sided"][f"{k}_P_beta_ge_1"]
         for vk in [v[1] for v in VAR] for k in ("TEB", "CLMU5"))
pbmax = max(SOP[vk]["one_sided"][f"{k}_P_beta_ge_1"]
            for vk in [v[1] for v in VAR] for k in ("TEB", "CLMU5"))

# v26 (user): one-line caption. The paired-difference intervals that this
# caption used to carry are now printed in the text of Sect. 3.2; verify that
# the manuscript still states exactly the values in the statistics package.
cap = (r"\caption{Model reproduction of the observed cross-site gradient in "
       r"three record sets.}")
_main_path = ROOT / "paper/manuscript_dtsa/main.tex"   # release: manuscript source not shipped
_main = io.open(_main_path, encoding="utf-8").read() if _main_path.exists() else None
for _, vk in VAR:
    d = SOP[vk]["paired_difference_TEB_minus_CLMU5"]["r"]
    tex = f"${d['point']:+.2f}$ $[{d['ci95'][0]:.2f},{d['ci95'][1]:.2f}]$"
    assert _main is None or tex in _main, ("paired difference missing from main.tex", vk, tex)

out = [r"\begin{table}[!tbp]", r"\centering\small", cap, r"\label{tab:modelvar}",   # v20: [t]-only floats blocked the queue
       r"\begin{tabular}{llcccc}", r"\toprule",
       r"Record set & Scheme & $s_m/s_o$ & $r$ & $\rho$ & "
       r"$\hat\beta_{m|o}$ [95\% CI] \\", r"\midrule"]
for i, r in enumerate(rows):
    if i and i % 2 == 0:
        out.append(r"\addlinespace")
    out.append(r)
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

p = ROOT / "paper/manuscript_dtsa/tables/table2_model_variants.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(f"wrote {p}")
print(f"  P(beta>=1) max across 6 cells = {pbmax:.4f}  (min {pb:.4f})")
for r in rows:
    print("  ", r)
