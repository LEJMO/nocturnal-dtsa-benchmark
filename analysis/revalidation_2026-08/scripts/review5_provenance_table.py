# -*- coding: utf-8 -*-
"""Emit Table 4 (tab:prov): per-scheme run provenance.

Answers the reviewer's question of WHAT is being compared: the output variable,
the aggregation level, the area weighting, the vegetation/non-urban treatment,
the air-temperature reference, the alignment convention and the mask. Every cell
is read from results/review5_run_provenance.json, which was built from the run
output and run configuration rather than from recollection.
"""
import io, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
P = json.load(open(ROOT / "results/review5_run_provenance.json", encoding="utf-8"))
CL, TB = P["clmu"], P["teb"]

# --- CLM-Urban facts, aggregated over the 19 records ---
cps = CL["per_site"]
nonurb = {v["landunit_nonurban_wtgcell"] for v in cps.values()}
urb = {v["landunit_urban_wtgcell"] for v in cps.values()}
ncol = {v["n_urban_columns"] for v in cps.values()}
wsum = {round(v["col_wtgcell_urban_sum"], 12) for v in cps.values()}
ident = max(v["third_identity_maxerr"] for v in cps.values())
offs = {v["align_offset"] for v in cps.values()}
dta = max(abs(v["max_abs_Tair_minus_corpus_forcing_Tair"]) for v in cps.values())
dld = max(abs(v["max_abs_FLDS_minus_corpus_forcing_LWdown"]) for v in cps.values())
roofs = [v["col_wtlunit"]["roof"] for v in cps.values()]

# --- TEB facts ---
tps = TB["output_variable"]
rows_ok = all(v["rows_equal_corpus_minus_one"] for v in tps.values())
gf = TB["garden_fraction"]
zg = [v["ZGARDEN"] for v in gf.values()]
gdev = max(abs(v["deviation"]) for v in gf.values())
emis_err = max(v["abs_error"] for v in TB["aggregation_identity"].values())
ta = P["air_temperature_reference"]
dta_teb = max(abs(v["teb_forcTA_vs_corpus_forcing_Tair_maxabs"]) for v in ta["per_site"].values())

rows = [
 ("Output variable",
  r"\texttt{FIRE}, total upwelling longwave",
  r"\texttt{LWU} column of the driver output"),
 ("Aggregation level",
  r"urban landunit; the gridcell is 100\% urban at every record "
  r"(non-urban weight " + f"{min(nonurb):.1f}" + r", urban " + f"{min(urb):.1f}" + r"), "
  r"so the landunit value and the gridcell mean coincide",
  r"town tile (built + garden), area-weighted inside the driver"),
 ("Facet weighting",
  f"{min(ncol)} urban columns (roof, sunwall, shadewall, impervious road, "
  r"pervious road) with per-record weights summing to "
  + f"{min(wsum):.0f}" + r"; the column-to-gridcell identity holds to "
  + f"{ident:.0e}" + r". Roof weight spans "
  + f"{min(roofs):.2f}--{max(roofs):.2f}",
  r"roof, wall and road facets with canyon view factors; the diagnosed town "
  r"emissivity reproduces the model's own value to "
  + f"{emis_err:.0e}"),
 ("Vegetation and non-urban",
  r"no non-urban vegetation or lake landunit exists in these runs; the "
  r"vegetated fraction of the footprint is carried by the urban pervious "
  r"road column, which stands in for lawns and parks without simulating "
  r"vegetation explicitly",
  r"garden tile, fraction "
  + f"{min(zg):.2f}--{max(zg):.2f}" + r" of the town, set from the metadata "
  r"tree, grass, bare-soil and water fractions (deviation "
  + f"{gdev:.0e}" + r")"),
 (r"$T_a$ reference for $\Delta T_{s-a}$",
  r"corpus harmonized forcing air temperature, \emph{not} the model 2~m "
  r"diagnostic; identical to the observations' to "
  + f"{dta:.0e}" + r"~K",
  r"same corpus forcing air temperature, identical to "
  + f"{dta_teb:.1e}" + r"~K (text rounding)"),
 (r"LW$_{\downarrow}$ used in the inversion",
  r"the model's own \texttt{FLDS}, which is bit-identical to the corpus "
  r"forcing field, so the two schemes are inverted on the same downwelling flux",
  r"corpus forcing downwelling longwave"),
 ("Alignment",
  r"offset $\min(17520,n)+1$, a truncated final window allowed; exact "
  r"air-temperature echo required (Supplementary Sect.~S2)",
  r"model row $i$ against corpus step $i{+}1$; output length is $n-1$ at every "
  r"record" + ("" if rows_ok else r" (NOT verified)")),
 ("Nocturnal mask",
  r"night and not pre-spin-up, intersected with finite observed and modelled "
  r"LW$_{\uparrow}$",
  r"the same mask, additionally requiring finite forcing air temperature and "
  r"downwelling longwave"),
]

# v26 (user): one-line caption; the provenance and the aggregation point are
# stated in the text of Sect. 2.5 where the table is cited.
cap = (r"\caption{Run provenance for the two schemes, read from the run output "
       r"and configuration.}")

out = [r"\begin{table}[!tbp]", r"\centering\footnotesize", cap, r"\label{tab:prov}",   # v20: [p]-only never filled a float page and blocked later tables
       r"\setlength{\tabcolsep}{3pt}",
       r"\renewcommand{\arraystretch}{1.0}",
       # v15: the v14 float was 35 pt too tall for the float page ("Float too
       # large for page" in the log) and the folio landed inside the last row.
       # The tabular is set in \scriptsize; column widths sum to 0.95\linewidth
       # so that with 3 pt padding the tabular stays inside the text block.
       r"\begingroup\scriptsize",
       r"\begin{tabular}{@{}p{0.18\linewidth}p{0.41\linewidth}p{0.36\linewidth}@{}}",
       r"\toprule",
       r" & CLM-Urban (CLM5.0/CLMU) & TEB \\", r"\midrule"]
for label, a, b in rows:
    out.append(f"{label} & {a} & {b} " + r"\\")
out += [r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{table}"]

p = ROOT / "paper/manuscript_dtsa/tables/table4_provenance.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"wrote {p}")
    print(f"  CLMU: nonurban weight {nonurb}, urban {urb}, columns {ncol}, "
          f"weight sum {wsum}, identity {ident:.1e}, offsets {offs}")
    print(f"  CLMU: max|Tair-corpus| {dta:.1e}, max|FLDS-corpus LWdown| {dld:.1e}")
    print(f"  TEB : rows == corpus-1 at every record: {rows_ok}, "
          f"ZGARDEN {min(zg):.2f}-{max(zg):.2f}, emissivity identity {emis_err:.1e}")
