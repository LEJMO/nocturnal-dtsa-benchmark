# -*- coding: utf-8 -*-
"""Rebuild Table 1 (tab:sites) adding the temporal-coverage columns the reviewer
asked for, as a LANDSCAPE float.

WHY sidewaystable: the 12-column body measures 493.98pt against a 345pt line width, so as an upright float it ran 13.9pt past the 612pt media box and
the CLMU5 bias column was physically clipped -- 19 of 19 rows truncated, and
seven of them truncated to a syntactically valid but WRONG number (+13.1
printed as "+1", -26.4 as "-2"). Rotated, the available width is the text height
= 550pt, so it fits with ~56pt of slack at unchanged ootnotesize.

Rebuild details: analysis window start/end, calendar months sampled, and the median
monthly valid-data rate. Existing columns and ordering are preserved exactly.

Sources: results/paper_stats_v1.json (per_site) for the existing columns and
results/review5_temporal-seasonal.json (table1_new_columns) for the new ones.
"""
import io, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sp = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
tsj = ROOT / "results/review5_temporal-seasonal.json"
cov = json.load(open(tsj, encoding="utf-8"))["table1_new_columns"]
PS = sp["per_site"]
SITES = sorted(PS, key=lambda s: PS[s]["dtsa_std"])

KOPPEN = {  # as printed in the committed table
    "US-Minneapolis1": "Dfa", "US-Minneapolis2": "Dfa", "KR-Ochang": "Dwa",
    "AU-Preston": "Cfb", "GR-HECKOR": "Csa", "US-WestPhoenix": "BWh",
    "UK-Swindon": "Cfb", "FI-Kumpula": "Dfb", "UK-KingsCollege": "Cfb",
    "SG-TelokKurau06": "Af", "US-Baltimore": "Cfa", "JP-Yoyogi": "Cfa",
    "NL-Amsterdam": "Cfb", "FI-Torni": "Dfb", "FR-Capitole": "Cfb",
    "KR-Jungnang": "Dwa", "CA-Sunset": "Cfb", "PL-Narutowicza": "Dfb",
    "PL-Lipowa": "Dfb"}
FLAG = {"US-Minneapolis1": r"$^{\dagger\ast}$", "US-Minneapolis2": r"$^{\dagger\ast}$",
        "KR-Ochang": r"$^{\ast}$", "US-WestPhoenix": r"$^{\ast}$",
        "FR-Capitole": r"$^{\ast\S}$", "PL-Lipowa": r"$^{\ddagger\ast}$",
        "PL-Narutowicza": r"$^{\P}$"}

rows = []
for s in SITES:
    r = PS[s]; c = cov[s]
    rows.append(
        f"{s}{FLAG.get(s,'')} & {KOPPEN[s]} & {r['n_night']:,} & "
        f"{c['start']} & {c['end']} & {c['months_of_12']} & "
        f"{c['monthly_valid_rate_median_pct']:.0f} & "
        f"{sp['obs_metadata'][s]['z_Ta']:.1f} & {r['albedo']:.3f} & "
        f"{r['dtsa_std']:+.2f} & {r['teb_lwup_bias']:+.1f} & "
        f"{r['clmu_lwup_bias']:+.1f} \\\\")

# v26 (user): captions are one-line titles; the column definitions and the
# meaning of every flag are in the text of Sect. 2.1 (label sec:sites). A
# symbol key stays under the table so the flags can be read where they occur.
cap = (r"\caption{The 19 evaluable Urban-PLUMBER records, ordered by the "
       r"record-mean nocturnal surface--air offset.}")
NOTE = (r"\par\vspace{2pt}{\scriptsize\raggedright Flags: $\dagger$~height "
        r"mismatch; $\ddagger$~roof--canyon composite; $\ast$~diagnostic gate "
        r"record; $\S$~variable mast height; $\P$~radiometric discontinuity "
        r"(Sect.~\ref{sec:sites}).\par}")

# v21: upright again. The sidewaystable was needed in the 12 pt review layout
# (345 pt line width); in the 3p layout the 8 pt footnotesize body (~395 pt)
# fits the 6.5 in measure, and the user asked for no rotated page.
out = [r"\begin{table}[!tbp]", r"\centering\footnotesize", cap, r"\label{tab:sites}",
       r"\setlength{\tabcolsep}{3pt}",
       r"\begingroup\scriptsize",   # 8 pt body ran 26 pt past the 3p measure; 7 pt fits
       r"\begin{tabular}{lcrcccccccrr}", r"\toprule",
       r"Site & K\"{o}ppen & $n$ & Start & End & Mon. & Valid & $z_{T_a}$ (m) & "
       r"Albedo & $\overline{\Delta T_{s-a}}$ (K) & TEB bias & CLMU5 bias \\",
       r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\endgroup", NOTE, r"\end{table}"]

p = ROOT / "paper/manuscript_dtsa/tables/table1_sites.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(f"wrote {p} ({len(rows)} rows)")
for r in rows[:4]: print("  ", r)
print("  ...")
