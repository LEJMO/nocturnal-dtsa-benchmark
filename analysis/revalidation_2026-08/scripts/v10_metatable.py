# -*- coding: utf-8 -*-
"""Emit Table 3 (tab:geom): station geometry and radiometer siting per record,
ordered as Table 1 (by observed offset).

Columns: H (mean building height), z_Ta (air-temperature measurement height),
z_Ta/H, z_rad (radiometer height, printed only where the observation-file
metadata documents it: the Minneapolis pair at 2 m; "n.d." elsewhere),
z_rad/H, roof plan fraction (site data csv). H, z_Ta and z/H come from the
harmonized Urban-PLUMBER site metadata (Lipson 2022) via
results/paper_stats_v1.json (obs_metadata).

History: the v12 manuscript added the z_rad columns by hand; this generator
was brought up to date in v26 so that re-running it reproduces the committed
table (body verified identical by git diff) instead of reverting it.
v26 (user): the caption is a one-line title; column definitions and the two
siting cautions are in the text of Sect. 2.1 (label sec:sites); a symbol key
stays under the table."""
import io, sys, json, csv
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
sp = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
d = sp["per_site"]; meta = sp["obs_metadata"]

def sdd(site, key):
    p = ROOT / f"data/urban-plumber/FullCollection/{site}/{site}_sitedata_v1.csv"
    if not p.exists(): return None
    for r in csv.DictReader(p.open(encoding="utf-8")):
        if r["parameter"] == key:
            try: return float(r["value"])
            except ValueError: return None
    return None

# radiometer height documented in the observation-file metadata (Sect. 2.1)
RAD_DOC = {"US-Minneapolis1": 2.0, "US-Minneapolis2": 2.0}
FLAG = {"US-Minneapolis1": r"$\dagger$", "US-Minneapolis2": r"$\dagger$",
        "PL-Lipowa": r"$\ddagger$", "FR-Capitole": r"$\S$"}

order = sorted(d, key=lambda s: d[s]["dtsa_std"])   # same order as Table 1
rows = []
for s in order:
    H = meta[s]["H"]; z = meta[s]["z_Ta"]; zh = meta[s]["z_over_H"]
    roof = sdd(s, "roof_area_fraction")
    assert H and zh and roof is not None, s
    if s in RAD_DOC:
        assert abs(meta[s]["rad_height"] - RAD_DOC[s]) < 1e-9, s
        zr, zrh = f"{RAD_DOC[s]:.1f}", f"{RAD_DOC[s] / H:.2f}"
    else:
        zr = zrh = "n.d."
    rows.append(f"{s.replace('_', chr(92) + '_')}{FLAG.get(s, '')} & {H:.1f} & {z:.1f} & "
                f"{zh:.1f} & {zr} & {zrh} & {roof:.2f} \\\\")

cap = r"\caption{Station geometry and radiometer siting for the 19 records.}"
NOTE = (r"\par\vspace{2pt}{\scriptsize\raggedright Flags as in Table~\ref{tab:sites}; "
        r"n.d.\ = not documented (Sect.~\ref{sec:sites}).\par}")
out = [r"\begin{table}[!tbp]", r"\centering\footnotesize",   # v20: [t]-only floats blocked the queue
       cap, r"\label{tab:geom}", r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{lcccccc}", r"\toprule",
       r"Site & $H$ (m) & $z_{T_a}$ (m) & $z_{T_a}/H$ & $z_{\mathrm{rad}}$ (m) & "
       r"$z_{\mathrm{rad}}/H$ & Roof frac. \\", r"\midrule"]
out += rows
out += [r"\bottomrule", r"\end{tabular}", NOTE, r"\end{table}"]
p = ROOT / "paper/manuscript_dtsa/tables/table3_geometry.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(f"wrote {p} ({len(rows)} rows)")
for r in rows: print("  ", r)
