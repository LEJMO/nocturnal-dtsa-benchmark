# -*- coding: utf-8 -*-
"""Emit Table A.1 (tab:benchproto): benchmark win counts under four fitting
protocols and four record sets.

v15: the sensitivity of the benchmark counts to the holdout unit and to the
composition of the training pool was described in prose only; the reviewer
asked for the already-computed counts to be tabulated. Sources (all inside
results/paper_stats_v1.json):
  (a) leave-one-record-out as first published   -> benchmark_city_holdout.city_holdout.delta_vs_published_record_loo
  (b) city holdout, benchmarks fitted on all 16 clusters, counts restricted -> ...win_counts_eval_only
  (c) city holdout, refitted inside the record set -> ...win_counts_refit_within_variant
  (d) city holdout, duplicated US-Minneapolis2 removed from every training pool
      -> benchmark_holdout_adversarial_verification.D_training_pool_deduplicated
Layout: protocols as row blocks, record sets as columns (fits the text width).
"""
import io, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S = json.load(open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
A = S["review5_additions"]
CH = A["benchmark_city_holdout"]["city_holdout"]
D = A["benchmark_holdout_adversarial_verification"]["D_training_pool_deduplicated"]

VAR = ["all19", "excl_mpls17", "core16", "core15"]
SCH = [("TEB", "TEB"), ("CLM-Urban", "CLMU5")]

def counts(proto, k, vk):
    if proto == "a":
        c = CH["delta_vs_published_record_loo"][vk][k]["published_record_loo"]; both = "beats_meaningful_two_REG2_KM3"
    elif proto == "b":
        c = CH["win_counts_eval_only"][k][vk]; both = "beats_meaningful_two_REG2_KM3"
    elif proto == "c":
        c = CH["win_counts_refit_within_variant"][k][vk]; both = "beats_meaningful_two_REG2_KM3"
    else:
        c = D[k]["counts"][vk]; both = "beats_both_informative"
    return c[both], c["beats_REG2"], c["beats_KM3"]

# consistency with the prose of Sect. 3.3
assert counts("b", "TEB", "all19")[0] == 3 and counts("b", "CLMU5", "all19")[0] == 13
assert counts("d", "TEB", "all19")[0] == 4 and counts("d", "CLMU5", "all19")[0] == 14
assert counts("d", "TEB", "core15")[0] == 2 and counts("d", "CLMU5", "core15")[0] == 10
assert counts("a", "TEB", "all19")[2] == 8 and counts("b", "TEB", "all19")[2] == 10

PROTO = [("a", "(a)"), ("b", "(b)"), ("c", "(c)"), ("d", "(d)")]
rows = []
for p, lab in PROTO:
    for i, (disp, k) in enumerate(SCH):
        cells = " & ".join("{} ({}/{})".format(*counts(p, k, vk)) for vk in VAR)
        rows.append(f"{lab if i == 0 else ''} & {disp} & {cells} \\\\")
    rows.append(r"\addlinespace")
rows.pop()

def hdr(a, b): return r"\multicolumn{1}{c}{\shortstack[c]{" + a + r"\\" + b + "}}"

# v26 (user): one-line caption; the entry format and protocols (a)-(d) are
# defined in the Supplement text (Sect. S3) where the table is cited. A
# one-line key keeps the entries readable at the table.
cap = (r"\caption{Benchmark win counts under four fitting protocols and four "
       r"record sets.}")
NOTE = (r"\par\vspace{2pt}{\scriptsize\raggedright Entries: records won against "
        r"both informative benchmarks (against REG2/KM3 separately).\par}")

out = [r"\begin{table}[p]", r"\centering\footnotesize", cap, r"\label{tab:benchproto}",
       r"\setlength{\tabcolsep}{3pt}",
       r"\begingroup\scriptsize",
       r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
       r"Protocol & Scheme & " + hdr("All 19", "records") + " & "
       + hdr("Excl.\\ Minneapolis", "(17)") + " & "
       + hdr("Conservative", "core (16)") + " & "
       + hdr("Core excl.", "Narutowicza (15)") + r" \\",
       r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\endgroup", NOTE, r"\end{table}"]

p = ROOT / "paper/manuscript_dtsa/tables/tableA1_benchmark_protocols.tex"
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"wrote {p}")
    for r in rows: print("  ", r)
