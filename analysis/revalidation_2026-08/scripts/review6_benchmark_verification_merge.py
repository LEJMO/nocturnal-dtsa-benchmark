# -*- coding: utf-8 -*-
"""Consolidate the adversarial verification of review5_benchmark_cityholdout
into one companion package under results/."""
import io, json, os, re, ast
from pathlib import Path
S = Path(os.environ["SCR"])
ROOT = Path(__file__).resolve().parents[3]


def load(name):
    p = S / name
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.load(io.open(p, encoding="utf-8"))


# seed sweep count table, parsed from the independent run's log
seed_counts = {}
for line in io.open(S / "adv.log", encoding="utf-8").read().splitlines():
    m = re.match(r"\[\s*([\d.]+)s\] (\S+) seed=(\d+) recLOO=(.*) cityHO=(.*)$", line)
    if m:
        seed_counts.setdefault(m.group(2), {})[m.group(3)] = {
            "record_loo": ast.literal_eval(m.group(4)),
            "city_holdout": ast.literal_eval(m.group(5))}

PS = json.load(io.open(ROOT / "results/paper_stats_v1.json", encoding="utf-8"))
pub = PS["review5_additions"]["nocturnal_benchmark_skill"]["benchmark"]

out = {
    "_merge_target": "review6_additions.benchmark_holdout_adversarial_verification",
    "meta": {
        "purpose": ("independent adversarial re-verification of "
                    "review5_benchmark_cityholdout (leave-one-city-cluster-out "
                    "benchmark refit). Frames and fold machinery re-implemented "
                    "from the raw files by a separate script (only "
                    "clmu_post_v11.align reused); own k-means seeds; plus two "
                    "sensitivities the original package did not run."),
        "verdict_on_original": ("every eval-only win count and every duplication "
                                "figure reproduced exactly; the original generator "
                                "is bit-for-bit reproducible (0 differences over "
                                "2763 leaves, wall-clock fields excluded). Two new "
                                "findings qualify its conclusions: the single count "
                                "it changes is k-means-seed-dependent, and its "
                                "'every headline count identical' claim holds only "
                                "while the duplicated Minneapolis observation stays "
                                "double-weighted in the benchmark training pool."),
        "constants": {"SIG": 5.67e-8, "EPS": 0.95,
                      "kmeans": {"k": 27, "n_init": 10, "max_iter": 300}},
        "generators": {
            "adv_cityholdout.py": "independent frames + record/city folds, seeds 0/12345/777",
            "adv_seed_margin.py": "record-LOO KM3 MAE at the Minneapolis records, 8 seeds",
            "adv_dedup_pool.py": "city folds with the training pool de-duplicated",
            "adv_dedup_control.py": "matched single-record training-pool removals (control)",
            "verify_dup.py": "raw + corpus duplication audit, three city pairs"},
        "paper_stats_v1_untouched": True,
        "manuscript_untouched": True,
    },
    "published_reference": {
        "record_loo_win_counts": pub["win_counts"],
        "minneapolis_per_record_mae_bit_identical": {
            sch: {s: {k: pub["per_record_mae"][sch][s][k]
                      for k in ("scheme_MAE", "REG1_MAE", "REG2_MAE", "KM3_MAE")}
                  for s in ("US-Minneapolis1", "US-Minneapolis2")}
            for sch in ("TEB", "CLMU5")},
    },
    "A_duplication_audit_independent": load("verify_dup.json"),
    "B_counts_by_seed_and_protocol": {
        "note": ("independent re-implementation. record_loo at seed 0 reproduces the "
                 "published win_counts exactly for both schemes; city_holdout "
                 "reproduces review5_benchmark_cityholdout exactly in every cell."),
        "by_scheme_seed": seed_counts},
    "C_kmeans_seed_fragility_at_minneapolis": load("adv_seed_margin.json"),
    "D_training_pool_deduplicated": load("adv_dedup_pool.json"),
    "E_training_pool_single_record_controls": load("adv_dedup_control.json"),
}
p = ROOT / "results/review6_benchmark_holdout_verification.json"
json.dump(out, io.open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False,
          default=float)


def leaves(o):
    if isinstance(o, dict):
        return sum(leaves(v) for v in o.values())
    if isinstance(o, list):
        return sum(leaves(v) for v in o)
    return 1


print("wrote %s" % p)
print("  %d leaves, %d bytes" % (leaves(out), p.stat().st_size))
for k, v in out.items():
    if k.startswith("_") or k == "meta":
        continue
    print("  %-46s %s" % (k, "present" if v is not None else "MISSING"))
