# -*- coding: utf-8 -*-
"""Merge the round-5 (v11) statistics packages into results/paper_stats_v1.json.

Round 5 was computed by several independent generators writing to their own
files, deliberately, so that concurrent runs could not clobber the single source
of truth. This script performs the one serialised merge.

Superseded keys are NEVER deleted: each is moved under a top-level `superseded`
block with the key path, the value, and the section that replaces it, so the
provenance of every previously published number survives.

Run AFTER the paper_stats.py -> attribution_covariates -> review2..4 ->
v10_tables chain, because paper_stats.py rewrites the whole file.

Usage: review5_merge.py [--dry-run]
"""
import io, sys, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PS = ROOT / "results/paper_stats_v1.json"
DRY = "--dry-run" in sys.argv

# source file -> (target path under paper_stats, list of key paths it supersedes)
SOURCES = [
    ("results/review5_adderley-propagation.json",
     "review5_additions.obs_uncertainty_v11",
     ["obs_uncertainty_sensitivity"]),
    ("results/review5_temporal-seasonal.json",
     "review5_additions.temporal_sampling",
     ["elimination.djf_exclusion"]),
    ("results/review5_permutation-rebuild.json",
     "review5_additions.permutation_v11",
     ["cluster_permutation.all", "cluster_permutation.excl_mpls",
      "review4_additions.albedo_given_skyemissivity_freedman_lane"]),
    ("results/clmu_v11_rebuild.json",
     "review5_additions.clmu_alignment_v11", []),
    ("results/geometry_v11.json",
     "review5_additions.station_geometry_v11", []),
    ("results/teb_persite_warming.json",
     "review5_additions.teb_persite_warming_v11", []),
    ("results/review5_scheme_ordering_paired.json",
     "review5_additions.scheme_ordering_paired", []),
    ("results/review5_benchmark_skill.json",
     "review5_additions.nocturnal_benchmark_skill", []),
    ("results/review5_benchmark_decomposition.json",
     "review5_additions.benchmark_error_decomposition", []),
    ("results/review5_core15.json",
     "review5_additions.core15_sensitivity", []),
    ("results/review5_benchmark_cityholdout.json",
     "review5_additions.benchmark_city_holdout", []),
    ("results/review5_run_provenance.json",
     "review5_additions.run_provenance", []),
    ("results/review5_obs_error_into_model.json",
     "review5_additions.obs_error_propagated_to_model", []),
    ("results/review6_adversarial_mc_bounds.json",
     "review5_additions.bootstrap_mc_bounds", []),
    ("results/review6_text_number_audit.json",
     "review5_additions.text_number_audit", []),
    # v15: the de-duplicated-training-pool and k-means-seed counts printed in
    # the benchmark paragraph come from this file; it must live in the single
    # source of truth like every other printed number.
    ("results/review6_benchmark_holdout_verification.json",
     "review5_additions.benchmark_holdout_adversarial_verification", []),
    # v16: exact 'own mean bias removed' counts (the centred-RMSE proxy printed
    # in v15 was only a lower bound).
    ("results/review7_debiased_mae.json",
     "review5_additions.benchmark_debiased_mae", []),
    # v27 (supervisor review 2026-09-23): core-16 metric families, the
    # albedo/sky-emissivity and Cfb-only checks (B-1) and the material
    # sensitivity in kelvin (E/F-3); read-only derivations from prior results.
    ("results/review8_core16_and_b1.json",
     "review8_additions", []),
]
# sections that replace a whole top-level block rather than nesting under review5
TOPLEVEL = [("results/review5_constrained.json",
             ["teb_constrained_experiments", "clmu_constrained_experiments",
              "amplitude_ordering_decomposition"])]


def leaves(o):
    if isinstance(o, dict): return sum(leaves(v) for v in o.values())
    if isinstance(o, list): return sum(leaves(v) for v in o)
    return 1


def get_path(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur: return None, False
        cur = cur[part]
    return cur, True


def pop_path(d, path):
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur: return None, False
        cur = cur[part]
    if not isinstance(cur, dict) or parts[-1] not in cur: return None, False
    return cur.pop(parts[-1]), True


def set_path(d, path, val):
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = val


def main():
    sp = json.load(open(PS, encoding="utf-8"))
    before = leaves(sp)
    print(f"paper_stats_v1.json: {before} leaves before merge")
    sup = sp.setdefault("superseded", {})
    merged, missing = [], []

    for rel, target, supersedes in SOURCES:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel); print(f"  [MISSING] {rel}"); continue
        blk = json.load(open(p, encoding="utf-8"))
        blk.pop("_merge_target", None); blk.pop("_supersedes", None)
        blk["_source_file"] = rel
        set_path(sp, target, blk)
        print(f"  [merge] {rel}  ->  {target}  ({leaves(blk)} leaves)")
        for kp in supersedes:
            val, ok = pop_path(sp, kp)
            if ok:
                sup[kp] = {"value": val, "replaced_by": target}
                print(f"      superseded {kp}  -> superseded.{kp}")
            else:
                print(f"      (nothing at {kp} to supersede)")
        merged.append(target)

    for rel, targets in TOPLEVEL:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel); print(f"  [MISSING] {rel}"); continue
        blk = json.load(open(p, encoding="utf-8"))
        for t in targets:
            if t not in blk:
                print(f"  [WARN] {rel} has no '{t}'"); continue
            old, ok = get_path(sp, t)
            if ok:
                sup[t] = {"value": old, "replaced_by": f"{t} (regenerated, {rel})"}
            sp[t] = blk[t]
            print(f"  [merge] {rel}:{t}  ->  {t}  ({leaves(blk[t])} leaves)")
            merged.append(t)
        if "_meta" in blk:
            set_path(sp, "review5_additions.constrained_experiments_meta", blk["_meta"])

    sp.setdefault("meta", {})["review5_merge"] = {
        "script": "analysis/revalidation_2026-08/scripts/review5_merge.py",
        "date": "2026-09-16",
        "merged_targets": merged,
        "missing_sources": missing,
        "note": ("round-5 packages were generated into separate files to avoid "
                 "concurrent-write loss; superseded keys are preserved under the "
                 "top-level 'superseded' block, never deleted."),
    }

    after = leaves(sp)
    print(f"\npaper_stats_v1.json: {after} leaves after merge (delta {after-before:+d})")
    if missing:
        print(f"MISSING SOURCES ({len(missing)}): {missing}")
    if DRY:
        print("\n--dry-run: nothing written"); return 0
    bak = PS.with_suffix(".json.pre_review5")
    if not bak.exists():
        shutil.copy2(PS, bak); print(f"backed up -> {bak.name}")
    with io.open(PS, "w", encoding="utf-8", newline="\n") as f:
        json.dump(sp, f, indent=2, ensure_ascii=False, default=float)
    chk = json.load(open(PS, encoding="utf-8"))
    print(f"re-read OK: {leaves(chk)} leaves, review5_additions="
          f"{sorted(chk.get('review5_additions', {}))}")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main())
