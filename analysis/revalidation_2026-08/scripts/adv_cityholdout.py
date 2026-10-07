"""Release alias (new file, no computation).

The adversarial benchmark verification was first written as a scratch-directory
module named adv_cityholdout.py; its consolidated copy in this folder is
review6_benchmark_holdout_verify.py. The scripts review6_benchmark_dedup_pool.py,
review6_benchmark_dedup_control.py and review6_benchmark_seed_fragility.py import
the frame builders and fold machinery under the old name, so this file re-exports
them. Set the environment variable SCR to a writable scratch directory before
running those scripts (they write their intermediate JSON there).
"""
from review6_benchmark_holdout_verify import (  # noqa: F401
    P, SITES, bench_folds, counts, frame_clmu, frame_teb, corpus)
