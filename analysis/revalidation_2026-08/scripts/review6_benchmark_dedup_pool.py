# -*- coding: utf-8 -*-
"""Residual-leak test the audit did not run: leave-one-city-cluster-out is
correct at the HOLDOUT, but every other fold's TRAINING pool still contains the
duplicated Minneapolis observation TWICE (2 x 19,723 = 12.2% of the 323,587-step
pooled nocturnal sample is one physical record counted twice). Refit the city
folds with the training pool DE-DUPLICATED (US-Minneapolis2 never trains), while
still evaluating all 19 records.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import io, sys, json, time
from pathlib import Path
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(os.environ["SCR"])))
import adv_cityholdout as A   # reuse my own frame builders + fold machinery

t0 = time.time()
SITES = A.SITES
frames = {"TEB": {s: A.frame_teb(s) for s in SITES},
          "CLMU5": {s: A.frame_clmu(s) for s in SITES}}
clmap = {}
for s in SITES:
    clmap.setdefault(A.P[s]["cluster"], []).append(s)
CITY = [sorted(v) for k, v in sorted(clmap.items())]
no_m = [s for s in SITES if not s.startswith("US-Minneapolis")]
SETS = {"all19": SITES, "excl_mpls17": no_m,
        "core16": [s for s in no_m if s != "PL-Lipowa"],
        "core15": [s for s in no_m if s not in ("PL-Lipowa", "PL-Narutowicza")]}
POOL_DEDUP = [s for s in SITES if s != "US-Minneapolis2"]
R = {"note": ("city folds, training pool de-duplicated (US-Minneapolis2 removed "
              "from every training pool; all 19 records still evaluated)"),
     "train_pool_size": len(POOL_DEDUP)}
for sch in ("TEB", "CLMU5"):
    bm, mt = A.bench_folds(frames[sch], CITY, POOL_DEDUP, 0)
    cts = {k: A.counts(frames[sch], bm, ss)[0] for k, ss in SETS.items()}
    R[sch] = {"per_record_mae": bm, "counts": cts,
              "n_train_steps": {k: v["n_train_steps"] for k, v in mt.items()}}
    print("[%7.1fs] dedup-pool %s %s" % (time.time() - t0, sch, cts), flush=True)
out = Path(os.environ["SCR"]) / "adv_dedup_pool.json"
json.dump(R, io.open(out, "w", encoding="utf-8"), indent=1, default=float)
print("wrote", out)
