# -*- coding: utf-8 -*-
"""Control for the de-duplicated-training-pool test: removing US-Minneapolis2
from the training pool also removes ~6% of the training steps, so the count
change could be a training-SIZE effect rather than a duplicate-specific one.
Repeat the city-fold refit dropping one ORDINARY record from the training pool
instead (three choices of comparable length), all evaluated on all 19 records.
TEB only, seed 0.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import io, sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(os.environ["SCR"])))
import adv_cityholdout as A

LOG = io.open(Path(os.environ["SCR"]) / "dedupctl_progress.log", "w",
              encoding="utf-8", buffering=1)
t0 = time.time()
SITES = A.SITES
frames = {s: A.frame_teb(s) for s in SITES}
clmap = {}
for s in SITES:
    clmap.setdefault(A.P[s]["cluster"], []).append(s)
CITY = [sorted(v) for k, v in sorted(clmap.items())]
no_m = [s for s in SITES if not s.startswith("US-Minneapolis")]
SETS = {"all19": SITES, "excl_mpls17": no_m,
        "core16": [s for s in no_m if s != "PL-Lipowa"],
        "core15": [s for s in no_m if s not in ("PL-Lipowa", "PL-Narutowicza")]}
n_steps = {s: int(frames[s]["obs"].size) for s in SITES}
LOG.write("frame sizes: %s\n" % json.dumps(n_steps))
DROPS = ["US-Minneapolis2", "US-Baltimore", "KR-Ochang", "UK-Swindon",
         "JP-Yoyogi", None]
R = {"frame_steps": n_steps, "pooled_total": int(sum(n_steps.values())),
     "variants": {}}
for drop in DROPS:
    pool = [s for s in SITES if s != drop] if drop else list(SITES)
    bm, _ = A.bench_folds(frames, CITY, pool, 0)
    cts = {k: A.counts(frames, bm, ss)[0] for k, ss in SETS.items()}
    key = drop or "none_baseline"
    R["variants"][key] = {"dropped_from_training": drop,
                          "dropped_steps": n_steps.get(drop, 0),
                          "counts": cts,
                          "minneapolis_KM3": bm["US-Minneapolis1"]["KM3"],
                          "minneapolis_REG2": bm["US-Minneapolis1"]["REG2"]}
    LOG.write("[%7.1fs] drop=%-16s steps=%6d all19 R2=%d KM3=%d both=%d | "
              "core15 R2=%d KM3=%d both=%d\n"
              % (time.time() - t0, key, n_steps.get(drop, 0),
                 cts["all19"]["beats_REG2"], cts["all19"]["beats_KM3"],
                 cts["all19"]["beats_both_informative"],
                 cts["core15"]["beats_REG2"], cts["core15"]["beats_KM3"],
                 cts["core15"]["beats_both_informative"]))
out = Path(os.environ["SCR"]) / "adv_dedup_control.json"
json.dump(R, io.open(out, "w", encoding="utf-8"), indent=1, default=float)
LOG.write("wrote %s\n" % out)
LOG.close()
