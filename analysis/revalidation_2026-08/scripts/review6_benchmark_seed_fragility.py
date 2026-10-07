# -*- coding: utf-8 -*-
"""Targeted: KM3 MAE at the Minneapolis records under RECORD-LOO for several
k-means seeds, to test whether the published count 8 is a seed artifact."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import io, sys, json, time
from pathlib import Path
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(os.environ["SCR"])))
import adv_cityholdout as A

SEEDS = [0, 1, 2, 3, 7, 42, 777, 12345]
t0 = time.time()
R = {}
for sch in ("TEB",):
    frames = {s: (A.frame_teb(s) if sch == "TEB" else A.frame_clmu(s)) for s in A.SITES}
    R[sch] = {}
    for s in ("US-Minneapolis1", "US-Minneapolis2"):
        mae = float(np.mean(np.abs(frames[s]["model"] - frames[s]["obs"])))
        R[sch][s] = {"scheme_MAE": mae, "by_seed": {}}
    for sd in SEEDS:
        # single record-LOO fold per Minneapolis record
        for s in ("US-Minneapolis1", "US-Minneapolis2"):
            b, _ = A.bench_folds(frames, [[s]], A.SITES, sd)
            R[sch][s]["by_seed"][str(sd)] = {
                "KM3": b[s]["KM3"], "REG2": b[s]["REG2"],
                "beats_KM3": bool(R[sch][s]["scheme_MAE"] < b[s]["KM3"]),
                "margin": R[sch][s]["scheme_MAE"] - b[s]["KM3"]}
            print("[%6.1fs] %s %s seed=%s KM3=%.4f scheme=%.4f margin=%+.4f beats=%s"
                  % (time.time() - t0, sch, s, sd, b[s]["KM3"],
                     R[sch][s]["scheme_MAE"],
                     R[sch][s]["by_seed"][str(sd)]["margin"],
                     R[sch][s]["by_seed"][str(sd)]["beats_KM3"]), flush=True)
out = Path(os.environ["SCR"]) / "adv_seed_margin.json"
json.dump(R, io.open(out, "w", encoding="utf-8"), indent=1, default=float)
print("wrote", out)
