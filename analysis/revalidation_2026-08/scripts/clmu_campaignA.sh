#!/usr/bin/env bash
# CLMU5 Campaign A (albedo-constrained): for each of 19 records, build baseline
# inputs, override ALL urban albedo fields (ROOF/IMPROAD/PERROAD/WALL x DIR/DIF,
# both numrad bands) to the site's observed midday albedo, run, save history.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # release: repository root
W=/root/clmu_work
PY=$W/venv/bin/python
OUT=$ROOT/external/clmu_albedo19
mkdir -p "$OUT"
LOG=$OUT/_run.log; : > "$LOG"
docker ps >/dev/null 2>&1 || { nohup dockerd >/var/log/dockerd.log 2>&1 & sleep 8; }
SITES=$("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ROOT/results/paper_stats_v1.json'))['per_site'])))")

for s in $SITES; do
  alb=$("$PY" -c "import json;print(json.load(open('$ROOT/results/paper_stats_v1.json'))['per_site']['$s']['albedo'])")
  "$PY" "$W/clmu_build.py" "$s" >/dev/null 2>&1
  # override albedo in the built surfdata (UT=2 urban class, all numrad bands)
  "$PY" - "$W/inputs/$s/surfdata.nc" "$alb" <<'PY'
import sys, xarray as xr, numpy as np
ds=xr.open_dataset(sys.argv[1]); a=float(sys.argv[2]); UT=2
for v in ["ALB_ROOF_DIR","ALB_ROOF_DIF","ALB_IMPROAD_DIR","ALB_IMPROAD_DIF",
          "ALB_PERROAD_DIR","ALB_PERROAD_DIF","ALB_WALL_DIR","ALB_WALL_DIF"]:
    ds[v].values[:,UT,0,0]=a
tmp=sys.argv[1]+".tmp"; ds.to_netcdf(tmp); ds.close()
import os; os.replace(tmp, sys.argv[1])
PY
  rm -rf "$W/runs/$s/scriptsfolder/g1_"* "$W/runs/$s/logfolder/g1_"* "$W/runs/$s/inputfolder/usp" 2>/dev/null
  out=$(timeout 3000 "$PY" "$W/clmu_run.py" "$s" 2>&1 | grep -oE '/root/clmu_work/runs/[^ ]+clm.nc' | tail -1)
  if [ -n "$out" ] && [ -f "$out" ]; then
    cp "$out" "$OUT/${s}_alb.nc"; echo "$s OK alb=$alb" >>"$LOG"
  else
    echo "$s FAILED" >>"$LOG"
  fi
done
echo "DONE $(date -u +%H:%M:%S)" >>"$LOG"
touch "$OUT/RUN_COMPLETE.flag"
