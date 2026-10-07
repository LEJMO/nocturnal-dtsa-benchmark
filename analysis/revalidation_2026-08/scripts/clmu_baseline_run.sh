#!/usr/bin/env bash
# CLMU5 BASELINE campaign: rebuild default inputs (no albedo override) and run
# all 19 records -> native effective albedo + per-record baseline validation.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # release: repository root
W=/root/clmu_work
PY=$W/venv/bin/python
OUT=$ROOT/external/clmu_baseline19
mkdir -p "$OUT"; LOG=$OUT/_run.log; : > "$LOG"; rm -f "$OUT/RUN_COMPLETE.flag"
docker ps >/dev/null 2>&1 || { nohup dockerd >/var/log/dockerd.log 2>&1 & sleep 8; }
SITES=$("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ROOT/results/paper_stats_v1.json'))['per_site'])))")
for s in $SITES; do
  # rebuild DEFAULT inputs (build_inputs = geometry-only override; materials incl. ALB stay default)
  "$PY" "$W/clmu_build.py" "$s" >/dev/null 2>&1
  rm -rf "$W/runs_base/$s" 2>/dev/null; mkdir -p "$W/runs_base/$s"
  # run reading /root/clmu_work/inputs (default) into runs_base
  sed -e "s#$W/runs/#$W/runs_base/#g" -e "s#{WORK}/runs/#{WORK}/runs_base/#g" "$W/clmu_run.py" > "$W/clmu_run_base.py" 2>/dev/null
  # clmu_run.py uses f"{WORK}/runs/{site}"; patch that construction
  "$PY" - "$W/clmu_run.py" "$W/clmu_run_base.py" <<'PY'
import sys
s=open(sys.argv[1]).read().replace('{WORK}/runs/','{WORK}/runs_base/')
open(sys.argv[2],'w').write(s)
PY
  out=$(timeout 4000 "$PY" "$W/clmu_run_base.py" "$s" 2>&1 | grep -oE "$W/runs_base/[^ ]+clm.nc" | tail -1)
  if [ -n "$out" ] && [ -f "$out" ]; then cp "$out" "$OUT/${s}_base.nc"; echo "$s OK" >>"$LOG"; else echo "$s FAILED" >>"$LOG"; fi
done
echo "DONE $(date -u +%H:%M:%S)" >>"$LOG"; touch "$OUT/RUN_COMPLETE.flag"
