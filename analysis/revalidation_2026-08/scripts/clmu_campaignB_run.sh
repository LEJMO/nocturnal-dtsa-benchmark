#!/usr/bin/env bash
# Correct CLMU5 Campaign B runner: inputs_spin/ (5 recycled years) already built.
# Runs each site with clmu_run_spin.py (explicit spin input/output dirs).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # release: repository root
W=/root/clmu_work
PY=$W/venv/bin/python
OUT=$ROOT/external/clmu_spinup19
mkdir -p "$OUT"; LOG=$OUT/_run.log; : > "$LOG"
rm -f "$OUT/RUN_COMPLETE.flag"
cp "$ROOT/analysis/revalidation_2026-08/scripts/clmu_run_spin.py" "$W/clmu_run_spin.py"
docker ps >/dev/null 2>&1 || { nohup dockerd >/var/log/dockerd.log 2>&1 & sleep 8; }
SITES=$("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ROOT/results/paper_stats_v1.json'))['per_site'])))")
for s in $SITES; do
  [ -f "$W/inputs_spin/$s/forcing.nc" ] || { echo "$s NO_INPUT" >>"$LOG"; continue; }
  rm -rf "$W/runs_spin/$s" 2>/dev/null; mkdir -p "$W/runs_spin/$s"
  out=$(timeout 6000 "$PY" "$W/clmu_run_spin.py" "$s" 2>&1 | grep -oE "$W/runs_spin/[^ ]+clm.nc" | tail -1)
  if [ -n "$out" ] && [ -f "$out" ]; then
    cp "$out" "$OUT/${s}_spin.nc"; echo "$s OK" >>"$LOG"
  else
    echo "$s FAILED" >>"$LOG"
  fi
done
echo "DONE $(date -u +%H:%M:%S)" >>"$LOG"
touch "$OUT/RUN_COMPLETE.flag"
