#!/usr/bin/env bash
# Run TEB Campaign B (equilibrium spin-up) driver over the full metforcing span
# per site, in parallel. Each site writes output/LWU.txt -> LWU_spin.txt.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # release: repository root
DRIVER=$ROOT/external/teb/build/driver
OUT=$ROOT/external/teb_spinup10
LOG=$OUT/_run.log
: > "$LOG"
SITES=$(ls -d "$OUT"/*/ 2>/dev/null | xargs -n1 basename | grep -v '^_')

run_one () {
  local s=$1 wd=$OUT/$1
  [ -f "$wd/input.nml" ] || { echo "$s NO_INPUT" >>"$LOG"; return; }
  rm -f "$wd/output"/* 2>/dev/null
  ( cd "$wd" && "$DRIVER" >driver.log 2>&1 )
  if [ -s "$wd/output/LWU.txt" ]; then
    cp "$wd/output/LWU.txt" "$wd/LWU_spin.txt"
    echo "$s OK $(wc -l <"$wd/LWU_spin.txt")" >>"$LOG"
  else
    echo "$s FAILED" >>"$LOG"
  fi
}
export -f run_one; export OUT DRIVER LOG
echo "start $(date -u +%H:%M:%S)" >>"$LOG"
echo "$SITES" | xargs -P 10 -I{} bash -c 'run_one "$@"' _ {}
echo "done $(date -u +%H:%M:%S)" >>"$LOG"
touch "$OUT/RUN_COMPLETE.flag"
