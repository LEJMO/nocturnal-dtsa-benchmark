#!/usr/bin/env bash
# CLMU5 Campaign B (equilibrium spin-up): rerun the 19 records with N recycled
# spin-up years (vs the baseline's 1), to test whether deep-substrate
# equilibration changes CLMU5's record-period nocturnal state (reviewer Major 3).
set -uo pipefail
NYR=${1:-5}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # release: repository root
W=/root/clmu_work
PY=$W/venv/bin/python
OUT=$ROOT/external/clmu_spinup19
mkdir -p "$OUT"
LOG=$OUT/_run.log; : > "$LOG"
docker ps >/dev/null 2>&1 || { nohup dockerd >/var/log/dockerd.log 2>&1 & sleep 8; }

# generate a spin-up build variant: prepend NYR recycled years (tile first year)
"$PY" - "$W/clmu_build.py" "$W/clmu_build_spin.py" "$NYR" <<'PY'
import sys
src=open(sys.argv[1]).read(); nyr=int(sys.argv[3])
old='''def prepend_and_pad(core, t):
    """Prepend one recycled year (first STEPS_PER_YEAR steps or whole record),
    append 2 pad days. Return (core_full, times_full, n_spin)."""
    n = len(t); dt = pd.Timedelta('1800s')
    nsp = min(STEPS_PER_YEAR, n)
    spin_t = pd.date_range(end=t[0]-dt, periods=nsp, freq='1800s')
    pad_steps = 96  # 2 days
    pad_t = pd.date_range(start=t[-1]+dt, periods=pad_steps, freq='1800s')
    times_full = spin_t.append(t).append(pad_t)
    core_full={}
    for k,v in core.items():
        core_full[k] = np.concatenate([v[:nsp], v, v[-pad_steps:]])
    return core_full, times_full, nsp'''
new=f'''def prepend_and_pad(core, t):
    """Prepend {nyr} recycled years (first-year block tiled), append 2 pad days."""
    n = len(t); dt = pd.Timedelta('1800s')
    base = min(STEPS_PER_YEAR, n); NYR = {nyr}; nsp = base*NYR
    spin_t = pd.date_range(end=t[0]-dt, periods=nsp, freq='1800s')
    pad_steps = 96
    pad_t = pd.date_range(start=t[-1]+dt, periods=pad_steps, freq='1800s')
    times_full = spin_t.append(t).append(pad_t)
    core_full={{}}
    for k,v in core.items():
        core_full[k] = np.concatenate([np.tile(v[:base], NYR), v, v[-pad_steps:]])
    return core_full, times_full, nsp'''
assert old in src, "prepend_and_pad block not found verbatim"
open(sys.argv[2],'w').write(src.replace(old,new))
print("wrote clmu_build_spin.py NYR=",nyr)
PY

SITES=$("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ROOT/results/paper_stats_v1.json'))['per_site'])))")
for s in $SITES; do
  # spin-up inputs into a separate dir so campaign A inputs are untouched
  OUTDIR=$W/inputs_spin/$s
  rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
  # build with spin variant, redirecting OUT
  OUTBASE=$("$PY" -c "print('$W/inputs_spin')")
  sed "s#OUT = \"/root/clmu_work/inputs\"#OUT = \"$W/inputs_spin\"#" "$W/clmu_build_spin.py" > "$W/clmu_build_spin_r.py"
  "$PY" "$W/clmu_build_spin_r.py" "$s" >/dev/null 2>&1
  rm -rf "$W/runs_spin/$s" 2>/dev/null; mkdir -p "$W/runs_spin/$s"
  # run: patch clmu_run to read inputs_spin + runs_spin for this
  sed -e "s#/root/clmu_work/inputs#$W/inputs_spin#g" -e "s#/root/clmu_work/runs#$W/runs_spin#g" "$W/clmu_run.py" > "$W/clmu_run_spin.py"
  out=$(timeout 6000 "$PY" "$W/clmu_run_spin.py" "$s" 2>&1 | grep -oE "$W/runs_spin/[^ ]+clm.nc" | tail -1)
  if [ -n "$out" ] && [ -f "$out" ]; then
    cp "$out" "$OUT/${s}_spin.nc"; echo "$s OK nyr=$NYR" >>"$LOG"
  else
    echo "$s FAILED" >>"$LOG"
  fi
done
echo "DONE $(date -u +%H:%M:%S)" >>"$LOG"
touch "$OUT/RUN_COMPLETE.flag"
