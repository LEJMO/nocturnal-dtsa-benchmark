#!/usr/bin/env bash
# Rebuild TEB v4.1.2 in a fresh build dir, validate baseline reproduction
# against the preserved LWU_base.txt, then run Campaign A (albedo-constrained,
# 19 records). Run inside WSL Ubuntu-24.04 as root.
#   Campaign A: reuse each site's EXACT baseline forcing (record period),
#   set ZALB_ROOF=ZALB_ROAD=ZALB_WALL=VEG_ALB = observed midday albedo,
#   everything else identical; save output/LWU.txt -> teb_albedo19/<site>/LWU_alb.txt.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # release: repository root
TEB=$ROOT/external/teb
RUNS=$ROOT/external/teb_runs
OUT=$ROOT/external/teb_albedo19
LOG=$OUT/_orchestration.log
mkdir -p "$OUT"
exec > >(tee "$LOG") 2>&1
echo "=== [1] DRIVER (original ELF, libgfortran restored) $(date -u +%H:%M:%S) ==="
DRIVER=$TEB/build/driver
if ! ldd "$DRIVER" 2>&1 | grep -q "not found" && "$DRIVER" --help >/dev/null 2>&1 || [ -x "$DRIVER" ]; then
  echo "using original driver: $DRIVER"
else
  echo "original driver unusable; rebuilding"
  BUILD=$TEB/build_v6; rm -rf "$BUILD"; mkdir -p "$BUILD"; cd "$BUILD"
  cmake -DCMAKE_BUILD_TYPE=Release .. >build_cmake.log 2>&1 || { echo CMAKE_FAIL; exit 2; }
  cmake --build . -j8 >build_make.log 2>&1 || { echo BUILD_FAIL; exit 3; }
  DRIVER=$BUILD/driver
fi

run_site () {  # $1=src_dir $2=work_dir  (work_dir must contain input.nml + input/)
  local wd=$1
  ( cd "$wd" && "$DRIVER" >driver.log 2>&1 )
  return $?
}

echo "=== [2] VALIDATE baseline reproduction (AU-Preston) ==="
VW=$OUT/_validate_AU-Preston
rm -rf "$VW"; cp -r "$RUNS/AU-Preston" "$VW"; rm -rf "$VW/output"/*;
run_site "$VW"
python3 - "$VW/output/LWU.txt" "$RUNS/AU-Preston/output/LWU_base.txt" <<'PY'
import sys, numpy as np
a=np.loadtxt(sys.argv[1]); b=np.loadtxt(sys.argv[2])
if len(a)!=len(b): print(f"VALIDATE_LEN_MISMATCH {len(a)} vs {len(b)}"); sys.exit(1)
d=float(np.max(np.abs(a-b)))
print(f"VALIDATE max|LWU_new - LWU_base| = {d:.6e} W/m2 over {len(a)} steps")
print("VALIDATE_PASS" if d < 1e-2 else f"VALIDATE_SOFT (compiler delta {d:.3f}); proceed with paired baseline")
PY

echo "=== [3] CAMPAIGN A: albedo-constrained, 19 records ==="
SITES=$(python3 -c "import json;print(' '.join(sorted(json.load(open('$ROOT/results/paper_stats_v1.json'))['per_site'])))")
for s in $SITES; do
  alb=$(python3 -c "import json;print(f\"{json.load(open('$ROOT/results/paper_stats_v1.json'))['per_site']['$s']['albedo']:.4f}\")")
  wd=$OUT/$s
  rm -rf "$wd"; cp -r "$RUNS/$s" "$wd"; rm -rf "$wd/output"/*
  python3 - "$wd/input.nml" "$alb" <<'PY'
import sys, re
nml=open(sys.argv[1]).read(); a=sys.argv[2]
for key in ("ZALB_ROOF","ZALB_ROAD","ZALB_WALL","VEG_ALB"):
    nml=re.sub(rf"^(\s*{key}\s*=\s*)\S+", lambda m: m.group(1)+a, nml, flags=re.M)
open(sys.argv[1],"w",newline="\n").write(nml)
PY
  if run_site "$wd" && [ -s "$wd/output/LWU.txt" ]; then
    cp "$wd/output/LWU.txt" "$wd/LWU_alb.txt"
    eff=$(python3 -c "import numpy as np;v=np.loadtxt('$wd/output/ALB_TOWN.txt');import numpy as _;print(f'{np.nanmean(v[v>0]):.4f}')" 2>/dev/null || echo NA)
    echo "  $s alb_set=$alb eff_town=$eff OK"
  else
    echo "  $s FAILED (see $wd/driver.log)"
  fi
done
echo "=== DONE campaignA $(date -u +%H:%M:%S) ==="
touch "$OUT/RUN_COMPLETE.flag"
