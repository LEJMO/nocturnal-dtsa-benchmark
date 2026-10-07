#!/usr/bin/env bash
# Adapt the frozen CLMU5 runner to the rebuilt WSL (/root/clmu_work) and
# validate baseline reproduction on KR-Ochang (archived clmu_dtsa=+0.046,
# clmu_bias=+10.79). Run as root in Ubuntu-24.04 with dockerd up.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # release: repository root
W=/root/clmu_work
PY=$W/venv/bin/python
mkdir -p "$W/inputs" "$W/runs"
# patch the frozen scripts' hardcoded home path -> /root/clmu_work
sed 's#/home/junmo/clmu_g1_work#/root/clmu_work#g' \
    "$ROOT/external/clmu_g1/build_inputs.py" > "$W/clmu_build.py"
sed 's#/home/junmo/clmu_g1_work#/root/clmu_work#g' \
    "$ROOT/external/clmu_g1/run_one.py" > "$W/clmu_run.py"

echo "=== BUILD KR-Ochang $(date -u +%H:%M:%S) ==="
"$PY" "$W/clmu_build.py" KR-Ochang 2>&1 | tail -3

echo "=== ensure dockerd ==="
docker ps >/dev/null 2>&1 || { nohup dockerd >/var/log/dockerd.log 2>&1 & sleep 8; }
docker ps >/dev/null 2>&1 && echo "docker up" || { echo "DOCKER_DOWN"; exit 5; }

echo "=== RUN KR-Ochang $(date -u +%H:%M:%S) ==="
cd "$W"
timeout 3000 "$PY" "$W/clmu_run.py" KR-Ochang 2>&1 | tail -8
echo "=== DONE $(date -u +%H:%M:%S) ==="
echo CLMU_VALIDATE_DONE
