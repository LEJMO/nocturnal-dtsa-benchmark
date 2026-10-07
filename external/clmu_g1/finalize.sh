#!/bin/bash
# Copy CLMU5 G1 artifacts from WSL workdir to the repo external/clmu_g1/ (Windows D:).
# Per site: run_params.json (config), <site>.json (metrics+provenance), surfdata.nc
# (exact config), and the raw CLM history output. Plus g1_summary.json at top.
set -e
cd /home/junmo/clmu_g1_work
. venv/bin/activate
python aggregate.py
DEST=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # release: this folder
mkdir -p $DEST
cp results/g1_summary.json $DEST/g1_summary.json
for s in US-Minneapolis1 US-Minneapolis2 KR-Ochang US-WestPhoenix PL-Lipowa FR-Capitole; do
  mkdir -p $DEST/$s
  cp -f inputs/$s/run_params.json $DEST/$s/ 2>/dev/null || true
  cp -f results/$s.json $DEST/$s/metrics.json 2>/dev/null || true
  cp -f inputs/$s/surfdata.nc $DEST/$s/surfdata.nc 2>/dev/null || true
  hist=$(ls -t runs/$s/outputfolder/lnd/hist/*.nc 2>/dev/null | head -1)
  if [ -n "$hist" ]; then cp -f "$hist" $DEST/$s/clm2_history.nc; fi
done
echo "=== external/clmu_g1 tree ==="
ls -la $DEST
for s in US-Minneapolis1 US-Minneapolis2 KR-Ochang US-WestPhoenix PL-Lipowa FR-Capitole; do echo "-- $s --"; ls -la $DEST/$s 2>/dev/null; done
