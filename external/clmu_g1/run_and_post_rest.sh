#!/bin/bash
cd /home/junmo/clmu_g1_work
. venv/bin/activate
for s in US-Minneapolis1 US-Minneapolis2 US-WestPhoenix PL-Lipowa FR-Capitole; do
  echo "########## $s : run ##########"
  rm -rf runs/$s/scriptsfolder/g1_* runs/$s/logfolder/g1_* runs/$s/inputfolder/usp 2>/dev/null
  python run_one.py $s 2>&1 | tail -3
  echo "########## $s : post ##########"
  python post.py $s 2>&1 | grep -E "align:|lwup_bias\"|qh_bias\"|dTsa_model\"|LWup_minus|TSKIN_minus|n_night" | head -20
  echo
done
echo "ALL_REST_DONE"
