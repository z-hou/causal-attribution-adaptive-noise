#!/bin/bash
# folktables ACSEmployment (CA, 2018 1-Year), same setting as the Adult main experiment (adaptive_dp.run):
# clean labels, non-adaptive attack, n_train = n_test = 2000, 8 + 8 reference models, seeds 0–9,
# ε_total ∈ {2, 4, 8, 16, 32, 50, 64, 70, 80, 90, 100}.
#   sbatch ACSEmployment/run_acs.sbatch          (submit from the project root)
#SBATCH -A naiss2026-4-1772-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J ldp_acs_emp
#SBATCH -o ACSEmployment/results/run_acs_%j.log
set -e
source env.sh
python -B ACSEmployment/acs_data.py
OUT=ACSEmployment/results/nonadaptive_labelclean
for T in 2 4 8 16 32 50 64 70 80 90 100; do
  echo "=== ε_total = $T"
  python -B ACSEmployment/run_acs.py --eps-total $T --out $OUT/epstotal$T/
done
python -B ACSEmployment/summarize.py $OUT
