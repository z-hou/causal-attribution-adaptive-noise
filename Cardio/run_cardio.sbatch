#!/bin/bash
# Cardio, non-adaptive attack, same settings as the Adult main run results/n2000_k8_total_10seeds/nonadaptive:
# n_train = n_test = 2000, 8 defender and 8 attacker reference models, seeds 0–9,
# ε_total ∈ {2, 4, 8, 16, 32, 50, 64, 70, 80, 90, 100}.
#   sbatch Cardio/run_cardio.sbatch          (submit from the project root)
#SBATCH -A naiss2026-4-1772-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 00:50:00
#SBATCH -J ldp_cardio
#SBATCH -o Cardio/results/run_cardio_%j.log
set -e
source env.sh
python -B Cardio/cardio_data.py
F="--n-train 2000 --n-test 2000 --k-att 8 --k-def 8 --non-adaptive --seeds 0 1 2 3 4 5 6 7 8 9"
OUT=Cardio/results/nonadaptive
for T in 2 4 8 16 32 50 64 70 80 90 100; do
  echo "=== ε_total = $T"
  python -B Cardio/run_cardio.py $F --eps-total $T --out $OUT/epstotal$T/
done
python -B Cardio/summarize.py $OUT
