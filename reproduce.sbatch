#!/bin/bash
# Adult main experiment: unit tests, then ε_total ∈ {2, 4, 8, 16, 32, 50, 64, 70, 80, 90, 100}, seeds 0–9.
# Reference results: results/n2000_k8_total_10seeds_labelclean/nonadaptive/.
#   sbatch reproduce.sbatch          (submit from the project root)
#SBATCH -A naiss2026-4-1772-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J adaptive_ldp
#SBATCH -o reproduce_%j.log
set -e
source env.sh
python -B -m pytest -q -p no:cacheprovider
for T in 2 4 8 16 32 50 64 70 80 90 100; do
  python -B -m adaptive_dp.run --eps-total $T --out results/main/epstotal$T/
done
