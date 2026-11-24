#!/bin/bash
# Reproduce all Adult results: unit tests, smoke test, then ε̄ ∈ {3, 5, 10, 20} with seeds 0 1 2.
#   sbatch reproduce.sbatch
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
python -B -m adaptive_dp.run --smoke --out results/smoke/
for E in 3 5 10 20; do
  python -B -m adaptive_dp.run --eps-mean $E --seeds 0 1 2 --out results/epsbar$E/
done
