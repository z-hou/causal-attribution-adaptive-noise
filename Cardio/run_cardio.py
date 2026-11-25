"""Run the current main pipeline (adaptive_dp.run) unchanged on Cardio.

Only the data pipeline is swapped (adaptive_dp.data.prepare → cardio_data.prepare); defender analysis,
budget allocation, input perturbation, training and the attack are the Adult code itself, and all
command-line flags are those of adaptive_dp.run. With --eps-total T the budget is split equally over the
11 attributes and the label: ε̄ = ε_y = T / 12.

    python -B Cardio/run_cardio.py --n-train 2000 --n-test 2000 --k-att 8 --k-def 8 --non-adaptive \
        --seeds 0 1 2 3 4 5 6 7 8 9 --eps-total 16 --out Cardio/results/nonadaptive/epstotal16/
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import adaptive_dp.run as pipeline  # noqa: E402
import cardio_data  # noqa: E402

pipeline.prepare = cardio_data.prepare

if __name__ == "__main__":
    pipeline.main()
