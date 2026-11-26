"""Run the current main pipeline (adaptive_dp.run) unchanged on folktables ACSEmployment (CA, 2018).

Only the data pipeline is swapped (adaptive_dp.data.prepare → acs_data.prepare); defender analysis,
budget allocation, input perturbation, training and the attack are the Adult code itself, and all
command-line flags are those of adaptive_dp.run. With --eps-total T the whole budget goes to the
16 attributes (ε̄ = T / 16); the label is not perturbed.

    python -B ACSEmployment/run_acs.py --eps-total 16 --out ACSEmployment/results/nonadaptive_labelclean/epstotal16/
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import acs_data  # noqa: E402
import adaptive_dp.run as pipeline  # noqa: E402

pipeline.prepare = acs_data.prepare

if __name__ == "__main__":
    pipeline.main()
