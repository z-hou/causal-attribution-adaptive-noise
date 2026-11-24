# Environment for the GH200 GPU nodes (source from the project directory).
# torch/numpy/pandas/scipy/pytest come from the PyTorch module; scikit-learn is borrowed from the
# already installed venv-causal-gh200 through PYTHONPATH (nothing new installed).
VENVS=/nobackup/proj/disk/zhou-project/personal/zhozho/from_alvis/zhou-project
module load GPU/PyTorch/2.9.1-CUDA-12.9.1-eb >/dev/null 2>&1
export PYTHONPATH=$VENVS/venv-causal-gh200/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}
export PYTHONHASHSEED=0
