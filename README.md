# 因果归因自适应输入端加噪：Adult 复现说明

本目录在 UCI Adult 上比较三种训练方式的测试准确率和成员推断攻击（MIA）成功率：

- **nonprivate**：不加噪；
- **uniform**：输入端本地差分隐私加噪，12 个属性的预算相同；
- **adaptive**：总预算与 uniform 相同，按每个属性对 MIA 和准确率的因果效应分配预算。

完整的实验规格（数据处理、超参数、检查项）见 [AGENT.md](AGENT.md)。本文件只说明怎样复现。

## 方法概要

1. **防御方分析**：只用公开数据 P_def、Z_def，不接触 D_train。先训练 4 个校准模型，用在线 RMIA 作代理攻击。再按 FSE 2022 的方式逐个干预属性：把全体记录的该属性设为同一取值，遍历所有取值后取平均。由此得到每个属性的 ΔMIA 和 ΔAcc。
2. **预算分配**：由 ΔMIA、ΔAcc 算出每个属性的预算份额 w_j，均值为 1。
3. **输入端加噪**：对 D_train 的每条记录只加一次噪声。类别属性用广义随机响应，连续属性用 Laplace 机制，标签用二元随机响应。uniform 中 ε_j = ε̄，adaptive 中 ε_j = ε̄·w_j，两者的总预算 ε_total = 12·ε̄ + ε_y 相同（ε_y = 3）。
4. **训练与攻击**：用加噪后的数据训练 MLP。攻击者用与防御方不相交的数据训练参考模型，再做离线 RMIA；另外报告损失攻击作为基线。

## 目录

```
adaptive_dp/        实验代码
  data.py           读取与校验 Adult、预处理、属性分组、六段划分
  model.py          MLP（95 → 512 → 256 → 2）
  train.py          训练函数（所有模型共用）
  perturb.py        输入端加噪：GRR / Laplace / 标签 RR
  rmia.py           RMIA 分数、AUC、TPR@FPR
  defender.py       校准模型、代理攻击、属性干预、ΔMIA / ΔAcc
  allocate.py       预算分配 w_j
  attack.py         攻击者参考模型、离线 RMIA、损失攻击
  run.py            命令行入口，串起全流程并写 summary.md
tests/              14 个单元测试
data/raw/adult/     UCI Adult 原始文件
AGENT.md            实验规格
env.sh              本集群 GH200 计算节点的环境设置
reproduce.sbatch    一键复现（Slurm）
```

## 环境

- 需要 Python ≥ 3.10，以及 `torch`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`pytest`。
- GPU 不是必需的。下面的参考数字是在 1 块 GH200 GPU 上跑出来的，每个预算的完整运行约 25 秒。
- **本集群**：`source env.sh`。它会加载 PyTorch 模块，并通过 `PYTHONPATH` 借用 `venv-causal-gh200` 里的 scikit-learn。这个脚本只能在 GH200 计算节点（aarch64）上用，在 x86 登录节点上不能用。
- **其他机器**：装好上面列出的依赖即可。代码只用相对于本目录的路径。

## 数据

`data/raw/adult/` 里应有 UCI Adult 的三个原始文件，来源：<https://archive.ics.uci.edu/dataset/2/adult>。

| 文件 | md5 |
| --- | --- |
| `adult.data` | `5d7c39d7b8804f071cdd1f2a7c460872` |
| `adult.test` | `35238206dfdf7f1fe215bbb874adecdc` |
| `adult.names` | `1a7cdb3ff7a1b709968b1c7a11def63e` |

可以用 `md5sum data/raw/adult/*` 核对。程序读入后会检查：合并后共 48,842 行，14 个属性加 1 个标签，标签只有 `<=50K` 和 `>50K` 两种取值。任何一项不符就报错停止。

## 复现

### 方式一：Slurm 一键运行

```bash
sbatch reproduce.sbatch
```

脚本依次执行：单元测试 → 冒烟测试 → ε̄ = 3、5、10、20 各一次完整实验，每次用种子 0、1、2。日志写到 `reproduce_<jobid>.log`，结果写到 `results/`。在 GH200 节点上，整个作业几分钟就能跑完。其他用户需要把脚本里的 `#SBATCH -A` 改成自己的项目账号。

### 方式二：手动运行

```bash
source env.sh                    # 或激活你自己的 Python 环境
python -m pytest -q              # 应显示 14 passed
python -m adaptive_dp.run --smoke --out results/smoke/
python -m adaptive_dp.run --eps-mean 3  --seeds 0 1 2 --out results/epsbar3/
python -m adaptive_dp.run --eps-mean 5  --seeds 0 1 2 --out results/epsbar5/
python -m adaptive_dp.run --eps-mean 10 --seeds 0 1 2 --out results/epsbar10/
python -m adaptive_dp.run --eps-mean 20 --seeds 0 1 2 --out results/epsbar20/
```

| 参数 | 含义 |
| --- | --- |
| `--eps-mean` | 属性平均预算 ε̄，默认 3（AGENT.md 的规定值）。标签预算 ε_y 固定为 3 |
| `--seeds` | 目标模型的种子，默认 `0 1 2` |
| `--out` | 输出目录，默认 `results/` |
| `--smoke` | 冒烟测试：数据缩小到 1/5，2 次轮换，连续属性取 3 个网格点，3 个 epoch，只跑种子 0 |

ε̄ = 3 是 AGENT.md 规定的主实验，5、10、20 是后来加跑的，各自 `summary.md` 的偏离记录里有说明。

完整实验遇到以下任一情况会报错停止（AGENT.md §10）：

- 代理攻击的平均 AUC ≤ 0.55；
- ΔMIA 的折半 Spearman 相关 < 0.5；
- nonprivate 模型的 rmia_auc ≤ 0.55。

### 输出文件

每个输出目录包含以下文件：

| 文件 | 内容 |
| --- | --- |
| `summary.md` | 汇总表、对比表、防御方分析、结论、偏离记录（**先看这个**） |
| `results.csv` | 9 个目标模型（3 种方法 × 3 个种子）的指标 |
| `allocation.csv` | 12 个属性的 ΔMIA、ΔAcc、w_j、两种方法各自的 ε_j 和噪声水平 |
| `interventions.csv` | 每个（属性，取值，轮换）组合的干预结果 |
| `defender_baseline.csv` | 不干预时 4 次轮换的代理攻击 AUC 和准确率 |
| `attacker_a_selection.csv` | 攻击者选择参数 a 的过程 |
| `splits.npz`、`preprocess.json` | 六段划分的下标和预处理参数 |
| `run_meta.json` | 运行设置和用时 |

## 预期结果

表中为 3 个种子的均值 ± 样本标准差。test_acc 和 TPR@1%FPR 的单位是 %。rmia 指离线 RMIA，loss_auc 是损失攻击的 AUC。nonprivate 不加噪，所以四档预算下的结果完全相同。

| ε̄ | ε_total | 方法 | test_acc | rmia_auc | TPR@1%FPR | loss_auc |
| --- | --- | --- | --- | --- | --- | --- |
| — | ∞ | nonprivate | 82.83 ± 0.44 | 0.5660 ± 0.0084 | 2.72 ± 0.05 | 0.5324 ± 0.0026 |
| 3 | 39 | uniform | 74.64 ± 3.10 | 0.5087 ± 0.0089 | 1.36 ± 0.29 | 0.5083 ± 0.0061 |
| 3 | 39 | adaptive | 75.26 ± 3.89 | 0.5019 ± 0.0036 | 1.06 ± 0.19 | 0.5045 ± 0.0039 |
| 5 | 63 | uniform | 80.63 ± 0.54 | 0.5286 ± 0.0007 | 2.04 ± 0.18 | 0.5180 ± 0.0028 |
| 5 | 63 | adaptive | 79.71 ± 1.76 | 0.5219 ± 0.0092 | 1.48 ± 0.10 | 0.5154 ± 0.0064 |
| 10 | 123 | uniform | 81.23 ± 0.44 | 0.5510 ± 0.0036 | 2.59 ± 0.07 | 0.5338 ± 0.0031 |
| 10 | 123 | adaptive | 81.78 ± 0.44 | 0.5526 ± 0.0033 | 2.58 ± 0.23 | 0.5339 ± 0.0013 |
| 20 | 243 | uniform | 81.77 ± 0.33 | 0.5534 ± 0.0038 | 2.52 ± 0.14 | 0.5349 ± 0.0036 |
| 20 | 243 | adaptive | 81.97 ± 0.19 | 0.5559 ± 0.0037 | 2.47 ± 0.08 | 0.5349 ± 0.0033 |

防御方分析与 ε̄ 无关，四次运行的结果相同：

- 代理攻击的平均 AUC 为 0.5740；
- ΔMIA 的折半 Spearman 相关为 0.9301；
- w_j 最小为 0.750（workclass），最大为 1.453（education-num）。

**可复现性**：2026-11-24 在 GH200 节点上用本目录代码重跑了四档预算。每个输出目录里除 `run_meta.json` 中的运行用时外，所有结果文件都与上表所依据的原始结果逐字节一致，包括 `results.csv`、`allocation.csv`、`interventions.csv`、`summary.md` 等。换成 CPU 或其他型号的 GPU 时，浮点运算顺序会不同，逐种子的数字可能有出入。

## 注意事项

- 每组只有 3 个种子。adaptive 与 uniform 的差值大多没有超过两组标准差之和，逐项判断见各 `summary.md` 的对比表。
- TPR@0.1%FPR（`rmia_tpr01`）只对应 7,000 个非成员中的 7 个误报点，数值不稳定，所以上表没有列出。
- 这里的 ε_total 是**本地**差分隐私预算（δ = 0）。每条记录在进入训练前就已经加噪，所以不能和 DP-SGD 的 ε 直接比较。
