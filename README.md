# 因果归因自适应输入端加噪：Adult 复现说明

本目录在 UCI Adult 上比较三种训练方式的测试准确率和成员推断攻击（MIA）成功率：

- **nonprivate**：不加噪；
- **uniform**：输入端本地差分隐私加噪，12 个属性的预算相同；
- **adaptive**：总预算与 uniform 相同，按每个属性对 MIA 和准确率的因果效应分配预算。

完整的实验规格（数据处理、超参数、检查项）见 [AGENT.md](AGENT.md)。本文件只说明怎样复现。当前代码和 AGENT.md 有几处不同，见文末「与 AGENT.md 的差异」。

## 方法概要

1. **防御方分析**：直接分析用 D_train 训练的目标模型 train(D_train, seed=0)，它就是 nonprivate 种子 0 的目标模型。
   - **代理攻击**：离线 RMIA（γ = 2）。成员是 D_train，非成员 N 是 P_def 的随机一半。4 个参考模型都用 P_def 的另一半训练，对所有被评估的记录都是 OUT，所以 Pr(x) 直接取参考模型的均值。
   - **属性干预**：按 FSE 2022 的方式，把 D_train、N 和 Z_def 全部记录的属性 j 设成同一个取值 α，遍历所有取值后求平均。由此得到每个属性的 ΔMIA 和 ΔAcc。
   - 只分析这一个模型，不对多个模型或轮换求平均。
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
  defender.py       代理攻击（目标模型 + 参考模型）、属性干预、ΔMIA / ΔAcc
  allocate.py       预算分配 w_j
  attack.py         攻击者参考模型、离线 RMIA、损失攻击
  run.py            命令行入口：串起全流程，写 results.csv 和 allocation.csv
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
- **参考环境**（`env.sh` 加载的版本，下面的参考数字就是用它得到的）：Python 3.13.5、torch 2.9.1（CUDA 12.9）、numpy 2.3.2、scikit-learn 1.9.1、pandas 2.3.1、scipy 1.16.1。

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

脚本依次执行：单元测试 → ε̄ = 3、5、10、20 各一次完整实验，每次用种子 0、1、2。日志写到 `reproduce_<jobid>.log`，结果写到 `results/`。在 GH200 节点上，整个作业几分钟就能跑完。其他用户需要把脚本里的 `#SBATCH -A` 改成自己的项目账号。

### 方式二：手动运行

```bash
source env.sh                    # 或激活你自己的 Python 环境
python -m pytest -q              # 应显示 14 passed
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
| `--non-adaptive` | 非自适应攻击：三种方法都用同一组参考模型（在不加噪的 P_att 上训练）和同一个 a。不加这个开关时为自适应攻击，参考模型按目标模型的方式加噪 |

ε̄ = 3 是 AGENT.md 规定的主实验，5、10、20 是后来加跑的。

### 输出

每次运行在 `--out` 目录下写两个文件，并在终端打印 3 个种子的均值 ± 标准差，可以直接和下面的「预期结果」对照。

| 文件 | 内容 |
| --- | --- |
| `results.csv` | 9 个目标模型（3 种方法 × 3 个种子）的指标：test_acc、train_acc、gap、rmia_auc、rmia_tpr1、rmia_tpr01、loss_auc，以及攻击者选出的参数 a |
| `allocation.csv` | 12 个属性的 ΔMIA、ΔAcc、预算份额 w_j，以及 uniform 和 adaptive 各自的 ε_j |

## 预期结果

表中为 3 个种子的均值 ± 样本标准差。test_acc 和 TPR@1%FPR 的单位是 %。rmia 指离线 RMIA，loss_auc 是损失攻击的 AUC。nonprivate 不加噪，所以四档预算下的结果完全相同。

| ε̄ | ε_total | 方法 | test_acc | rmia_auc | TPR@1%FPR | loss_auc |
| --- | --- | --- | --- | --- | --- | --- |
| — | ∞ | nonprivate | 82.83 ± 0.44 | 0.5660 ± 0.0084 | 2.72 ± 0.05 | 0.5324 ± 0.0026 |
| 3 | 39 | uniform | 74.64 ± 3.10 | 0.5087 ± 0.0089 | 1.36 ± 0.29 | 0.5083 ± 0.0061 |
| 3 | 39 | adaptive | 76.35 ± 0.68 | 0.5016 ± 0.0031 | 1.09 ± 0.14 | 0.5018 ± 0.0038 |
| 5 | 63 | uniform | 80.63 ± 0.54 | 0.5286 ± 0.0007 | 2.04 ± 0.18 | 0.5180 ± 0.0028 |
| 5 | 63 | adaptive | 80.06 ± 1.12 | 0.5175 ± 0.0066 | 1.84 ± 0.04 | 0.5138 ± 0.0066 |
| 10 | 123 | uniform | 81.23 ± 0.44 | 0.5510 ± 0.0036 | 2.59 ± 0.07 | 0.5338 ± 0.0031 |
| 10 | 123 | adaptive | 81.51 ± 0.47 | 0.5524 ± 0.0043 | 2.58 ± 0.10 | 0.5343 ± 0.0027 |
| 20 | 243 | uniform | 81.77 ± 0.33 | 0.5534 ± 0.0038 | 2.52 ± 0.14 | 0.5349 ± 0.0036 |
| 20 | 243 | adaptive | 82.10 ± 0.21 | 0.5539 ± 0.0041 | 2.54 ± 0.15 | 0.5348 ± 0.0032 |

防御方分析与 ε̄ 无关，四次运行的结果相同：

- 不干预时，代理攻击的 AUC 为 0.5631，目标模型在 N 上的准确率为 82.84%；
- w_j 最小为 0.443（sex），最大为 1.575（capital-gain）。

**可复现性**：2026-11-24 在 GH200 节点上，用 `env.sh` 环境把同样的运行重复跑过，`results.csv` 和 `allocation.csv` 每次都逐字节一致。申请 1 个还是 8 个 CPU 不影响结果。逐字节一致需要相同的软件版本：换用其他版本的 torch，或换成 CPU、其他型号的 GPU 时，浮点运算会有细微差别，训练轨迹随之改变。这时数据划分不变，但防御方分配、攻击参数 a 和逐种子的指标都会和上表不同。

## 注意事项

- 每组只有 3 个种子。adaptive 与 uniform 的差值大多没有超过两组标准差之和，要下结论需要更多种子。
- TPR@0.1%FPR（`rmia_tpr01`）只对应 7,000 个非成员中的 7 个误报点，数值不稳定，所以上表没有列出。
- **预算分配依赖私有数据**：w_j 是在 D_train 上训练的模型上算出来的。ε_total 只覆盖输入端加噪，分配这一步本身会泄露 D_train 的信息，这部分没有计入隐私预算。AGENT.md 原本只用公开数据做分配，正是为了避免这个问题。
- 这里的 ε_total 是**本地**差分隐私预算（δ = 0）。每条记录在进入训练前就已经加噪，所以不能和 DP-SGD 的 ε 直接比较。

## 与 AGENT.md 的差异

- **防御方分析**（2026-11-24 的决定）：不再用 4 个校准模型轮换取平均（AGENT.md §6.1–6.2），改为分析 train(D_train, seed=0)，做法见「方法概要」第 1 步。N 和参考模型训练集 R 来自 P_def 的一个随机排列（种子 2000），4 个参考模型的种子是 2001–2004。三个种子的 uniform 和 adaptive 共用同一套 w_j。隐私上的后果见「注意事项」。
- **预算**：ε̄ = 5、10、20 是在 AGENT.md 规定的 ε̄ = 3 之外加跑的。标签预算 ε_y 始终是 3。
- **精简**：
  - 去掉了冒烟测试（§11）和运行时的阈值检查（§10.6–10.8）。§10.7 本来就需要多次轮换，在单模型分析下无法计算。
  - 去掉了 `summary.md` 等报告文件（§8.3），只输出 `results.csv` 和 `allocation.csv`。
  - 单元测试（§10.2–10.5）保留。

## 实现细节

AGENT.md 没有规定、由实现决定的地方：

- **随机数**：
  - 数据划分、各种抽样和输入端加噪都用 `numpy.random.default_rng(seed)`。
  - 模型初始化用 `torch.manual_seed(seed)`，打乱数据用以 seed 初始化的 `torch.Generator`。
  - 目标模型的加噪种子和训练种子都是 seed。
- **加噪顺序**：同一个生成器先依次处理属性 j = 1…12，再处理标签。广义随机响应的类别数 k_j 按完整类别表计算（adult.names 中的类别加 `Unknown`），包括数据里没有出现过的类别。
- **攻击者**：
  - 参考模型 i 的抽样种子和训练种子都是 4000 + i，加噪种子是 4100 + i。
  - 选 a 时，成员和非成员各取 1,000 条：成员是参考模型 1 独有的训练记录，非成员来自 Z_att。两者用种子 4000 依次抽取，AUC 并列时取较小的 a。
- **数值精度**：Pr(x | m) 先把 float32 logits 转成 float64，再做 softmax。
- **数据来源**：`data/raw/adult/` 里的文件复制自本机已有的 UCI 原始文件，没有重新下载。md5 见「数据」一节。
