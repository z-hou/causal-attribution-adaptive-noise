# 因果归因自适应输入端加噪：针对成员推断的防御

本目录比较三种训练方式的测试准确率和成员推断攻击（MIA）成功率：

- **nonprivate**：不加噪；
- **uniform**：对每条训练记录的属性加一次本地差分隐私噪声，各属性预算相同；
- **adaptive**：总预算与 uniform 相同，按每个属性对 MIA 和准确率的因果效应分配预算。

标签不敏感，不加噪；噪声只加在属性上，目的是抵御成员推断。主实验在 UCI Adult 上；[ACSEmployment/](ACSEmployment/README.md) 和 [Cardio/](Cardio/README.md) 用同一套代码跑其他数据集。

最初的实验规格见 [AGENT.md](AGENT.md)。当前代码和它有几处不同，见文末「与 AGENT.md 的差异」。

## 方法概要

1. **防御方分析**：直接分析目标模型 train(D_train, seed 0)。
   - **代理攻击**：离线 RMIA（γ = 2）。成员是 D_train，非成员 N 是 P_def 的随机一半。8 个参考模型各用 P_def 另一半中的 2,000 条训练，对所有被评估的记录都是 OUT，所以 Pr(x) 取参考模型的均值。
   - **属性干预**：按 FSE 2022 的方式，把 D_train、N 和 Z_def 全部记录的属性 j 设成同一个取值 α，遍历所有取值后求平均，得到 ΔMIA_j（代理攻击 AUC 的变化）和 ΔAcc_j（目标模型在 N 上准确率的变化）。
2. **预算分配**：V_j = max(−ΔMIA_j, 0)、U_j = max(−ΔAcc_j, 0)，各自除以均值；w_j = clip(s · ((U_j + 0.05)/(V_j + 0.05))^(1/3), 0.1, 10)，s 使 w 的均值为 1。
3. **输入端加噪**：每条记录的属性只加一次噪声。类别属性用广义随机响应，连续属性加 Laplace 噪声后截断到 [0, 1]。J 个属性时，uniform 中 ε_j = ε_total / J，adaptive 中 ε_j = ε_total / J · w_j，两者都满足 Σ_j ε_j = ε_total。
4. **训练与评估**：用加噪后的数据训练 MLP。攻击者是**非自适应**的：用不加噪的 P_att 训练 8 个参考模型，在自己的数据上选一次 a，三种方法都用同一个攻击器。成员是 D_train 的 2,000 条，非成员是 D_test 的 2,000 条。另外报告损失攻击作为基线。

## 目录

```
adaptive_dp/          实验代码
  data.py             读取与校验 Adult、预处理、属性分组、六段划分
  model.py            MLP（d_in → 512 → 256 → 2）
  train.py            训练函数（所有模型共用）
  perturb.py          属性加噪：广义随机响应 / Laplace
  rmia.py             离线 RMIA 分数、AUC、TPR@FPR
  defender.py         代理攻击、属性干预、ΔMIA / ΔAcc
  allocate.py         预算分配 w_j，以及按 ε_total 拆分预算
  attack.py           攻击者参考模型、选 a、离线 RMIA、损失攻击
  run.py              命令行入口，写 results.csv 和 allocation.csv
tests/                13 个单元测试
data/raw/adult/       UCI Adult 原始文件
ACSEmployment/        folktables ACSEmployment（加州 2018）上的同一实验
Cardio/               Cardio 数据集上的同一实验
figure/               折线图及绘图脚本 plot_results.py
results/              实验结果（含早期设置的结果，见文末）
AGENT.md              最初的实验规格
env.sh                本集群 GH200 计算节点的环境设置
reproduce.sbatch      一键复现 Adult 主实验（Slurm）
```

## 环境

- 需要 Python ≥ 3.10，以及 `torch`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`pytest`；画图另需 `matplotlib`。
- GPU 不是必需的。参考结果是在 1 块 GH200 GPU 上跑出来的，每个预算（10 个种子）约 25 秒。
- **本集群**：`source env.sh`。它会加载 PyTorch 模块，并通过 `PYTHONPATH` 借用 `venv-causal-gh200` 里的 scikit-learn。这个脚本只能在 GH200 计算节点（aarch64）上用，在 x86 登录节点上不能用。
- **其他机器**：装好上面列出的依赖即可。代码只用相对于本目录的路径。
- **参考环境**（`env.sh` 加载的版本）：Python 3.13.5、torch 2.9.1（CUDA 12.9）、numpy 2.3.2、scikit-learn 1.9.1、pandas 2.3.1、scipy 1.16.1。

## 数据

`data/raw/adult/` 里应有 UCI Adult 的三个原始文件，来源：<https://archive.ics.uci.edu/dataset/2/adult>。

| 文件 | md5 |
| --- | --- |
| `adult.data` | `5d7c39d7b8804f071cdd1f2a7c460872` |
| `adult.test` | `35238206dfdf7f1fe215bbb874adecdc` |
| `adult.names` | `1a7cdb3ff7a1b709968b1c7a11def63e` |

可以用 `md5sum data/raw/adult/*` 核对。程序读入后会检查：合并后共 48,842 行，14 个属性加 1 个标签，标签只有 `<=50K` 和 `>50K` 两种取值。任何一项不符就报错停止。删掉 `fnlwgt` 和 `education` 后剩 12 个属性，编码后 95 维。

## 复现

### 方式一：Slurm 一键运行

```bash
sbatch reproduce.sbatch
```

先跑单元测试，再依次跑 ε_total = 2、4、8、16、32、50、64、70、80、90、100，每档 10 个种子，结果写到 `results/main/epstotal<T>/`，日志写到 `reproduce_<jobid>.log`。其他用户需要把脚本里的 `#SBATCH -A` 改成自己的项目账号。

### 方式二：手动运行

在 GH200 计算节点上、项目根目录下：

```bash
source env.sh
python -m pytest -q                                              # 应显示 13 passed
python -m adaptive_dp.run --eps-total 64 --out results/main/epstotal64/
python -B figure/plot_results.py                                 # 重画 figure/ 下的折线图
```

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--eps-total` | 总隐私预算 ε_total，全部分给属性 | 必填 |
| `--seeds` | 目标模型的种子 | `0 1 2 … 9` |
| `--out` | 输出目录 | `results/` |

其余设置固定在代码里：D_train 和 D_test 各取前 2,000 条（`run.py` 的 `N_TRAIN`、`N_TEST`），防御方和攻击者各 8 个参考模型（`defender.K_REF`、`attack.K_ATT`），标签不加噪，攻击为非自适应。

### 输出

| 文件 | 内容 |
| --- | --- |
| `results.csv` | 30 个目标模型（3 种方法 × 10 个种子）：test_acc、train_acc、gap、rmia_auc、rmia_tpr1、rmia_tpr01、loss_auc，以及攻击者选出的 a |
| `allocation.csv` | 每个属性的 ΔMIA、ΔAcc、w_j，以及 uniform 和 adaptive 各自的 ε_j |

终端会打印 10 个种子的均值 ± 标准差。

## 预期结果（Adult）

参考结果在 `results/n2000_k8_total_10seeds_labelclean/nonadaptive/`。表中是 10 个种子的均值；每个预算下较好的值加粗，† 表示 adaptive 与 uniform 的配对 t 检验 p < 0.05。

**不加噪**：准确率 83.71%，攻击 AUC 0.5924，TPR@1%FPR 4.61%。

| ε_total | 准确率 uniform (%) | 准确率 adaptive (%) | AUC uniform | AUC adaptive | TPR uniform (%) | TPR adaptive (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 70.54 | **70.56** | 0.5000 | 0.5000 | 0.98 | **0.97** |
| 4 | 73.03 | **73.53**† | 0.5018 | 0.5018 | 1.02 | **0.95** |
| 8 | 74.15 | **75.27** | 0.5018 | **0.5012** | **0.93** | 1.00 |
| 16 | **77.97** | 77.51 | 0.5014 | 0.5014 | **0.99** | 1.00 |
| 32 | 79.46 | **79.86** | 0.5163 | **0.5104**† | 1.22 | **1.02** |
| 50 | **81.15** | 81.11 | 0.5456 | **0.5228**† | 2.01 | **1.43**† |
| 64 | 82.13 | **82.16** | 0.5650 | **0.5392**† | 2.60 | **2.20**† |
| 70 | **82.24** | 82.22 | 0.5715 | **0.5496**† | 3.02 | **2.40**† |
| 80 | 82.43 | **82.77** | 0.5789 | **0.5616**† | 3.64 | **2.98**† |
| 90 | 82.60 | **82.96**† | 0.5833 | **0.5739**† | 3.98 | **3.36**† |
| 100 | 82.79 | **83.04** | 0.5831 | **0.5791**† | 3.90 | **3.79** |

防御方分析与 ε_total 无关：不干预时代理攻击 AUC 为 0.5738；w_j 最小为 0.51（workclass），最大为 1.62（education-num）。攻击者选出的 a = 0.6。

**可复现性**：2026-11-26 用当前代码和 `env.sh` 环境重跑了 ε_total = 16、64，`results.csv` 和 `allocation.csv` 与参考结果逐字节一致。换用其他版本的 torch，或换成 CPU、其他型号的 GPU 时，浮点运算会有细微差别，逐种子的数字会和上表不同。

## 注意事项

- **隐私保证**：标签不加噪，形式化保证只覆盖属性（属性级 LDP）；ε_total = 32–100 的形式化保证本身也很弱，所以本实验应理解为针对成员推断的实证防御。
- **预算分配依赖私有数据**：w_j 是在 D_train 训练的模型上算出来的，这一步的泄露没有计入 ε_total。
- 每档 10 个种子；只用一种 MLP，训练集 2,000 条。
- TPR@0.1%FPR（`rmia_tpr01`）只对应 2,000 个非成员中的 2 个误报点，数值很不稳定，上表没有列出。

## 与 AGENT.md 的差异

- **防御方分析**：不再用 4 个校准模型轮换取平均（§6.1–6.2），改为分析 train(D_train, seed 0)，代理攻击为离线 RMIA，8 个参考模型。§10.7 的折半 Spearman 检查不再适用。
- **样本量**：D_train、D_test 各 2,000 条（AGENT.md 为 7,000）；参考模型也用 2,000 条训练。
- **预算**：标签不加噪；用总预算 ε_total 指定，全部平均分给属性（AGENT.md 为 ε̄ = 3、ε_y = 3）。
- **攻击**：非自适应攻击，8 个参考模型（AGENT.md 为按方法加噪的 2 个参考模型）。
- **精简**：没有冒烟测试（§11）、运行时阈值检查（§10.6–10.8）和 `summary.md` 等报告文件（§8.3）。
- **早期设置的结果**：`results/` 下的 `epsbar*`、`nonadaptive`、`z_unchanged`、`n2000_k8`、`n2000_k8_total`、`n2000_k8_total_10seeds` 等目录，以及 `figure/*_labelnoised.*`，来自早期设置（标签加噪、自适应攻击、按 ε̄ 设预算等）。当前代码不再生成它们；对应的代码可以在 git 历史中找到。

## 实现细节

- **随机数**：
  - 数据划分、各种抽样和加噪都用 `numpy.random.default_rng(seed)`；模型初始化用 `torch.manual_seed(seed)`，打乱数据用以 seed 初始化的 `torch.Generator`。
  - 防御方：P_def 按种子 2000 的排列分成 N 和 R；参考模型 k（k = 1…8）的训练子集抽样和训练都用种子 2000 + k。
  - 攻击者：参考模型 i（i = 1…8）的训练子集抽样和训练都用种子 4000 + i；选 a 只用参考模型 1（临时目标）和 2（参考），成员和非成员各 1,000 条，用种子 4000 抽取，AUC 并列时取较小的 a。
  - 目标模型的加噪种子和训练种子都是 seed。
- **加噪顺序**：同一个生成器依次处理属性 j = 1, 2, …。广义随机响应的类别数 k_j 按完整类别表计算（adult.names 中的类别加 `Unknown`），包括数据里没有出现过的类别。
- **数值精度**：Pr(x | m) 先把 float32 logits 转成 float64，再做 softmax。
- **数据来源**：`data/raw/adult/` 里的文件复制自本机已有的 UCI 原始文件，没有重新下载。
