# AGENT.md — 因果自适应加噪 DP-SGD：在 Purchase-100 上的最小验证

## 1. 目标

实现一种"按因果分数自适应分配噪声"的 DP-SGD，并在 Purchase-100 上与均匀 DP-SGD 对比，回答一个问题：

> 在**相同的 (ε, δ)** 下，自适应加噪相比均匀加噪，测试精度（效用）和成员推断攻击成功率（隐私）各变化了多少？

交付物：

- 可运行的代码；
- 通过的单元测试；
- `results/results.csv`；
- `results/summary.md`。

## 2. 执行规则

1. 严格按本文件的定义、超参数和流程执行，不自行增加方法、数据集或指标。
2. 不在 D_train 或 D_test 上调任何超参数。本文件给的超参数都是固定值。
3. 遇到本文件没有覆盖的情况，或者某一步无法按描述完成，先停下来问，不要猜。
4. 任何偏离都写进 `results/summary.md` 的"偏离记录"一节。
5. 先跑冒烟测试（§10），通过后再跑完整实验。
6. 第 9 节的检查不通过时，先修实现，不要跳过。
7. 不要为了让结果更好看而改动任何设置。结果不利也照实报告。

## 3. 环境

- Python ≥ 3.10。
- 依赖：`torch ≥ 2.1`（需要 `torch.func`）、`numpy`、`scikit-learn`、`opacus`（**只用于隐私核算**）、`pandas`。
- 缺依赖时先问用户，不要换成别的库。
- 建议用 GPU；没有 GPU 也能跑，只是慢。

## 4. 数据

- **文件**：`data/raw/purchase/dataset_purchase`。这是逗号分隔文本：第 1 列为标签（1–100），第 2–601 列为 600 个二值特征。
  - 文件不存在时，尝试从 GitHub `privacytrustlab/datasets` 获取 Purchase-100（通常是 `dataset_purchase.tgz`）。
  - 获取失败就停下来，请用户把文件放到上述路径。
  - 实际格式与描述不同时，按实际格式写载入代码，并记录在偏离记录里。
- **载入后检查**（不通过就停）：
  - 共 197,324 行；
  - 特征 600 列，取值只有 {0, 1}；
  - 标签恰好有 100 个不同取值，统一转成 0–99。
- **划分**：用种子 0 生成一个随机排列，依次切出三段不重叠的子集，其余数据不用。下标保存到 `results/splits.npz`。

  | 子集 | 大小 | 用途 |
  | --- | --- | --- |
  | D_train | 20,000 | 训练目标模型，也是成员推断里的"成员" |
  | D_test | 20,000 | 测精度，也是成员推断里的"非成员" |
  | D_aux | 20,000 | 计算因果分数、训练攻击用的参考模型；视为公开数据 |

- **特征频率**：π_j = 特征 j 在 D_aux 上取 1 的比例（j = 1…600）。

## 5. 模型与非私有训练

- **模型**：`MLP`，结构为 `fc1: 600→512`、tanh、`fc2: 512→256`、tanh、`fc3: 256→100`，输出 logits。层名必须是 `fc1`、`fc2`、`fc3`，其中 `fc1.weight` 的形状为 [512, 600]。
- **非私有训练 `train_nonprivate`**：
  - Adam，lr = 1e-3；
  - batch = 256，随机打乱；
  - 30 个 epoch；
  - 交叉熵损失，无正则。
- **两个记号**（z 为 logits，y 为真实标签）：
  - `φ(x, y) = z_y − logsumexp_{c≠y} z_c`，即 log(p_y / (1 − p_y)) 的数值稳定写法；
  - `u(x, y) = z_y − logsumexp_c z_c`，即 log p_y。

## 6. 因果分数与噪声分配

这一节只使用 D_aux，不消耗隐私预算。

### 6.1 代理模型

1. 用种子 0 把 D_aux 随机对半分成 H1、H2，各 10,000 条。
2. 训练两个模型对，都用 `train_nonprivate`：
   - 对 1：f_A 在 H1 上训练，f_ref 在 H2 上训练，成员集 M = H1，非成员集 N = H2；
   - 对 2：两半角色互换。

### 6.2 分数定义

记号 `x^(j←a)` 表示把 x 的第 j 维置为 a ∈ {0, 1}，其余维不变。对每个模型对、每个特征 j：

```
g(x, y) = φ_{f_A}(x, y) − φ_{f_ref}(x, y)

G_j(a)  = mean over (x,y) ∈ M of g(x^(j←a), y)
V_j     = mean over M of g(x, y)  −  [ π_j · G_j(1) + (1 − π_j) · G_j(0) ]

H_j(a)  = mean over (x,y) ∈ N of u_{f_A}(x^(j←a), y)
U_j     = mean over N of u_{f_A}(x, y)  −  [ π_j · H_j(1) + (1 − π_j) · H_j(0) ]
```

- V_j 是风险分：把特征 j 按边缘分布随机化后，成员差距下降多少。
- U_j 是效用分：同样随机化后，非成员上的对数似然下降多少。
- 最终的 V 和 U 取两个模型对的平均。

### 6.3 计算方法（必须这样做）

第一层是线性的，所以对每个模型先计算 `Z = X · fc1.weightᵀ + fc1.bias`。

- 对特征 j 和取值 a，干预后的第一层预激活为 `Z + (a − X[:, j]) ⊗ fc1.weight[:, j]`。
- 只需从 tanh 开始跑后面几层。
- 按特征分块，用批量计算加速。

### 6.4 后处理

1. 计算 V⁺ = max(V, 0) 和 U⁺ = max(U, 0)。
2. 分别除以各自的均值，使两者的均值都为 1。如果某一项全为 0，停下来报告。
3. 后文的 U 和 V 都指这一步处理后的值。

### 6.5 分配 `allocate(U, V, λ=1.0, ε_min=0.1)`

求 ε_1…ε_600，满足：

```
ε_j = max( ε_min, ( 2·U_j / (λ·V_j + ν) )^(1/3) ),    且  Σ_j ε_j = 600
```

- 用二分法求 ν > 0：在对数空间 [1e-12, 1e12] 上二分 200 次，使总和等于 600，相对误差小于 1e-6。总和随 ν 单调递减。
- U_j = 0 的特征自动得到 ε_min。

### 6.6 裁剪系数与输出

- 各特征的裁剪系数为 C_j = κ / ε_j，κ = 1.0。
- 输出 `results/allocation.csv`，列为 `j, pi, U, V, eps, C`。
- 在 `summary.md` 里报告：
  - ε_j 的最小值、中位数、最大值；
  - U 最高的 10 个特征和 V 最高的 10 个特征；
  - Spearman(V, π) 和 Spearman(V, U)。

## 7. DP-SGD（均匀与自适应共用同一实现）

**函数** `train_dpsgd(D, C, κ, σ, q, T, seed)`，其中 C 是长度为 600 的向量。

- 均匀 DP-SGD：C_j = κ，对所有 j。
- 自适应 DP-SGD：用 §6.6 的 C。

**固定超参数**：

- q = 512 / 20000，即期望 batch B = q·N = 512；
- T = ceil(30 / q)，即约 30 个 epoch；
- Adam，lr = 1e-3；
- δ = 1e-5。

**σ 的确定**：用 Opacus 的 RDP 核算求 σ：`opacus.accountants.utils.get_noise_multiplier(target_epsilon=ε, target_delta=δ, sample_rate=q, steps=T, accountant="rdp")`。

- 求出 σ 后，用 `opacus.accountants.RDPAccountant` 按 (σ, q) 调用 T 次 `step`，再用 `get_epsilon(δ)` 复核。
- 复核结果必须落在 [ε − 0.05, ε] 内。
- 记录 σ 和复核后的 ε。

**每一步 t = 1…T：**

1. **采样**：Poisson 采样，D 中每条记录独立以概率 q 入选。批为空时第 2–4 步的和记为 0，照样执行第 5–8 步。
2. **逐样本梯度**：用 `torch.func`（`functional_call` + `vmap(grad(...))`）计算每条样本交叉熵对所有参数的梯度。为控制显存，按 128 条一个微批计算，并逐微批累加第 4 步的结果。逐样本裁剪本身是精确的，分微批不影响结果。
3. **缩放**：把 `fc1.weight` 的逐样本梯度（形状 [b, 512, 600]）按列除以 C，即第 j 列除以 C_j。其余所有参数的梯度（包括 `fc1.bias`）都除以 κ。
4. **裁剪与求和**：对每个样本 i，计算缩放后全部参数梯度拼接起来的 L2 范数 n_i，乘以 1 / max(1, n_i)，然后对批内求和。
5. **加噪**：给求和结果的每个元素加独立的 N(0, σ²) 噪声。随机数生成器用 seed 固定。
6. **映射回原空间**：`fc1.weight` 按列乘以 C，其余参数乘以 κ。
7. **归一化**：除以期望 batch B（不是实际 batch 大小）。
8. **更新**：把结果写进 `.grad`，执行一次 Adam 更新。

当所有 C_j = κ 时，这个过程与标准 DP-SGD（裁剪阈值 κ、噪声标准差 σκ）完全等价。对任意 C，隐私保证都相同，只由 σ、q、T 决定。

## 8. 实验与评估

### 8.1 运行组合

| 方法 | ε | 种子 |
| --- | --- | --- |
| nonprivate | — | 0, 1, 2 |
| uniform | 2, 4, 8 | 0, 1, 2 |
| adaptive | 2, 4, 8 | 0, 1, 2 |

- 种子控制模型初始化、Poisson 采样和噪声；数据划分始终固定为种子 0。
- 每一次运行都训练两个模型：
  - **目标模型**：在 D_train 上训练；
  - **参考模型**：在 D_aux 上用**完全相同的方法和设置**训练，包括相同的 σ、q、T 和 C，种子为 seed + 100。参考模型只供校准攻击使用。

### 8.2 指标

成员集为 D_train 全部 20,000 条，非成员集为 D_test 全部 20,000 条。

- **效用**：目标模型在 D_test 和 D_train 上的精度，以及两者之差。
- **攻击 1：损失攻击**。分数取目标模型交叉熵损失的相反数，分数越高越像成员。
- **攻击 2：参考校准攻击**。分数为 φ_target(x, y) − φ_ref(x, y)。
- **两种攻击都报告**：
  - AUC；
  - TPR@1%FPR 和 TPR@0.1%FPR。用 `sklearn.metrics.roc_curve` 计算，TPR@α 取所有满足 fpr ≤ α 的点中最大的 tpr。

### 8.3 输出

**`results/results.csv`**：每次运行一行，共 21 行。列为：

`method, eps_target, eps_achieved, sigma, seed, test_acc, train_acc, gap, loss_auc, loss_tpr1, loss_tpr01, cal_auc, cal_tpr1, cal_tpr01`

**`results/summary.md`** 包含：

1. **汇总表**：每个（方法，ε）一行，各指标给出 3 个种子的均值 ± 标准差。
2. **对比表**：对每个 ε，给出 adaptive 减 uniform 的差值，包括 Δtest_acc、Δcal_tpr1、Δcal_auc 的均值，并标出该差值的绝对值是否大于两组标准差之和。
3. **分配信息**：§6.6 要求报告的内容。
4. **结论**：最多 3 句话，只陈述数字，不做推测。
5. **偏离记录**。

## 9. 检查（不通过就停下来修）

1. 数据载入检查（§4）。
2. **单元测试 `tests/test_intervention.py`**：随机抽 10 个特征，比较 §6.3 的快速计算和朴素计算（真的构造 x^(j←a) 再完整前向）得到的 G_j(a)，最大绝对误差应小于 1e-4。
3. **单元测试 `tests/test_weighted_clip.py`**：在一个 8 条样本的小批上，取 C 全为 κ，并用同一组噪声，比较 §7 第 2–7 步的结果和标准 DP-SGD 公式（逐样本裁剪到 κ、求和、加 N(0, σ²κ²)、除以 B）的结果，最大绝对误差应小于 1e-5。
4. **单元测试 `tests/test_allocate.py`**：
   - Σε_j 与 600 的相对误差小于 1e-6，且 min ε_j ≥ ε_min；
   - λ = 0 且所有 U_j > 0 时，ε_j 与 U_j^(1/3) 成正比（相对误差小于 1e-4）。
5. **核算复核**：§7 的 ε 复核通过。
6. **非私有模型的信号检查**：nonprivate 的 loss_auc 应大于 0.55。否则说明成员信号太弱或实现有误，停下来报告。
7. **DP 训练是否崩溃**：如果 ε = 8 时 uniform 的 test_acc 低于 10%，停下来报告，不要自行调 lr。

## 10. 运行方式

### 代码结构

```
adaptive_dp/
  data.py       载入与划分
  model.py      MLP
  train.py      train_nonprivate、train_dpsgd
  scores.py     代理模型、V 和 U 的计算
  allocate.py   allocate
  attacks.py    两种攻击与 TPR@FPR
  run.py        命令行入口，串起全部流程
tests/
  test_intervention.py
  test_weighted_clip.py
  test_allocate.py
results/
```

### 执行顺序

1. 运行 `pytest tests/`，全部通过。
2. **冒烟测试**：`python -m adaptive_dp.run --smoke`。
   - 设置：每个子集只取 2,000 条，epoch 降到 2，只跑 ε = 8 和种子 0。
   - 要求：10 分钟内端到端跑通，并生成三个结果文件。
3. **完整实验**：`python -m adaptive_dp.run --eps 2 4 8 --seeds 0 1 2 --out results/`。
4. 检查 §9 的第 5–7 项，确认 `summary.md` 已完整生成。

### 完成标准

- 三个单元测试全部通过；
- `results/` 中有 `splits.npz`、`allocation.csv`、`results.csv`（21 行）和 `summary.md`；
- 所有 DP 运行的 ε 复核都通过。

## 11. 可选（完成标准达成后、且用户同意时再做）

增加方法 `adaptive_u_only`：与 adaptive 相同，但 λ = 0，即只按效用分配，相当于 AdLM 的思路。用它检验风险分 V 是否带来了额外收益。
