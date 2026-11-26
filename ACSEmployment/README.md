# ACSEmployment：因果归因自适应输入端加噪（非自适应攻击）

把当前 Adult 主实验流程原样搬到 folktables 的 ACSEmployment 任务上（加州，ACS 2018 1-Year 个人数据）。只换了数据读取和预处理，其余步骤（防御方分析、预算分配、输入端加噪、训练、攻击）直接调用 `adaptive_dp/` 里的代码，没有任何改动。设定与 Adult 主实验（`python -m adaptive_dp.run`）相同：标签不加噪，预算全部分给属性；非自适应攻击，三种方法都用同一组参考模型（在不加噪的 P_att 上训练）和同一个 a。

## 设定（与 Adult 一致）

| 项目 | 取值 |
| --- | --- |
| 六段划分 | train 7,000 / test 7,000 / P_def 14,000 / Z_def 2,000 / P_att 14,000 / Z_att 2,000，划分种子 0（`adaptive_dp.data.make_splits`，在去重后的 201,780 条上划分） |
| 实际使用 | 取 D_train、D_test 各前 2,000 条（`adaptive_dp/run.py` 的 `N_TRAIN`、`N_TEST`） |
| 防御方 | 代理离线 RMIA，8 个参考模型，每个用 P_def 另一半中的 2,000 条训练 |
| 攻击者 | 非自适应离线 RMIA（γ = 2），8 个参考模型，每个用 P_att 中的 2,000 条训练；a 在攻击者自己的数据上选 |
| 预算 | `--eps-total T`，T ∈ {2, 4, 8, 16, 32, 50, 64, 70, 80, 90, 100}，全部平均分给 16 个属性：ε̄ = T/16；标签不加噪 |
| 种子 | 0–9（10 个） |
| 模型与训练 | MLP（99 → 512 → 256 → 2），Adam lr 1e-3，batch 256，30 epoch |

注意：属性个数不同，同一 ε_total 下每属性的预算也不同：Adult 12 个属性（ε̄ = T/12），Cardio 11 个（T/11），ACSEmployment 16 个（T/16）。

## 数据

- 来源：美国人口普查局 ACS PUMS，就是 folktables 的 `ACSDataSource(survey_year='2018', horizon='1-Year', survey='person').get_data(states=['CA'])` 下载的文件。
  - `data/raw/csv_pca.zip`（内含 `psam_p06.csv`），md5 `2d87384683b319f0b2f5901d1eda569c`，<https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip>
  - `data/raw/PUMS_Data_Dictionary_2018.csv`，md5 `b7bf7eb7632ca40165aba018d5ce969a`，<https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.csv>
- 任务完全按 `folktables.ACSEmployment` 的定义：
  - 16 个特征：AGEP, SCHL, MAR, RELP, DIS, ESP, CIT, MIG, MIL, ANC, NATIVITY, DEAR, DEYE, DREM, SEX, RAC1P；
  - 标签为 ESR == 1（“平民，在职且在工作”），ESR 缺失时记为 0；
  - 不做行筛选，缺失值填 0。folktables 代码里写的是 `np.nan_to_num(x, -1)`，但这里 −1 传给的是 numpy 的 `copy` 参数，所以实际填的是默认值 0。有缺失值的属性里 0 都不是合法取值（读取时会检查），所以 0 就是单独的 N/A 类别。
- 已核对：去重前的特征矩阵和标签与 folktables 0.0.12 的 `ACSEmployment.df_to_numpy` 输出逐值一致（378,817 × 16）。
- 读入后检查：两个文件的 md5、行数 378,817、全部是个人记录、AGEP 无缺失、所有类别取值都在数据字典里、one-hot 正确、去重后编码记录唯一。任何一项不符就报错。

### 去重

原始的 378,817 条中，有 177,037 条与另一条记录完全相同（16 个特征加标签都一样），占 47%，其中 46,871 条是 16 岁以下的儿童。完全相同的记录可能同时落进成员集和非成员集，攻击无法区分，所以在划分之前去重：每条不同的 (x, y) 只保留文件中第一次出现的那条，剩 201,780 条，阳性率 45.88%。去重是本实验加的，folktables 原任务没有这一步。

### 预处理

属性类型按 2018 PUMS 数据字典确定：

- **AGEP**（年龄）：数值型（N），作为连续属性，用 P_def 的 min/max（0 和 94）缩放到 [0, 1]。
- **其余 15 个**：字符编码型（C），作为类别属性做 one-hot。类别表取自数据字典，N/A 记为 0，包括数据里没出现的类别（ANC 的 8，“2018 年部分 PUMA 隐去”），广义随机响应的 k_j 按完整类别表计算。

| 属性 | k | 属性 | k | 属性 | k | 属性 | k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCHL | 25 | ESP | 9 | ANC | 5 | DREM | 3 |
| RELP | 18 | CIT | 5 | MIG | 4 | DIS, NATIVITY, DEAR, DEYE, SEX | 2 |
| RAC1P | 9 | MAR | 5 | MIL | 5 | | |

编码后共 99 维。

## 文件

```
ACSEmployment/
  acs_data.py         读取、校验、ACSEmployment 特征提取、去重、预处理、六段划分；prepare() 与 adaptive_dp.data.prepare() 接口相同
  run_acs.py          把 adaptive_dp.run 的数据入口换成 acs_data.prepare，然后执行原流程
  summarize.py        汇总 10 个种子的均值 ± 标准差，以及 adaptive − uniform 的配对比较（与 Cardio 的相同）
  run_acs.sbatch      一键复现（Slurm）
  data/raw/           Census 原始文件和 2018 数据字典
  results/
    run_acs_5761077.log              完整运行日志（含逐种子结果）
    nonadaptive_labelclean/          当前设置（标签不加噪）：epstotal<T>/、summary.csv、comparison.csv
    nonadaptive/epstotal<T>/         早期的标签加噪运行：results.csv（30 个目标模型）、allocation.csv（16 个属性）
    nonadaptive/summary.csv          每个 ε_total、每种方法的均值和标准差
    nonadaptive/comparison.csv       adaptive − uniform 的配对差值、p 值、胜出种子数
```

## 复现

在项目根目录下：

```bash
sbatch ACSEmployment/run_acs.sbatch
```

或在 GH200 计算节点上手动运行：

```bash
source env.sh
python -B ACSEmployment/acs_data.py      # 打印数据检查和预处理参数
for T in 2 4 8 16 32 50 64 70 80 90 100; do
  python -B ACSEmployment/run_acs.py --eps-total $T --out ACSEmployment/results/nonadaptive_labelclean/epstotal$T/
done
python -B ACSEmployment/summarize.py
```

`run_acs.py` 的参数与 `python -m adaptive_dp.run` 完全相同。运行时不需要安装 folktables。早期的标签加噪结果（`results/nonadaptive/`）于 2026-11-25 跑出，作业 5761077，用时 4 分 44 秒。标签不加噪的结果（`results/nonadaptive_labelclean/`）也于 2026-11-25 跑出；2026-11-26 用当前代码重跑 ε_total = 64，`results.csv` 和 `allocation.csv` 与之逐字节一致。

## 结果（非自适应攻击）

> 本节各表来自早期的标签加噪运行（ε̄ = ε_y = T/17，`results/nonadaptive/`），当前代码不再生成。当前设置（标签不加噪）的结果在 `results/nonadaptive_labelclean/`：运行 `python -B ACSEmployment/summarize.py` 可打印汇总表，折线图见 `figure/acs_employment.png`。

10 个种子的均值 ± 样本标准差。test_acc 和 TPR@1%FPR 的单位是 %。成员是 D_train 的 2,000 条，非成员是 D_test 的 2,000 条。攻击者选出的 a 在所有设置下都是 0.0。nonprivate 不加噪，所有 ε_total 下结果相同。

| ε_total | 方法 | test_acc | rmia_auc | TPR@1%FPR | loss_auc |
| --- | --- | --- | --- | --- | --- |
| ∞ | nonprivate | 72.30 ± 0.53 | 0.7110 ± 0.0146 | 4.68 ± 0.70 | 0.6240 ± 0.0065 |
| 2 | uniform | 51.22 ± 3.15 | 0.5038 ± 0.0037 | 1.21 ± 0.35 | 0.5009 ± 0.0035 |
| 2 | adaptive | 51.15 ± 3.56 | 0.5041 ± 0.0063 | 1.30 ± 0.41 | 0.5010 ± 0.0055 |
| 4 | uniform | 51.47 ± 3.96 | 0.5030 ± 0.0066 | 1.30 ± 0.45 | 0.5012 ± 0.0057 |
| 4 | adaptive | 51.01 ± 4.08 | 0.5030 ± 0.0077 | 1.29 ± 0.39 | 0.5010 ± 0.0072 |
| 8 | uniform | 52.58 ± 3.19 | 0.5005 ± 0.0058 | 1.17 ± 0.26 | 0.4964 ± 0.0048 |
| 8 | adaptive | 52.07 ± 2.82 | 0.5013 ± 0.0039 | 1.28 ± 0.38 | 0.4981 ± 0.0042 |
| 16 | uniform | 54.47 ± 2.02 | 0.5016 ± 0.0096 | 1.18 ± 0.37 | 0.4971 ± 0.0094 |
| 16 | adaptive | 54.79 ± 2.03 | 0.5037 ± 0.0112 | 1.23 ± 0.33 | 0.4987 ± 0.0107 |
| 32 | uniform | 61.27 ± 2.63 | 0.5142 ± 0.0076 | 1.43 ± 0.37 | 0.5071 ± 0.0078 |
| 32 | adaptive | 61.59 ± 2.02 | 0.5104 ± 0.0097 | 1.26 ± 0.39 | 0.5019 ± 0.0091 |
| 50 | uniform | 66.15 ± 1.09 | 0.5494 ± 0.0068 | 1.49 ± 0.22 | 0.5325 ± 0.0057 |
| 50 | adaptive | 66.62 ± 1.00 | 0.5300 ± 0.0078 | 1.33 ± 0.28 | 0.5169 ± 0.0087 |
| 64 | uniform | 68.05 ± 1.27 | 0.5953 ± 0.0066 | 2.31 ± 0.34 | 0.5639 ± 0.0059 |
| 64 | adaptive | 68.38 ± 1.40 | 0.5518 ± 0.0077 | 1.85 ± 0.26 | 0.5330 ± 0.0083 |
| 70 | uniform | 68.67 ± 0.98 | 0.6142 ± 0.0067 | 2.59 ± 0.37 | 0.5763 ± 0.0059 |
| 70 | adaptive | 69.01 ± 0.80 | 0.5635 ± 0.0072 | 1.83 ± 0.31 | 0.5377 ± 0.0073 |
| 80 | uniform | 69.41 ± 0.92 | 0.6449 ± 0.0050 | 3.35 ± 0.61 | 0.5942 ± 0.0035 |
| 80 | adaptive | 69.81 ± 0.86 | 0.5845 ± 0.0061 | 2.29 ± 0.34 | 0.5531 ± 0.0041 |
| 90 | uniform | 69.83 ± 0.63 | 0.6734 ± 0.0068 | 3.72 ± 0.75 | 0.6113 ± 0.0064 |
| 90 | adaptive | 69.95 ± 0.88 | 0.6060 ± 0.0075 | 2.76 ± 0.56 | 0.5646 ± 0.0041 |
| 100 | uniform | 70.40 ± 0.95 | 0.6875 ± 0.0084 | 3.66 ± 0.65 | 0.6190 ± 0.0054 |
| 100 | adaptive | 70.09 ± 0.81 | 0.6244 ± 0.0074 | 3.23 ± 0.62 | 0.5755 ± 0.0045 |

### adaptive − uniform（按种子配对）

p 为配对 t 检验的 p 值，A>U 为 adaptive 高于 uniform 的种子数（共 10 个）。

| ε_total | Δtest_acc (pp) | p | A>U | Δrmia_auc | p | A>U | ΔTPR@1% (pp) | p | A>U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | −0.07 | 0.888 | 5/10 | +0.0003 | 0.787 | 6/10 | +0.08 | 0.455 | 5/10 |
| 4 | −0.46 | 0.334 | 4/10 | −0.0000 | 0.981 | 5/10 | −0.00 | 0.945 | 5/10 |
| 8 | −0.51 | 0.446 | 5/10 | +0.0008 | 0.561 | 6/10 | +0.11 | 0.457 | 7/10 |
| 16 | +0.32 | 0.619 | 5/10 | +0.0022 | 0.245 | 5/10 | +0.06 | 0.426 | 7/10 |
| 32 | +0.32 | 0.609 | 7/10 | −0.0038 | 0.107 | 3/10 | −0.16 | 0.101 | 2/10 |
| 50 | +0.48 | 0.061 | 8/10 | −0.0194 | <0.001 | 0/10 | −0.16 | 0.153 | 3/10 |
| 64 | +0.33 | 0.493 | 5/10 | −0.0435 | <0.001 | 0/10 | −0.46 | 0.007 | 1/10 |
| 70 | +0.34 | 0.343 | 5/10 | −0.0507 | <0.001 | 0/10 | −0.76 | 0.001 | 1/10 |
| 80 | +0.41 | 0.192 | 7/10 | −0.0604 | <0.001 | 0/10 | −1.06 | <0.001 | 0/10 |
| 90 | +0.11 | 0.635 | 6/10 | −0.0674 | <0.001 | 0/10 | −0.96 | 0.006 | 0/10 |
| 100 | −0.30 | 0.510 | 4/10 | −0.0631 | <0.001 | 0/10 | −0.42 | 0.081 | 3/10 |

### 防御方分析与预算分配

防御方分析与 ε_total 无关，11 次运行的 `allocation.csv` 中 ΔMIA、ΔAcc、w_j 完全相同。不干预时，代理攻击的 AUC 为 0.7021，目标模型在 N 上的准确率为 69.89%。

| j | 属性 | ΔMIA | ΔAcc | w_j |
| --- | --- | --- | --- | --- |
| 1 | AGEP | −0.0292 | −0.0647 | 1.935 |
| 2 | SCHL | −0.1217 | −0.0221 | 0.877 |
| 3 | MAR | −0.0682 | −0.0073 | 0.760 |
| 4 | RELP | −0.1228 | −0.0476 | 1.116 |
| 5 | DIS | −0.0441 | −0.0338 | 1.384 |
| 6 | ESP | −0.1302 | −0.1282 | 1.514 |
| 7 | CIT | −0.0693 | −0.0023 | 0.570 |
| 8 | MIG | −0.0635 | −0.0257 | 1.133 |
| 9 | MIL | −0.1004 | −0.0687 | 1.342 |
| 10 | ANC | −0.0714 | −0.0001 | 0.423 |
| 11 | NATIVITY | −0.0191 | +0.0006 | 0.622 |
| 12 | DEAR | −0.0299 | +0.0006 | 0.545 |
| 13 | DEYE | −0.0383 | −0.0079 | 0.932 |
| 14 | DREM | −0.0714 | −0.0669 | 1.484 |
| 15 | SEX | −0.0398 | −0.0095 | 0.969 |
| 16 | RAC1P | −0.0863 | +0.0002 | 0.392 |

## 观察

- **nonprivate 的泄露比 Adult、Cardio 都大**：训练准确率 95.9%，测试准确率 72.3%，差 23.6 个百分点；RMIA AUC 0.711，TPR@1%FPR 4.68%。
- **低预算（ε_total ≤ 16）时模型基本失效**：test_acc 在 51%–55%，不高于全猜多数类的水平（测试集阳性率约 45%，即多数类约 55%）。RMIA AUC 约 0.50，两种方法没有可分辨的差别。
- **ε_total ≥ 50 时，adaptive 在准确率不变的情况下显著降低了 MIA**：
  - rmia_auc 低 0.019–0.067，10 个种子全部更低，p < 0.001；
  - ε_total = 64–90 时 TPR@1%FPR 也低 0.46–1.06 个百分点，p < 0.01；loss_auc 同样更低；
  - test_acc 与 uniform 的差都在 ±0.5 个百分点以内，没有一档显著；
  - 例如 ε_total = 80 时，adaptive 的准确率为 69.81%，RMIA AUC 为 0.585；uniform 要把 AUC 压到 0.585 以下，得降到 ε_total = 50（AUC 0.549），这时准确率只有 66.15%。
- **原因**：ACSEmployment 各属性的 ΔMIA 和 ΔAcc 分化明显。RAC1P、ANC、CIT、NATIVITY、DEAR 几乎不影响准确率（|ΔAcc| < 0.003），其中 RAC1P、ANC、CIT 对 MIA 的效应却不小，这几个属性拿到的预算最少（w ≈ 0.39–0.62）。省下的预算给了 AGEP、ESP、DREM、DIS、MIL 等对准确率重要的属性。uniform 在这些不影响准确率的属性上保留了较多信息，让模型能记住个别成员，所以泛化差距更大：ε_total = 100 时 uniform 为 21.5 个百分点，adaptive 为 13.7。
- 这与 Cardio 不同。Cardio 各属性的 ΔMIA 很接近，分配主要由 ΔAcc 决定，adaptive 表现为准确率更高、MIA 略高。

## 注意事项

- 与 Adult 一样，w_j 是在 D_train 训练出的目标模型上算出来的，分配这一步本身的隐私泄露没有计入 ε_total。
- ε_total 是本地差分隐私预算（δ = 0），不能和 DP-SGD 的 ε 直接比较。
- TPR@0.1%FPR（`rmia_tpr01`）只对应 2,000 个非成员中的 2 个误报点，数值不稳定，所以上表没有列出，原始值在各 `results.csv` 中。
- 只用了加州 2018 年的数据。去重是本实验加的步骤，所以结果不能和 folktables 原任务上的数字直接比较。
