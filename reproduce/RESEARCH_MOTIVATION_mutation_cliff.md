# Mutation Cliff:SKEMPIv2 上的数据分析与 ADiT 失效证据(Research Motivation 草稿)

> 一句话:**在 SKEMPIv2 上,"近乎相同的突变 → ΔΔG 巨变"(mutation cliff)真实且普遍存在,
> 而当前 all-atom ΔΔG 模型(ADiT)在 cliff 上系统性地"抹平"——只预测出约一半的跳变幅度、
> 同位点排序失效。聚合指标(overall Pearson 0.66)完全掩盖了这一失效模式。**

## 1. 背景与定义
化学信息学里的 **activity cliff**:结构高度相似的分子活性却差异巨大,是 QSAR/打分模型的已知失效点。
把它迁移到 **蛋白突变效应(ΔΔG)** 任务:

> **Mutation cliff** = 同一复合物界面上、mutant 序列高度相似(尤其同一位点仅替换氨基酸、或仅差一个突变步)
> 但实验 ΔΔG 差异巨大的一对/一组突变。

## 2. 数据与方法
- 数据:SKEMPIv2,ADiT-S 三折交叉验证的**逐样本池化预测**(`reproduce/skempi_per_sample_pred.csv`,6694 样本)。
- 相似度定义:**突变距离 d** = 两条 mutant 序列不同残基位置数(缺失位点视为 WT)。`d=1` = 差一个突变步
  (同位点换 AA,或加/减一个突变)。
- **cliff 指数 SALI** = `|ΔΔΔG| / d`(每突变步的 ΔΔG 变化,越大越陡)。
- **关键对照——标签噪声下界**:用"完全相同突变的重复测量"的 ΔΔG 离散度,作为噪声基线,证明 cliff 是真实信号。
- 脚本:`mutation_cliff_analysis.py`(单点/同位点)、`mutation_cliff_extended.py`(multi-site/SALI/出图)。

## 3. 发现 1:Mutation cliff 真实存在,远超噪声
| 量(kcal/mol) | 数值 |
|---|---|
| 标签噪声下界(相同突变重复测量 ΔΔG range,498 组) | median **0.30** / mean 0.49 / p90 1.23 |
| 同界面+同位点不同替换 AA 的 ΔΔG range(378 组深扫位点) | median **0.91** / mean 1.58 / **max 12.9** |
| 同位点 range **>2 / >3** 的组占比 | **25.4% / 14.6%** |

→ 同一位点只换一个氨基酸,ΔΔG 能差 **2–13 kcal/mol**,而噪声下界仅 ~0.3。**cliff 是真实且常见的**(图 `fig_cliff_vs_noise.png`:cliff 分布重尾远超噪声)。

## 4. 发现 2:ADiT 系统性"抹平" cliff
- **预测幅度压缩**:同位点内真值 ΔΔG range median 0.91,模型预测 range 仅 0.67。
- **跳变低估(同位点突变对,7933 对)**:真值跳变 vs 预测跳变 Pearson 0.57 / Spearman 0.50;
  cliff 对(|ΔΔΔG真|>2,1745 对)**模型只预测出 ~0.47 倍幅度**,真实大跳变里仅 **40%** 被判为大跳变(>3 时仅 33%)。
- **SALI cliff(高 SALI>2,占 6.3%)**:模型预测/真值跳变比中位 **0.52**。
- **同位点排序失效(k≥3,133 组)**:组内 Spearman median 0.50、**mean 仅 0.37、21% 为负**。
- 图:`fig_jump_true_vs_pred.png`(大跳变点全落在 y=x 下方)、`fig_jump_saturation.png`(预测跳变随真值跳变增大而饱和)。

## 5. 发现 3:single 与 multi-site 近邻都存在 cliff(d=1 细分)
| d=1 子类 | 对数 | \|ΔΔΔG真\| 中位 | p90 | 真>2 占比 | 预测/真值比 |
|---|---|---|---|---|---|
| single 同位点换 AA | 7106 | 0.90 | 3.20 | 21.8% | 0.79 |
| multi 近邻(加/减一个突变) | 2119 | 0.81 | 3.03 | 21.4% | 0.71 |

→ cliff 不是单点突变独有;**多突变背景下"差一个突变步"同样出现大跳变,且模型低估更重**。
(图 `fig_absT_by_distance.png`:即使 d=1,\|ΔΔΔG\| 的大跳变依然存在。)

## 6. 极端实例(同位点、模型几乎完全抹平)
| 界面 | 位点 | A vs B | ddG_A / ddG_B | 真值跳变 | 模型预测跳变 |
|---|---|---|---|---|---|
| 1CHO_EFG_I | I15 | E vs W | +6.76 / −1.69 | **8.45** | **0.34** |
| 1PPF_E_I | I18 | R vs V | +7.18 / −0.49 | 7.68 | −0.31 |
| 1R0R_E_I | I10 | D vs I | +5.23 / −1.80 | 7.03 | −0.07 |
| 2JEL_LH_P | P34 | N vs Q | +6.82 / 0.00 | 6.82 | −0.17 |

这些多是**蛋白酶-抑制剂的 P1 关键位点**(1CHO/1PPF/1R0R 的 I15/I18/I10):同一位点不同替换效应天差地别,模型却预测几乎相同。

## 7. 机制假说(可攻的研究点)
单残基替换下**模型输入几乎不变**(共享 WT 主链结构,仅一个残基的原子/ESM 嵌入变化)→ 表征近乎不变 →
输出近乎恒定 → **cliff 被抹平**。即:**当前 all-atom ΔΔG 模型在函数上偏"平滑",与 ΔΔG landscape 的"尖锐性"本质冲突。**
(注:mt 结构由 FoldX 在固定主链上建模,可能进一步削弱模型对单点替换的敏感度——结构 vs 序列表征对 cliff 的贡献本身值得拆解。)

## 8. 研究方向(motivation 落点)
1. **cliff-aware 表征/损失**:放大单残基替换的表征差异(如 substitution-specific embedding、对比学习把 cliff 对推开),或在损失里加权 cliff 对。
2. **不确定性 / 拒识**:在 cliff 邻域输出高不确定性,而非自信地给平滑值。
3. **评测改革**:把 **per-interface 排序 + cliff 子集指标 + SALI** 纳入标准评测,避免 overall 指标掩盖失效。
4. **序列 vs 结构归因**:量化 FoldX 建模结构对单点敏感度的限制。

## 9. 诚实的局限
- 基于**单次训练**(三折池化、单 seed);严谨需多 seed/多模型,并验证 baselines 是否同样平滑(很可能是普遍现象)。
- ΔΔG 有测量噪声;已用噪声下界(~0.3)对照,cliff 信号(>2–3)远超之,但个别大跳变可能含噪声/不同实验条件。
- 大界面配对做了 CAP=250 采样以控规模;同位点深扫组数有限(378 组、k≥3 仅 133 组)。

## 10. 产物
- 数据:`skempi_per_sample_pred.csv`、`skempi_mutation_cliff_pairs.csv`(同位点对)、`skempi_same_site_groups.csv`、
  `skempi_cliff_pairs_SALI_top2000.csv`(按 SALI 排序)。
- 图:`fig_cliff_vs_noise.png`、`fig_jump_true_vs_pred.png`、`fig_jump_saturation.png`、`fig_absT_by_distance.png`。
- 脚本:`mutation_cliff_analysis.py`、`mutation_cliff_extended.py`。
