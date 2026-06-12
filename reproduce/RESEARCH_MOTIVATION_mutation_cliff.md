# Mutation Cliff:SKEMPIv2 上的数据分析与 ADiT 失效证据(Research Motivation 草稿)

> 一句话:**在 SKEMPIv2 上,"近乎相同的突变 → ΔΔG 巨变"(mutation cliff)真实且普遍存在**(以 cliff 指数
> `SALI = |ΔΔΔG|/d` 度量:**6.3% 的突变对每突变步跳变 >2 kcal/mol**);**而 ADiT 在 cliff 上系统性"抹平"——
> 跳变/陡峭度幅度只预测出约 60%(capture 随 cliff 加剧从 ~1 跌到 0.57),极端 cliff 上甚至预测反向**。
> 聚合指标(overall Pearson 0.66)完全掩盖了这一失效模式。

## 1. 背景与定义
化学信息学里的 **activity cliff**:结构高度相似的分子活性却差异巨大,是 QSAR/打分模型的已知失效点。
迁移到 **蛋白突变效应(ΔΔG)** 任务:

> **Mutation cliff** = 同一复合物界面上、mutant 序列高度相似(尤其同一位点仅替换氨基酸、或仅差一个突变步)
> 但实验 ΔΔG 差异巨大的一对/一组突变。

## 2. 数据与方法
- 数据:SKEMPIv2,ADiT-S 三折交叉验证的**逐样本池化预测**(`reproduce/skempi_per_sample_pred.csv`,6694 样本)。
- **突变距离 d**:两条 mutant 序列**逐点比对、取值不同的位点数**(等价 Hamming 距离;只在两者突变位点并集上比即可,
  其余位点同为 WT)。同位点不同 AA → d=1;不同位点 → d≥2;相同 mutant 已去重故无 d=0。
- **cliff 指数 SALI** = `|ΔΔΔG| / d`(每突变步引起的 ΔΔG 变化,越大越陡)。**这是刻画 cliff 的核心指标**:
  big ΔΔG 若跨很多突变步(大 d)不稀奇,跨一步(小 d)才是"悬崖"。
- **配对集合(§3 SALI、§4 跳变分析共用)**:每个 complex 内 distinct mutant(同突变集合的重复测量先取均值)
  两两任意配对(distinct mutant >250 的大 complex 做 **CAP=250 随机采样**,seed=0),pool 全部 341 个 complex
  共 **270,608 对**。
- **site-matched 噪声基线(§3)**:在同一批 cliff 位点上,用"同位点·同一个 AA 的重复测量"的 ΔΔG 离散度作噪声基线
  (而非全数据任意重复),排除"这些位点本身难测"的替代解释。
- **重复测量如何进训练/测试**:SKEMPI 同一突变的多次测量**各作为独立样本保留**(标签=各自原始实验值),**非取 median**;
  池化口径与 RDE/DiffAffinity/ADiT 一致;实测对 overall 指标无实质影响(去重后 Pearson 0.664 vs 池化 0.662)、且
  split-by-complex 下重复样本恒在同折、无泄漏。
- 脚本:`mutation_cliff_analysis.py`(单点/同位点)、`mutation_cliff_extended.py`(配对/SALI 基础版)、
  `mutation_cliff_viz.py`(§3/§4 的分布图与 ADiT 失效分析,产物在 `mutation_analysis/`)。

## 3. 发现 1:Mutation cliff 真实存在
对全部 270,608 个突变对,从分布形状、尾部、与相似度的关系三个角度看 cliff 指数 **SALI = |ΔΔΔG|/d**。

**Evidence 1 — SALI 的 ECDF:每突变步的大跳变占比可观,且最相似(d=1)处最陡**
<table><tr>
<td><img src="mutation_analysis/ecdf_sali_combined.png" width="420"></td>
<td><img src="mutation_analysis/ecdf_sali_by_d.png" width="420"></td>
</tr></table>

- 全体对:SALI median 0.44,但 **6.3% 的对 SALI>2**(每突变步跳 >2 kcal/mol)、p99 **3.29**、max **12.9**——重尾。
- 分 d:**d=1 的尾部最重**(SALI>2 占 **21.7%**,ECDF 在 x=2 处只到 ~0.78);**单步替换正是最陡的悬崖**。

**Evidence 2 — 排序后 100 等量分位的均值曲线:cliff 是尾部事件**
<table><tr>
<td><img src="mutation_analysis/qmean_sali_combined.png" width="420"></td>
<td><img src="mutation_analysis/qmean_sali_by_d.png" width="420"></td>
</tr></table>

- 把 SALI 从小到大排序、均分 100 桶取均值:前 ~90 个分位平缓(<1.6),**末 ~6 个分位骤升到 3–13**——
  典型重尾,**陡峭 cliff 是少数尾部事件而非整体平移**。
- 分 d:d=1 曲线整体最高(单步最陡),高 d 因除以大 d 而整体下移。

**Evidence 3 — 各 d 的 SALI 点云 + 统计**
<img src="mutation_analysis/pointcloud_sali_by_d.png" width="500">

| d | n | median | 离散系数 CV | p90 | p99 | max | 平滑<0.5 | 悬崖>2 |
|---|---|---|---|---|---|---|---|---|
| 1 | 9,225 | **0.885** | **1.08** | 3.15 | 6.87 | 12.9 | 32.8% | **21.7%** |
| 2 | 147,323 | 0.444 | 1.06 | 1.82 | 3.41 | 5.52 | 53.6% | 8.1% |
| 3 | 36,662 | 0.669 | 0.81 | 1.73 | 2.74 | 4.38 | 40.4% | 5.9% |
| 4 | 22,022 | 0.466 | 0.90 | 1.47 | 2.41 | 3.19 | 52.6% | 3.9% |
| 5 | 14,724 | 0.404 | 0.83 | 1.10 | 1.77 | 2.54 | 59.4% | 0.2% |
| **all** | **270,608** | **0.440** | **1.07** | **1.65** | **3.29** | **12.9** | **54.2%** | **6.3%** |

- **最相似(d=1)处 SALI 最高**:median 0.885、**21.7% 的对每步跳 >2**——相似度越高,单步悬崖越突出。
- **固定 d 内 SALI 高度分散**:各 d 的 CV ≈ 0.8–1.08(>0.8),**相似突变里 SALI 既有近 0、又有 >2 的尾部**,分布不集中。
- **相似度与 SALI 弱(负)相关**:`corr(d, SALI)` Pearson **−0.21** / Spearman **−0.13**——距离越大 SALI 越小(被 d 归一),
  但**单步(d=1)反而最陡**,印证 cliff 集中在"最相似"邻域。

## 4. 发现 2:ADiT 系统性"抹平" cliff(只看 ddG / |ΔΔΔG|)
- **预测幅度压缩**:同位点内真值 ΔΔG range median 0.91,模型预测 range 仅 0.67。
- **跳变低估(同位点突变对,7933 对)**:真值跳变 vs 预测跳变 Pearson 0.57 / Spearman 0.50;cliff 对(|ΔΔΔG真|>2,
  1745 对)模型只预测出 ~0.47 倍幅度,真实大跳变里仅 40% 被判为大跳变(>3 时仅 33%)。
- **同位点排序失效(k≥3,133 组)**:组内 Spearman median 0.50、**mean 仅 0.37、21% 为负**。

### 4.1 把 cliff(高 SALI)与 non-cliff(低 SALI)分开,比模型的 effect 预测
按 cliff 指数 SALI 分桶(0–4 细分 + >4),在每桶比较模型对 mutation effect(ΔΔG 差)的预测。

<img src="mutation_analysis/adit_cliff_failure.png" width="760">

| 真值 SALI 桶(cliff 严重度) | n | 真值 \|ΔΔΔG\| mean | 预测 \|ΔΔΔG\| mean | **capture=预测/真值** |
|---|---|---|---|---|
| 0–0.5 | 146,700 | 0.79 | 1.13 | 1.43* |
| 0.5–1 | 60,651 | 2.43 | 1.72 | 0.71 |
| 1–1.5 | 30,467 | 3.36 | 2.01 | 0.60 |
| 1.5–2 | 15,801 | 4.16 | 2.61 | 0.63 |
| 2–3 | 12,920 | 5.27 | 3.04 | 0.58 |
| 3–4 | 2,982 | 6.44 | 3.79 | 0.59 |
| >4 | 1,087 | 7.12 | 4.38 | 0.61 |

**左图(幅度 flatten)**:cliff 越严重,真值 effect 一路爬升(0.8→7.1),但**预测 effect 被压在低位**(1.1→4.4)——
两条柱的差距越拉越大。**非 cliff(SALI<2)capture = 0.86,cliff(SALI≥2)只有 0.58**:模型只复现出 cliff 区约六成的
effect 幅度,把悬崖"抹平"。(*0–0.5 桶 capture=1.43>1 是模型对近 0 effect 的"默认高估",非有效信号。)

**右图(cliff 识别失败)**:在真值确为陡峭 cliff 的对里,模型把它也预测成 cliff 的比例(recall)极低——
**SALI≥2 仅 28.8%、SALI≥3 仅 29.6%**(即**漏掉约 70% 的真实 cliff**);真值 cliff(SALI>2)的预测 SALI 中位被压到
~1.36(真值 2.49 的一半)。**模型基本不会主动"喊出"一个 cliff。**

> 说明:若改用 signed 跳变的 **Pearson/Spearman**,会**随 cliff 反而上升**(0.37→0.83)——这是动态范围/SNR 的内禀
> 效应(大 cliff 跳变绝对值大、易判方向),只能说明"方向/谁大谁小"被保住,**不能**用来衡量 cliff 上的预测质量。
> cliff 失效体现在**幅度被压缩 + cliff 识别失败**,而非方向。

### 4.2 全配对的真值/预测跳变表 + 极端失配 case
- **完整表**(270,608 对):`mutation_analysis/sali_pairs_table.csv`,列 = `PDB, site(差异位点), d, ddG_A, ddG_B,
  true_jump, pred_jump, diff_percent`,其中 **`diff_percent = (pred_jump − true_jump) / true_jump`**(预测跳变相对真值
  跳变多/少多少);已按 **`PDB → d↑ → |diff_percent|↓`** 排序。
- **Top 极端失配 case(真值跳变 >4 kcal/mol,模型严重失配,按 diff_percent 最负):**

| PDB | 差异位点 | d | ddG_A | ddG_B | 真值跳变 | 预测跳变 | diff_percent |
|---|---|---|---|---|---|---|---|
| 1PPF_E_I | I18,I32 | 2 | 6.64 | 2.52 | +4.11 | **−7.46** | −282% |
| 1R0R_E_I | I10,I13,I16,I18,I27 | 5 | 5.90 | 1.89 | +4.01 | **−6.04** | −250% |
| 1PPF_E_I | I10,I17,I18,I19,I21,… | 9 | 7.54 | 3.28 | +4.26 | **−6.40** | −250% |
| 1PPF_E_I | I18 | 1 | 3.33 | 7.54 | −4.21 | **+5.81** | −238% |
| 1R0R_E_I | I10,I12,…,I43,I5 | 10 | 1.78 | 5.90 | −4.13 | **+5.48** | −233% |
| 1CHO_EFG_I | I12,I14,…,I46,I7 | 11 | −0.03 | 4.23 | −4.26 | **+5.53** | −230% |

- **现象**:这些对的真值跳变 4–5 kcal/mol,**ADiT 不仅低估,反而预测出方向相反、幅度相当的跳变**(diff_percent ≈ −230~−282%)。
- **集中在蛋白酶-抑制剂界面**(1PPF / 1R0R / 1CHO 的 P1 附近热点位点)。
- **结论**:与 §4.1 互补——平均看模型"方向多半对、幅度压缩",但在**最陡的极端 cliff 上会彻底失配(预测反向)**;
  cliff 失效**横跨多界面、多种 d,是系统现象**。

## 5. 产物
- 数据:`skempi_per_sample_pred.csv`、`skempi_mutation_cliff_pairs.csv`、`skempi_same_site_groups.csv`、
  `skempi_cliff_pairs_SALI_top2000.csv`;**全配对表 `mutation_analysis/sali_pairs_table.csv`(270,608 行)**。
- 图(`mutation_analysis/`):`ecdf_sali_combined.png`/`ecdf_sali_by_d.png`(Evidence 1)、`qmean_sali_combined.png`/`qmean_sali_by_d.png`(Evidence 2)、`pointcloud_sali_by_d.png`(Evidence 3)、`adit_cliff_failure.png`(§4.1)。
- 图(根目录,早期版本/旁证):`fig_cliff_vs_noise.png`、`fig_jump_true_vs_pred.png`、`fig_jump_saturation.png`、`fig_absT_by_distance.png`。
- 脚本:`mutation_cliff_analysis.py`、`mutation_cliff_extended.py`、`mutation_cliff_viz.py`。
