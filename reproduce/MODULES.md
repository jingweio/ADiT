# ADiT 复现 — 实验相关核心模块说明

> 论文:*Towards All-Atom Foundation Models for Biomolecular Binding Affinity Prediction* (ICLR 2026)
> 本文档面向**复现 protein-ligand (LBA) 与 protein-protein (SKEMPIv2) 两类 binding affinity 预测**,
> 只覆盖与这两个实验直接相关的模块。代码基于 **PyTorch-Lightning + Hydra** 模板。

---

## 0. 总体数据流(一图看懂)

```
配置 (Hydra: configs/) ──┐
                         ├─► adit/train.py  (finetune 入口, 读 configs/train.yaml)
                         └─► adit/test.py   (测试入口,   读 configs/test_ec.yaml)
                                   │
   ┌───────────────────────────────┼─────────────────────────────────┐
   ▼                                ▼                                 ▼
DataModule                        LightningModule                  Trainer
ProteinDataModule          LBAModule / SkempiModule              (Lightning)
   │  (Dataset + collate)         │  (net + mlp + loss + metrics)
   ▼                              ▼
LBADataset / SkempiDataset   net = ADiT  ──►  complex_feat ──► MLP4downstream ──► 预测值
   │  + FeatureTransform         │
   ▼                             └─ SeqEmbedder(+ESM-2-650M) → Pairformer → Diffusion Transformer
特征字典(aatype/坐标/mask/esm_repr…)
```

- **预测目标**:LBA → `neglog_aff`(回归亲和力);SKEMPI → `ddG`(突变引起的结合自由能变化 ΔΔG)。
- **复现 = 先 finetune 预训练 backbone(`ckpts/adit_S.ckpt`)→ 再用 finetune 后的 ckpt 测试**。

---

## 1. 入口脚本

| 文件 | 作用 | 关键点 |
|---|---|---|
| `train.sh` / `test.sh` | 薄包装,实际执行 `python adit/train.py $@` / `adit/test.py $@` | 设 `NUMEXPR_MAX_THREADS=8` |
| `adit/train.py` | **finetune 入口**,默认配置 `configs/train.yaml` | 见下方加载逻辑 |
| `adit/test.py`  | **测试入口**,默认配置 `configs/test_ec.yaml`(`test: true`) | 用 `trainer.test(ckpt_path=...)` 加载 finetune 后的完整 Lightning ckpt |

### finetune 时如何加载预训练权重(`adit/train.py:89` → `adit/utils/checkpoint_utils.py`)
```python
model, ckpt_path = checkpoint_utils.load_model_checkpoint(
    model, cfg.get("ckpt_path"), load_weight_only=True)   # 默认 True
```
- 传入 `ckpt_path=ckpts/adit_S.ckpt`(`.ckpt`)时:取出 `state_dict`,**去掉 `net.` 前缀**,以
  **`strict=False`** 加载进 `model.net`(即 ADiT backbone)。下游的 `mlp` 头是**随机初始化**的。
- 函数返回的 `ckpt_path` 被置为 **`None`**,所以 `trainer.fit` **不会**做 Lightning 全量 resume
  ——这正是"加载预训练 backbone 后从头 finetune"的语义,而非续训。
- `rootutils.setup_root(..., indicator=".project-root")` 在 import 时自动设置 `PROJECT_ROOT`
  环境变量,`configs/paths/*` 里所有相对路径都以仓库根为基准,**无需手动 export**。

---

## 2. 配置体系(Hydra)

组合方式:`configs/train.yaml` 的 `defaults` 把 data / model / trainer / callbacks 等拼起来,
`experiment=<name>` 再覆盖特定超参。命令行用 `++a.b=c` 覆盖、`+a.b=c` 新增键。

| 实验 config | data | model | trainer | 关键超参 |
|---|---|---|---|---|
| `configs/experiment/lba_S.yaml`    | `lba`    | `lba`    | `ddp` | `mlp.input_dim=128`,`max_epochs=4`,监控 `val/mse_loss` |
| `configs/experiment/skempi_S.yaml` | `skempi` | `skempi` | `ddp` | `mlp.input_dim=256`,`min/max_epochs=60`,`lr=1e-4` |

- `configs/paths/env.yaml`:数据/权重路径锚点 —— `lba_path=dataset/LBA`、`skempi_path=dataset/skempi`、
  `skempi_esm_repr_path=dataset/skempi_esm_repr`。
- `configs/trainer/ddp.yaml`:默认 `accelerator=gpu, devices=4`。**本机只有 2 卡**,复现时用
  `++trainer.devices=1` 覆盖(见 `run_pl_lba.sh` / `run_ppi_skempi.sh`)。
- checkpoint 落盘:`outputs/<日期>/<task_name>_<时间>/checkpoints/`(`configs/hydra/default.yaml`
  的 `run.dir` + Lightning `ModelCheckpoint`)。

---

## 3. 数据层

### 3.1 `adit/data/protein_datamodule.py` — `ProteinDataModule`
- `setup('fit')`:对有 `set_training_mode` 的数据集调用之,再按 `train_val_split=[0.95,0.05]` 切 train/val。
- `setup('test')`:调用 `set_testing_mode`,整个数据集作为 test set。
- `BatchTensorConverter`:把 "list of dict" 整理成 "dict of tensor",并对变长张量**右侧 padding 对齐**。
- DDP 下自动把 `batch_size` 除以 `world_size`(要求可整除)。

### 3.2 `adit/data/components/dataset.py`
| 类 | 任务 | split 机制 |
|---|---|---|
| `LBADataset` | 蛋白-配体亲和力 | 读 `LBA/lba_{identity_30,identity_60}_indices/{train,val,test}_indices.txt` 指定的索引 |
| `SkempiDataset` | 蛋白-蛋白 ΔΔG | 目录 `skempi/{split_0,split_1,split_2}` 为**三折交叉验证**的三折 |

**SKEMPI 三折机制(关键,容易误解)**:
- `set_training_mode()`:`test_split` 之外的两折 → 训练;`test_split` 那一折 → 验证。
- `set_testing_mode()`:`test_split` 那一折 → 测试。
- ⚠️ **验证集 == 测试折**(代码里 val 与 test 是同一折,没有独立 val 集)。
- 论文 Table 2 的 PPI 数字是**三折分别留出当 test、各得一个 `result_split_i.pkl`,再拼接**算 overall
  指标(`scripts/skempi_metric.py`)。**本次复现只跑 `split_0` 一折**,得到的是单折指标,不等于论文 overall 值。

### 3.3 `adit/data/components/feature_transform.py`
- `LBAFeatureTransform` / `FeatureTransform`:把原始结构转成模型输入(单位 `nm`、截断长度
  LBA=500 / skempi=1000、`strip_missing_residues`、`recenter_and_scale` 等),产出 `aatype`、
  `atom_positions`、各种 `mask`、`token_idx`、`chain_index` 等字段。

---

## 4. 模型层

### 4.1 任务模块 `adit/models/downstream_module.py`
继承链:`BaseLitModule` → `BaseDownstreamModule` → `LBAModule` / `SkempiModule`。

- `BaseDownstreamModule.forward`:`net(batch)` 得到 `complex_feat`,再过 `mlp` 得到标量预测。
- **`LBAModule`**:`get_target = batch["neglog_aff"]`,直接回归。`mlp.input_dim=128`。
- **`SkempiModule`**:预测 ΔΔG 用**反对称差分**(抵消系统偏置),`mlp.input_dim=256`:
  ```python
  pred = mlp([mt_feat, wt_feat]) - mlp([wt_feat, mt_feat])   # 对 wt/mt 各跑一遍 net
  ```
  `data_object_split` 把 batch 里 `wt_`/`mt_` 前缀字段拆成两套输入。
- `on_test_epoch_end`:把 `(accession_code, all_preds, all_targets)` pickle 到 `model.save_file`
  —— 这就是 `result_split_0.pkl` 的来源。
- `Metrics`:训练/验证用 `metrics`,测试用 `metrics_test`。LBA 测试报 `mse_loss/spearmanr/pearsonr`;
  SKEMPI 测试报 `mse_loss/mae_loss/spearmanr/pearsonr`。

### 4.2 主干网络 `adit/models/net/adit/model.py` — `ADiT`
AlphaFold3 改造而来的**全原子表征编码器**,4 个子模块:

| 子模块 | 文件 | 作用 |
|---|---|---|
| `SeqEmbedder` | `seq_embedder.py` | token 初始特征;内部用 **ESM-2-650M** 编码序列(`batch["esm_repr"]`) |
| `RelativePositionEncoding` | `relative_position_encoder.py` | token 对的相对位置编码(含跨链信息) |
| `SimplePairFormer` | `pairformer.py` | 轻量化 Pairformer(去掉 AF3 中昂贵部分),更新 token / token-pair 表征 |
| `DiffusionModule` | `diffusion_module.py` | 原子级 + token 级 **Diffusion Transformer**;`N_block_atom/token` 控制深度 |

- 注意:此处 diffusion transformer 被用作**几何编码器**(把观测到的结构编码成表征),
  而非生成式去噪采样 —— 这是论文"从生成式结构预测 → 表征学习"改造的核心。
- 输出 `batch['complex_feat']` 作为复合物的整体表征。
- `ADiT-S` 配置:`token_dim=384, atom_dim=128, N_block_atom=3, N_block_token=5`(见 `configs/model/lba.yaml`;
  skempi 在 `experiment` 里另调 `N_block_*` 等)。

### 4.3 预测头 `adit/models/net/mlp.py` — `MLP4downstream`
把 `complex_feat`(LBA:128 维;SKEMPI:拼接 wt+mt = 256 维)映射到标量。

### 4.4 ESM 封装 `adit/models/net/esm2/esm.py` — `ESM`
- 从 `ckpts/esm2_t33_650M_UR50D.pt` 加载 **ESM-2-650M**(`output_dim=1280`,33 层)。
- ⚠️ 顶部 `import esm` 用的是 **`fair-esm`** 包(`esm.pretrained.load_model_and_alphabet_core`)——
  README 的依赖清单**漏列**了它,复现环境已补装(见 `setup_env.sh`)。
- SKEMPI 走 `scripts/dump_esm_repr.py` **离线预计算** ESM 表征存到 `dataset/skempi_esm_repr/`,
  训练时直接读取,避免每个 step 重复跑 650M 模型。

---

## 5. 复现相关脚本

| 文件 | 作用 |
|---|---|
| `scripts/dump_esm_repr.py` | 为 SKEMPI(及 HER2)离线计算 ESM 表征。本次只用到 `main_skempi()`;`main_HER2()` 因未下载 HER2 数据而是空操作 |
| `scripts/skempi_metric.py` | 把 `result_split_0/1/2.pkl` **拼接**后算 overall Pearson/Spearman/RMSE/MAE(需三折齐全) |

---

## 6. 一句话复现路径

```
setup_env.sh ──► download_assets.sh ──┐
                                       ├─► run_pl_lba.sh      (PL / 蛋白-配体)
                                       │     ├─ smoke_lba
                                       │     └─ full_lba_train → full_lba_test
                                       └─► run_ppi_skempi.sh  (PPI / 蛋白-蛋白)
                                             ├─ run_step_A_esm_repr   (先跑: dump_esm_repr.py)
                                             ├─ smoke_skempi
                                             └─ full_skempi_train → full_skempi_test  (split_0 一折)
```

> 复现命令按实验拆成两个文件:**`run_pl_lba.sh`**(蛋白-配体)和 **`run_ppi_skempi.sh`**(蛋白-蛋白)。

---

## 7. 运行产物 / 日志 / 复现(Hydra + Lightning)

本仓库用 **Hydra 管配置 + PyTorch Lightning 跑训练**。每次 `train.py`/`test.py` 运行都会在
`outputs/<日期>/<...>/` 下**自动**生成一整套执行记录(与你是否做 `> log` 重定向无关)。

### 7.1 框架分工:Hydra vs Lightning
- **Hydra**:配置管理。把 `train.yaml 默认 → experiment=... → 命令行 ++/+/=` 合成成一棵 `cfg`,
  并自动把配置快照写到 `.hydra/`。
- **PyTorch Lightning**(简称 Lightning):训练执行框架。封装训练/验证/测试循环、DDP、混合精度、
  checkpoint、日志、回调(EarlyStopping / ModelCheckpoint)。本 repo 的 `SkempiModule`/`LBAModule`
  是 `LightningModule`,`ProteinDataModule` 是 `LightningDataModule`,`Trainer` 跑全程。
  你看到的 `metrics.csv`、`hparams.yaml`、`last.ckpt`、`val/test` 指标、DDP 策略 **都是 Lightning 的产物/约定**。
- 配合关系:**Hydra 组装 `cfg` → 实例化成 Lightning 的 Module/Trainer → Lightning 训练并产出 ckpt/日志**。

### 7.2 输出目录命名
- `train.py`(用 `train.yaml`,含 `hydra: default`)→ 目录名带前缀:`outputs/<日期>/<task_name>_<时-分-秒>/`
  (例:`skempi_S_split_0_15-57-12`)。
- `test.py`(用 `test_ec.yaml`,**注释掉了 `hydra: default`**)→ 退回 Hydra 内置默认:`outputs/<日期>/<时-分-秒>/`
  (只有时间、无 task 前缀,例:`20-38-13`)。

### 7.3 各文件含义
| 文件 | 谁写的 | 含义 |
|---|---|---|
| `.hydra/overrides.yaml` | Hydra | 本次**命令行覆盖项**(≈ 接在 `python adit/train.py` 后面那串参数;不含环境变量) |
| `.hydra/config.yaml` | Hydra | 本次**完整合成后的 app 配置**(data+model+trainer+callbacks+paths+logger…)。复现主要看它 |
| `.hydra/hydra.yaml` | Hydra | **Hydra 框架自身**的配置(输出目录格式、job 名、日志、sweeper/launcher)。复现科学结果时基本可忽略 |
| `config_tree.log` | 项目 `rich_utils`(`extras.print_config`) | 与 config.yaml **同内容**的**树状美化打印**,给人肉眼核对 |
| `tags.log` | 项目 | 本次的 tags |
| `train.log` / `test.log` | 项目 RankedLogger / Hydra job logging | 运行日志 |
| `csv/version_0/metrics.csv` | **Lightning CSVLogger** | **逐 epoch/step 指标**(权威训练曲线) |
| `csv/version_0/hparams.yaml` | **Lightning** `save_hyperparameters()` | 模型超参,**会被一并写进 ckpt** 用于加载时重建模型 |
| `checkpoints/last.ckpt` | **Lightning** ModelCheckpoint 回调 | 模型权重(见 7.5) |

### 7.4 config.yaml vs hparams.yaml(命令配置 vs 记录配置)
- **`config.yaml` = 命令配置 / 全量**:Hydra 由你的命令合成的**整套实验配置**。跟"这次运行"绑。
- **`hparams.yaml` = 记录配置 / 模型专属**:Lightning 记下的**模型超参**(net 维度、optimizer、loss…),
  **随 `last.ckpt` 一起保存**,加载 ckpt 时用它重建模型结构。
- 关系:`hparams.yaml` ≈ `config.yaml` 的 `model` 部分的副本,但用途是"随模型走、可重建",不只是查看。

### 7.5 metrics.csv 解读 + last.ckpt 语义(重要)
- `train/loss` **就是 MSE**(skempi/LBA 的 `loss=MseLoss`),故 `train/loss` 与 `train/mse loss` 数值相同。
- `val/mse_loss` **是在 test 折上算的**:skempi `set_training_mode` 把 `test_split`(如 split_0)那一折
  **同时当 validation**(无独立 val 集)→ 属于**软泄漏**,LR 调度也监控它。
- **`last.ckpt` 不是"验证最优模型"**:skempi `save_top_k=0`(不存 best)+ `save_last=true` →
  只存**最后一轮(epoch 60)的权重**。**固定训 60 epoch、用最终轮权重做推理/报告**,不按 val 挑 epoch。

### 7.6 如何复现某次运行
- **办法 A(推荐,最稳)**:把该次 `.hydra/overrides.yaml` 里的项原样再传一遍(代码+configs 受版本控制,
  同样 overrides → 同样 cfg):
  ```bash
  bash train.sh experiment=skempi_S ++trainer.devices=1 ++trainer.strategy=auto \
    ++data.batch_size=8 ++data.dataset.test_split=split_0 \
    ckpt_path=ckpts/adit_S.ckpt task_name=skempi_S_split_0
  ```
- **办法 B(用存档配置)**:`python adit/train.py --config-path <那次>/.hydra --config-name config`
  (能跑,但存档是"已合成"配置,偶有细微差异,日常更推荐 A)。
- ⚠️ **复现精度**:`seed=42` 但 `trainer.deterministic=false` + 数据 shuffle/dropout/GPU 非确定性 →
  结果**接近但非逐位一致**;要更可复现可加 `++trainer.deterministic=true`(更慢)。

---

## 8. 复现结果(ADiT-S)

环境:RTX A4500(20GB),每折 60 epoch finetune 约 4.5 小时;全程通过 `run_ppi_skempi.sh` /
`run_pl_lba.sh` 入口启动。

### 8.1 蛋白-蛋白 ΔΔG(SKEMPIv2,完整三折交叉验证)

**三折单折指标**(每折留出当 test):

| Fold | 测试样本 | Pearson↑ | Spearman↑ | RMSE↓ | MAE↓ |
|---|---|---|---|---|---|
| split_0 | 2531 | 0.625 | 0.506 | 1.605 | 1.139 |
| split_1 | 2413 | 0.708 | 0.512 | 1.378 | 0.973 |
| split_2 | 1750 | 0.690 | 0.539 | 1.891 | 1.355 |

**三折汇总**(三折预测拼接后算整体 = 论文 Table 2 协议,`scripts/skempi_metric.py`):

| 指标 | 本次复现 | 论文 ADiT-S | 差异 |
|---|---|---|---|
| Pearson↑ | **0.6625** | 0.660 | +0.0025(≈论文"≤0.002可忽略"水平) |
| Spearman↑ | **0.5136** | 0.524 | −0.010 |
| RMSE↓ | **1.6101** | 1.597 | +0.013 |
| MAE↓ | **1.1355** | 1.132 | +0.004 |

→ **复现成功**:Pearson 实质匹配,RMSE/MAE 几乎一致,Spearman 略低(单次训练+非确定性的正常波动)。

### 8.2 蛋白-配体亲和力(LBA,identity_30 + identity_60,完整复现)

均用**验证最优 ckpt**(LBA 有真正的 val 集、`save_top_k=1` 按 `val/mse_loss` 选),全程经 `run_pl_lba.sh` 入口:

| 基准 | 指标 | 本次复现 | 论文 ADiT-S | 差异 |
|---|---|---|---|---|
| **LBA-30**(identity_30) | Pearson↑ | 0.604 | 0.626 | −0.022 |
| | Spearman↑ | 0.600 | 0.618 | −0.018 |
| | RMSE↓ | 1.404 | 1.337 | +0.067 |
| **LBA-60**(identity_60) | Pearson↑ | **0.734** | 0.740 | −0.006 |
| | Spearman↑ | **0.726** | 0.740 | −0.014 |
| | RMSE↓ | **1.416** | 1.413 | +0.003 |

→ **LBA-60 几乎完美吻合**;**LBA-30 接近但偏低**(见下方已知问题)。

> ⚠️ **已知问题(LBA-30,待修)**:`configs/experiment/lba_S.yaml` 自带 `max_epochs=4`,而
> identity_30 的复现命令未覆盖它(README 也只对 identity_60 加 `++trainer.max_epochs=100`)。
> 因此 **LBA-30 实际只训了 4 个 epoch,明显训练不足**,导致 Pearson/Spearman 略低于论文、RMSE 偏高。
> 对比:LBA-60 训了 100 epoch → 高度吻合。**回头修复方案**:给 LBA-30 也加 `++trainer.max_epochs=100`
> (或一个合适的更大值)重训重测,预计指标能逼近论文 0.626/0.618/1.337。本次先按 repo 默认值如实记录。

### 8.3 复现产物位置
- 预测(PPI):`reproduce/result_split_0|1|2.pkl`
- 预测(LBA):`reproduce/result_lba_identity_30.pkl`、`reproduce/result_lba_identity_60.pkl`
- checkpoint:`outputs/<日期>/skempi_S_split_<n>_<时间>/checkpoints/last.ckpt`(PPI,用 last);
  `outputs/<日期>/lba_S_identity_<30|60>_<时间>/checkpoints/epoch_*.ckpt`(LBA,用验证最优)
- 日志:`reproduce/full_{skempi,lba}_{train,test}_*.log`、各运行的 `outputs/.../csv/version_0/metrics.csv`

---

## 9. SKEMPI 逐样本分析:按突变位点数(#mutation-site)分层

把三折池化预测拆到每个样本(`reproduce/skempi_per_sample_pred.csv`,6694 条,格式参考 `demo_only.csv`),
按"突变位点数"分层分析。脚本:`reproduce/skempi_per_sample_analysis.py`(分布+全局指标)、
`reproduce/analyze_ranking_hypothesis.py`(range 混杂检验)、`reproduce/skempi_per_interface_by_nsites.py`(per-interface 指标)。
> 数据源是三折 CV 池化预测(每样本由"训练时没见过它"的那折模型预测);num_mutation_sites 由 accession_code 解析(突变段按逗号计数)。

### 9.1 数据分布
single(n=1)= **4935(73.7%)**;multi(n>1)= **1759(26.3%)**;最多 27 个位点(长尾极少)。

### 9.2 指标随口径变化(关键)—— 全局池化 vs per-interface
| | 全局池化 Spearman | **per-interface Spearman(≥10/界面)** |
|---|---|---|
| single(n=1) | 0.484 | **0.336** |
| multi(n>1) | 0.546 | **0.362** |
| overall | 0.514 | **0.341** |

- **全局池化会高估排序力**:overall 0.514 → per-interface 0.341,因为池化混入了"不同界面 ddG 基线不同"的跨界面方差。**within-interface(per-structure)才是生物上有意义的排序指标**。
- per-interface 完整表(K=10):n=1 → P=0.381/S=0.336/RMSE=1.192/MAE=0.927;multi → P=0.360/S=0.362/RMSE=1.971/MAE=1.656;overall → P=0.376/S=0.341/RMSE=1.393/MAE=1.092。详见 `reproduce/skempi_per_interface_by_nsites_K10.csv`。

### 9.3 "multi 排序更好"的辨析(结论:是统计假象)
- 现象:**全局** Spearman multi(0.546)> single(0.484)且显著(bootstrap CI 几乎不重叠);但 multi 的 **Pearson 反而更低**。
- 混杂:multi 的真实 ddG 动态范围大得多(std 2.70 vs 1.74、|ddG|均值 2.42 vs 1.32),**range 越宽 rank 相关越虚高**。把两组裁到相同 ddG 窗后,multi Spearman 0.546→0.439、single 几乎不变 → 优势消失甚至反转。
- 换 **per-interface** 口径后:single vs multi 的 Spearman 贴平(0.336 vs 0.362),Pearson single 反而更高(0.381 vs 0.360),K=5 时 single 两项都更高。
- **结论**:multi 先前的"排序优势"是 **全局池化 + 宽 ddG range** 造成的假象,**并非真实的 within-interface 排序能力**。

### 9.4 真实且稳健的差距
**multi 的绝对误差确实更大**(per-interface RMSE 1.97 vs 1.19、MAE 1.66 vs 0.93),不随口径变化——这是真实差距(多位点突变涉及 epistasis,更难精确预测)。
→ 正确结论:**multi 在绝对精度上更难;在 within-interface 排序上并不比 single 强。**

### 9.5 口径局限(必看)
- 按 exact n 分层后覆盖很稀疏:per-interface 下只有 **n=1、n=2、single、multi、overall** 样本量够;**exact n≥3 不可信**(如 n=5 仅 1 个界面、n≥6 多为 0)。
- 领域标准指标:**per-structure Pearson/Spearman**(每界面 ≥10 突变、跨界面取均值,头条)+ **overall** Pearson/Spearman/RMSE/MAE + ddG 符号的 **AUROC**;参考 RDE-Network / DiffAffinity。
  ⚠️ **ADiT 仓库 `skempi_metric.py` 报的是 overall 池化**,与某些前作强调的 per-structure 不是同一口径,横向对比需统一。

---

## 10. HER2(抗体-抗原)任务协议说明(本次未复现,仅厘清)

结论基于 README、配置、以及论文正文 4.5 节 + Appendix D.5 + Table 6。

- **训练/测试关系**:**在 SKEMPIv2 上训练 → 在 HER2 binders 上零样本测试(跨任务迁移)**。论文 4.5:
  "With the models trained on SKEMPIv2, we test their performance on the HER2 binders test set."
- **HER2 是纯测试集**:Table 6 显示 HER2 的 Train/Validation = —,**Test = 419**(抗体-抗原亲和力);
  README 抗体-抗原部分**只有 `test.sh`、无 `train.sh`**,`ckpt_path="PATH_TO_FINETUNED_CKPT"`。
- **用的模型**:SKEMPIv2-ΔΔG-finetuned 的 ADiT(按 S/M/L 尺寸,见 Table 3);`experiment=her2_L` 用
  `data=her2`(HER2Dataset)+ `model=skempi` 架构。HER2 需先用 `scripts/dump_esm_repr.py` 的
  `main_HER2()` 生成 `dataset/HER2_esm_repr/`。
- ⚠️ **未公开的细节:具体用了 SKEMPI 的哪一折 checkpoint(还是三折集成)——论文(含 Appendix D.5)
  没有写明**。代码层面每个 SKEMPI ckpt 都是"训 2 折、留 1 折"(无全量训练模式),所以 HER2 用的必然是
  某个 2-折训练的 SKEMPI-finetuned ckpt,但"哪一折/是否集成"无法从 repo 或论文确定;HF 上也只发布了
  预训练 ckpt、没有 SKEMPI-finetuned ckpt,无法反推。
- **影响**:HER2 是外部数据,训练只用 SKEMPIv2,用哪折都不构成对 HER2 的泄漏;此处仅为协议记录,
  本次复现未涉及 HER2(范围限定 PL + PPI)。
