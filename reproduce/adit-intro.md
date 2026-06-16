# ADiT 在 SKEMPIv2 上的 mutation-effect prediction：模型与超参一览

本篇回答两个问题，依据 **论文 `Sources/ADiT.pdf`（ICLR 2026, Appendix D.4 + Table 5/6）** 与
**本仓库 `ADiT/configs/`** 的实际复现配置，并标出二者差异：

1. SKEMPIv2 ΔΔG（蛋白-蛋白突变效应）这个 downstream 任务涉及哪些模型/训练超参？
2. 作者对 ADiT-S/M/L 三个规模是否用了**相同**超参？

> 任务定义：预测突变引起的结合自由能变化 ΔΔG。SkempiModule 用**反对称差**预测：
> `pred = MLP([mt,wt]) − MLP([wt,mt])`（对 wt / mt 各跑一遍冻结-ESM + ADiT 主干），loss = MSE。
> 评测：split-by-complex 三折交叉验证（follow DiffAffinity / Liu et al. 2024），指标 Pearson / Spearman / RMSE / MAE。

---

## 1. 一句话结论

- **训练超参（learning rate、batch、dropout、epochs、截断长度、optimizer、loss、split）：S/M/L 三个规模完全相同。**
  论文 D.4 只给出**单一一组**训练设置，未按模型规模区分（对照 D.3 Davis 任务才出现"lr 按规模不同"）。
- **网络架构：S/M/L 不同——这正是 scaling study 本身。** 差异**只在 token 流**（宽度/深度/头数随规模放大）；
  **atom 流三者完全一致**。
- **一处复现偏差需注意**：论文 D.4 写 **lr = 3e-5**，而本仓库 `configs/experiment/skempi_{S,M,L}.yaml`
  三者都用 **lr = 1e-4**（详见 §7）。
- **超参不是逐规模搜出来的，而是 once-for-all**：以 ADiT-M 为锚、继承前人协议固定一套配方，S/L 只缩放架构、原样套用（§4）。
- **"50 残基截断"不是序列上 ±50，而是突变中心的 3D 空间 patch**：取离突变位点最近的 50 个残基（跨两条链、含界面邻域）（§5）。
- **预训练 crop（≤250）并不锁死下游上下文长度**：相对位置编码 length-agnostic + 下游头可重训，能否吃 full-complex 本质是显存工程问题（§6）。

---

## 2. 模型架构超参（Table 5）——S/M/L **不同**

| 配置项（论文 Table 5 / 仓库 net 字段） | ADiT-S | ADiT-M | ADiT-L | 随规模变化? |
|---|---|---|---|---|
| Atom Repr Dim (`atom_dim`) | 128 | 128 | 128 | 否 |
| Atom Pair Dim (`atom_pair_dim`) | 16 | 16 | 16 | 否 |
| Atom Enc/Dec Layers (`N_block_atom`) | 3 | 3 | 3 | 否 |
| Atom Enc/Dec Heads (`N_head_atom`) | 4 | 4 | 4 | 否 |
| **Token Repr Dim (`token_dim`)** | **384** | **384** | **768** | **是** |
| **Token Pair Dim (`token_pair_dim`)** | **32** | **64** | **128** | **是** |
| **Token Attn Layers (`N_block_token`)** | **3** | **12** | **24** | **是** |
| **Token Attn Heads (`N_head_token`)** | **8** | **8** | **16** | **是** |
| Cross-attn `N_query` / `N_key` | 32 / 128 | 32 / 128 | 32 / 128 | 否 |
| dropout | 0.0 | 0.0 | 0.0 | 否 |
| MLP head `input_dim` | 256 | 256 | 256 | 否 |
| ESM backbone | ESM-2-650M（冻结） | 同 | 同 | 否 |
| **可训练参数量** | **~12M** | **~35M** | **~253M** | **是** |

要点：
- **唯一变化的就是 token-stream 的容量**（`token_dim`、`token_pair_dim`、`N_block_token`、`N_head_token`）；
  atom-stream（dim/pair/layers/heads）和 cross-attention、dropout、MLP head 在三个规模间**逐字相同**。
- ADiT-L 的 atom/token transformer 层数对齐 AlphaFold 3。
- 冻结的 ESM-2-650M（~651M 参数）不计入"可训练参数"；ADiT-S 实测 trainable = 12,363,697（≈12M），
  与论文"12M"一致（来源：`outputs/.../skempi_S_split_0_*/csv/version_0/hparams.yaml`）。
- 仓库映射：`configs/experiment/skempi_{S,M,L}.yaml` 的 `model.net.*` 字段 = Table 5；其 base 默认在
  `configs/model/skempi.yaml`（base 里 `N_block_atom: 8`、`mlp.input_dim: 128`，**被 experiment 覆盖**为 3 / 256，
  以 experiment 配置为准）。

---

## 3. 训练 / 数据超参（Appendix D.4）——S/M/L **相同**

| 超参 | 值（S = M = L） | 论文出处 | 仓库出处 |
|---|---|---|---|
| Optimizer | AdamW，weight_decay = 0 | D.4（论文写 "Adam"） | `model/skempi.yaml` |
| Learning rate | **3e-5**（论文）/ **1e-4**（仓库，见 §4） | D.4 | experiment `*.yaml` |
| LR scheduler | ReduceLROnPlateau(factor 0.1, patience 10), monitor `val/mse_loss` | —（论文未提） | `model/skempi.yaml` |
| Effective batch size | 8（`batch_size 8` × `devices 1`） | D.4 | experiment + data |
| Dropout | 0.0 | D.4 | net 配置 |
| Epochs | 60（min=max=60） | D.4 | experiment `*.yaml` |
| Loss | MSE（`MseLoss`） | —（ΔΔG 回归） | `model/skempi.yaml` |
| Seed | 42 | — | experiment `*.yaml` |
| Train/Val split | 0.95 / 0.05 | — | `data/skempi.yaml` |
| 输入截断 | 突变中心 **3D 最近 50 残基**（spatial patch，跨两链；`skempi_truncate=true`, `truncation_size=50`, `consecutive=false`）；机制详见 §5 | D.4 | `data/skempi.yaml` |
| 其它预处理 | unit=nm, `truncate_length=1000`, strip missing residues, recenter & scale | — | `data/skempi.yaml` |
| 数据增强 | 无 rotation augmentation（与 LBA/Davis 不同） | D.4 | — |
| Split | split-by-complex 三折 CV（fold0/1/2） | D.4 / 4.4 | `test_split=split_{0,1,2}` |
| 硬件 | 单张 A100-40G | D.4 | — |

**三折规模（论文 Table 6）：** fold0 train 4765 / test 1929；fold1 4282 / 2412；fold2 4341 / 2353。
（注：论文正文称 SKEMPIv2 覆盖 348 complexes / 7085 muts；ADiT 实际建模数据 DS-ADIT 为 341 complexes / 6694 records。）

> 关键：以上**整张表 S/M/L 取值完全一致**。论文 D.4 通篇只给一组训练设置、没有任何"按规模区分"的措辞，
> 这与 D.3（Davis）形成对照——D.3 明确写 "lr = 1e-4 for ADiT-S and 3e-5 for ADiT-M and ADiT-L"，
> 而 D.4（SKEMPI）只写单一一个 lr，**说明 SKEMPI 任务上作者对 S/M/L 用了相同训练超参**。

---

## 4. 超参是怎么选出来的？——once-for-all，不是逐规模择优

**作者没有对 S/M/L 分别做超参寻优**，而是固定一套训练配方、三个规模通用；变的只有架构。论文里**没有任何 ADiT downstream 的超参搜索描述**（"tune" 字样只出现在 pretraining 噪声尺度 σ 的 preliminary experiments，及为 **Protenix baseline** 调参）。证据：

1. **Ablation 锚定在 ADiT-M**：Table 4 标题即 "Ablation study … **based on ADiT-M**"，并把"模型规模"当**单因子开关**扫（`#3 w/ larger size`→L、`#4 w/ smaller size`→S）⇒ ADiT-M 是参考模型，S/L 在其配方上"只改尺寸"。
2. **D.4 只给一组训练设置、无规模区分**：对照 D.3（Davis）明确写 "lr = 1e-4 for S and 3e-5 for M/L"——可见作者若按规模区分会明说；SKEMPI 只有单一一个 lr。
3. **Pretraining 也是"协议统一、只为显存调 batch 组成"**：Table 5 中 S/M/L 的 training steps 全是 180k，唯一差异是 L 因显存把 batch/GPU·grad-acc·#GPU 拆成不同组合（属工程适配，非超参择优）。

**这套配方怎么"定"下来的**：基本是**继承 + 轻量验证**——split-by-complex 三折与突变-patch 截断都是 follow DiffAffinity / RDE（Liu et al. 2024）的既定协议；`monitor: val/mse_loss` 的 early-stop / ReduceLROnPlateau 只用 5% 验证集**选 checkpoint、降 lr**（是固定超参下选模型，不是搜超参）。

> 为什么"统一超参"反而正确：scaling study 必须**把训练协议钉死**，才能把性能提升干净归因于"模型变大"，否则 size 与 HP 混淆。所以三规模共享配方是**受控变量设计**。**诚实 caveat**：论文并不声称这套超参对每个规模都最优，统一配方可能让最小的 S 或最大的 L 略欠调。

---

## 5. 输入截断机制：突变中心的 3D spatial patch（不是序列 ±50）

**易误解点**：所谓"截断到 50 残基"**不是序列上取突变前后各若干个**，而是 **取离突变位点 3D 空间最近的 50 个残基**。代码路径（`adit/data/components/data_utils.py:45–100`）：

1. `extract_dest_protein_and_mutation_mask`：先保留 **chain A ∪ chain B 的全部残基**（整个二元复合物），并标出突变位点；
2. `truncation_skempi`：对每个残基算其 Cα 到**最近突变位点**的 3D 欧氏距离，`topk(k=50, largest=False)` 取**最近的 50 个**（突变位点距离置 0 必留）；`consecutive=false` ⇒ **不强制序列连续**。

⇒ 实际喂给模型的是一个**以突变为中心、跨两条链、覆盖局部界面邻域的 50 残基 spatial patch**。

**为什么取 50（三个叠加原因）**：
- **沿用 RDE / DiffAffinity 的 mutation-patch 范式**：ΔΔG_bind 主要由突变局部界面环境决定（论文 D.4：truncate based on distance to mutation-related residues, ≤50）。RDE 类用 128，ADiT 取更紧的 50。
- **all-atom 显存约束（主因）**：原子级 token 数大、token 级注意力 ~O(N²)，小 patch 才能单卡 batch 训练。
- **信噪比**：只看决定 ΔΔG 的局部，远端残基不进来当噪声。

**可否扩成 full-complex**：机制上能（调 `truncation_size` 或关 `skempi_truncate`）。便宜的中间路线是先把 50 → **128 / 256**（对齐 RDE 量级）重新 fine-tune 做敏感性实验；真·full-complex 的障碍主要是显存（见 §6）。

---

## 6. 预训练目标与上下文长度：crop≤250 并不锁死下游

- **预训练 = denoising diffusion 重建结构**：给原子坐标加噪再去噪；fine-tune 时 time step 固定为 0（干净样本），即把预训练主干**当结构特征提取器**用。
- **"crop" = 局部 patch 重建**：每个预训练样本不是整条大复合物，而是 `continues_crop`（连续片段）/ `spatial_crop`（空间邻域）裁出的 **≤250 残基 patch**（Table 5：S/M=250、L=200），因 GPU 显存约束（D.1）。⇒ 模型只在 ≤250 的局部 patch 上练过。
- **关键：下游上下文长度不必与预训练 crop 一致。** ADiT 用 **AlphaFold 式相对位置编码**（`relative_position_encoder.py`，`r_max=32`）：残基相对偏移 **clamp 到 [−32,+32]** + 一个"跨链"bin，**完全 length-agnostic、无学习到的绝对位置**。per-token / per-atom 特征照常产出，下游 pool + MLP 头重训即可消费——**预训练 crop=250 不是架构硬墙**。

**修正后的"扩 full-complex"障碍清单**：

| 障碍 | 判断 |
|---|---|
| **显存 / 算力** | **唯一硬约束**：token 级注意力 ~O(N²)（50→256 约 26×，→1000+ 更贵）+ 原子展开 + ESM。工程问题，非不可能。 |
| 预训练分布不一致 | **软问题**，且被相对位置 clamp 削弱：>250 是没练过的 regime，但位置编码本就局部+length-invariant，顶多轻微分布漂移，**非"做不到"**。 |
| 下游头一致性 | **非障碍**：换上下文重新 fine-tune 头（乃至主干）即可，成本低。 |

> 结论：扩到 full-complex 主要是**显存工程 + 重新 fine-tune**，不是架构不可能。剩下真正开放的是**经验问题**——更多上下文是否提升 ΔΔG？先验上不乐观（RDE/DiffAffinity 故意 crop、局部主导），但值得用 **50→128→256 消融**实测。

---

## 7. 论文 vs 本仓库复现：唯一实质差异 = learning rate

| | 论文 D.4 | 本仓库 `skempi_{S,M,L}.yaml` |
|---|---|---|
| learning rate | **3e-5**（S=M=L） | **1e-4**（S=M=L） |
| batch / dropout / epochs / 截断 / split | 8 / 0.0 / 60 / 50 / by-complex 3-fold | 完全一致 ✓ |

- 其余训练与架构超参，仓库与论文**逐项吻合**。
- 仅 **lr 不同**：论文 3e-5、仓库 1e-4。仓库在 base `model/skempi.yaml` 与三个 experiment 配置里都写死 1e-4。
- 论文里**唯一**出现"lr 按规模不同"的是 Davis（D.3）；SKEMPI（D.4）与本仓库都对 S/M/L 用**同一个 lr**，
  只是数值不同（3e-5 vs 1e-4）。复现/对齐论文时若要严格一致，应把三个 experiment 配置的 `model.optimizer.lr` 改为 `3e-5`。

---

## 8. 数据来源
- 论文：`Sources/ADiT.pdf` — §3.4 Pre-training & Fine-tuning、§4.1 Pre-training Setup、§4.4 Protein-Protein、
  Table 4（ablation on ADiT-M）、Appendix **Table 5**（Model Hyperparameters）、**Table 6**（Dataset statistics）、**D.1/D.4**。
- 仓库：`configs/experiment/skempi_{S,M,L}.yaml`、`configs/model/skempi.yaml`、`configs/data/skempi.yaml`；
  截断逻辑 `adit/data/components/{dataset.py,data_utils.py}`；相对位置编码 `adit/models/net/adit/relative_position_encoder.py`；
  运行入口 `reproduce/run_ppi_skempi.sh`；实测参数量 `outputs/.../skempi_S_split_0_*/csv/version_0/hparams.yaml`。
