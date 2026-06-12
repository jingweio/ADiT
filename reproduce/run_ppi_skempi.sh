#!/bin/bash
# ============================================================================
# ADiT 复现 —— 蛋白-蛋白 突变效应预测(SKEMPIv2 / PPI mutation-effect prediction)
#   任务   : 预测 ddG(突变引起的结合自由能变化 ΔΔG)
#   模型   : ADiT-S(换成 experiment=skempi_M / skempi_L 即可用更大模型)
#   划分   : test_split = split_0(三折交叉验证中的其中一折)
#
# 本文件是“命令参考库”:source 之后调用其中的函数即可,例如:
#   source reproduce/run_ppi_skempi.sh && run_step_A_esm_repr
#
# ddG 由 SkempiModule 以“反对称差”方式预测:
#   pred = MLP([mt,wt]) - MLP([wt,mt])   (对 wt 和 mt 各跑一遍主干网络)
#
# 前置(各跑一次):  bash reproduce/setup_env.sh ; bash reproduce/download_assets.sh
# ============================================================================
# 注意:本文件是给 `source` 用的,所以不要用 `set -euo pipefail`——
# 那会把 errexit 带进你的交互终端,任何命令返回非零都会直接关闭终端。
cd "$(dirname "${BASH_SOURCE[0]}")/.."        # 切到仓库根目录(source 时用 BASH_SOURCE)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate adit
# 让 CUDA 的 GPU 编号和 nvidia-smi 一致(CUDA 默认按算力排序,会把两张卡的序号弄反)。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export GPU=1                                  # nvidia-smi 的 1 号卡 = RTX A4500（20GB）

# ----------------------------------------------------------------------------
# 步骤 A（必需，任何 skempi 训练/测试之前先跑一次）。
# 为每个 SKEMPI 复合物（野生型 wt + 突变型 mt）预计算冻结的 ESM-2 表征，
# 存到 dataset/skempi_esm_repr_a100/{wt,mt}_<code>.pt（canonical = a100 重算版；A4500 旧版已删）。SkempiModule 直接读这些文件，
# 而不是每一步都重跑 ESM（同一个 wt/mt 蛋白会被很多突变复用，预计算省很多时间）。
# 耗时约 1 小时（~6.7k 个复合物）。
#   （脚本里还会调用 main_HER2()；因为没下载 HER2 数据，这部分是空操作——
#     'HER2_esm_repr' 是抗体-抗原任务才会用到的目录，与本任务无关。）
# ----------------------------------------------------------------------------
run_step_A_esm_repr() {
  CUDA_VISIBLE_DEVICES=$GPU python scripts/dump_esm_repr.py
}

# ----------------------------------------------------------------------------
# 冒烟测试 —— 只训 1 个 epoch、几个 batch。用于验证链路能跑通，不是真实结果。
# 覆盖链路：数据 -> 预计算ESM -> ADiT(对 wt/mt 各一遍) -> MLP -> 反向传播
#          -> 存checkpoint -> 测试 -> ddG 指标 -> 落 pkl。
# ----------------------------------------------------------------------------
smoke_skempi() {
  CUDA_VISIBLE_DEVICES=$GPU bash train.sh experiment=skempi_S \
    ++trainer.devices=1 ++trainer.strategy=auto ++trainer.min_epochs=1 ++trainer.max_epochs=1 \
    ++data.batch_size=2 \
    ++data.dataset.test_split=split_0 \
    +trainer.limit_train_batches=4 +trainer.limit_val_batches=4 +trainer.limit_test_batches=4 \
    ++callbacks.model_checkpoint.save_top_k=1 \
    ++model.save_file=reproduce/smoke_skempi_split_0.pkl \
    test=true \
    ckpt_path=ckpts/adit_S.ckpt \
    task_name=smoke_skempi_S_split_0
}

# ----------------------------------------------------------------------------
# 全量复现（请自行启动；耗时长，60 个 epoch）。需先完成步骤 A。
# skempi_S 配置 save_top_k=0 但 save_last=true，所以用 last.ckpt 来测试，路径在：
#   outputs/<年-月-日>/skempi_S_split_0_<时-分-秒>/checkpoints/last.ckpt
# ----------------------------------------------------------------------------
# 用法: full_skempi_train [split]   split 默认 split_0,可传 split_0/split_1/split_2
full_skempi_train() {
  local SPLIT="${1:-split_0}"
  CUDA_VISIBLE_DEVICES=$GPU bash train.sh experiment=skempi_S \
    ++trainer.devices=1 ++trainer.strategy=auto ++data.batch_size=8 \
    ++data.dataset.test_split="$SPLIT" \
    ckpt_path=ckpts/adit_S.ckpt \
    task_name="skempi_S_$SPLIT"
}

# 用法: full_skempi_test <ckpt路径> [split]   split 默认 split_0,需与 ckpt 对应
full_skempi_test() {
  local CKPT="$1"; local SPLIT="${2:-split_0}"
  CUDA_VISIBLE_DEVICES=$GPU bash test.sh experiment=skempi_S \
    ++trainer.devices=1 ++trainer.strategy=auto ++data.batch_size=1 \
    ++data.dataset.test_split="$SPLIT" \
    ++model.save_file="reproduce/result_$SPLIT.pkl" \
    ckpt_path="$CKPT"
  # 打印出来的 test/pearsonr、test/spearmanr、test/mse_loss、test/mae_loss
  # 就是该 split 这一折的指标;预测落盘到 reproduce/result_<split>.pkl。
}

# ----------------------------------------------------------------------------
# （可选）论文 Table 2 的数字 = 三折拼接后的整体指标。要对齐它，
# 需把 split_1 和 split_2 也各训练+测试一遍（分别存 result_split_1.pkl /
# result_split_2.pkl），然后拼接计算整体指标：
#   python scripts/skempi_metric.py -i reproduce
# ----------------------------------------------------------------------------
full_skempi_aggregate_3folds() {
  python scripts/skempi_metric.py -i reproduce
}

echo "source 本文件后可调用: run_step_A_esm_repr | smoke_skempi | full_skempi_train [split] | full_skempi_test <ckpt> [split] | full_skempi_aggregate_3folds"
