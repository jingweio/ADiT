#!/bin/bash
# ============================================================================
# ADiT 复现 —— 蛋白-配体 结合亲和力预测(LBA / Protein-Ligand binding affinity)
#   任务   : 回归 neglog_aff（蛋白-配体复合物的结合亲和力）
#   模型   : ADiT-S(换成 experiment=lba_M / lba_L 即可用更大模型)
#   划分   : split_identity_threshold = identity_30（论文主推的 LBA-30 基准）
#
# 本文件是“命令参考库”:source 之后调用其中的函数即可,例如:
#   source reproduce/run_pl_lba.sh && smoke_lba
#
# 关于 ESM:LBA 不需要离线预计算 ESM。模型在 SeqEmbedder 内部于运行时实时计算
# ESM-2-650M 表征(LBA 不传入预计算的 esm_repr)。这一点与 SKEMPI/PPI 脚本不同,
# 后者是离线预计算 ESM 的。
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
# 冒烟测试 —— 只训 1 个 epoch、几个 batch。用于验证链路能跑通,不是真实结果。
# 覆盖链路:数据 -> 模型内实时ESM -> ADiT -> MLP -> 反向传播 -> 存checkpoint -> 测试 -> 指标。
# ----------------------------------------------------------------------------
smoke_lba() {
  CUDA_VISIBLE_DEVICES=$GPU bash train.sh experiment=lba_S \
    ++trainer.devices=1 ++trainer.strategy=auto ++trainer.min_epochs=1 ++trainer.max_epochs=1 \
    ++data.batch_size=2 \
    ++data.dataset.split_identity_threshold=identity_30 \
    +trainer.limit_train_batches=4 +trainer.limit_val_batches=4 +trainer.limit_test_batches=4 \
    ++callbacks.model_checkpoint.save_top_k=1 \
    test=true \
    ckpt_path=ckpts/adit_S.ckpt \
    task_name=smoke_lba_S
}

# ----------------------------------------------------------------------------
# 全量复现（请自行启动；耗时长）。微调产出的 checkpoint 在：
#   outputs/<年-月-日>/lba_S_id30_<时-分-秒>/checkpoints/   （用 find outputs -name '*.ckpt' 查找）
# README 原版用 devices=2 batch_size=16；这里改成单卡。
# 注意:experiment=lba_S 在 identity_30 路径下自带 max_epochs=4。
#       若要跑 identity_60,论文会加 ++trainer.max_epochs=100。
# ----------------------------------------------------------------------------
# 用法: full_lba_train [identity_30|identity_60]   默认 identity_30
#   identity_30 用 lba_S 配置默认 max_epochs=4;identity_60 按论文加 ++trainer.max_epochs=100
full_lba_train() {
  local THR="${1:-identity_30}"
  local EXTRA=""
  [ "$THR" = "identity_60" ] && EXTRA="++trainer.max_epochs=100"
  CUDA_VISIBLE_DEVICES=$GPU bash train.sh experiment=lba_S \
    ++trainer.devices=1 ++trainer.strategy=auto ++data.batch_size=8 \
    ++data.dataset.split_identity_threshold="$THR" $EXTRA \
    ckpt_path=ckpts/adit_S.ckpt \
    task_name="lba_S_$THR"
}

# 用法: full_lba_test <ckpt路径> [identity_30|identity_60]   需与 ckpt 对应
full_lba_test() {
  local CKPT="$1"; local THR="${2:-identity_30}"
  CUDA_VISIBLE_DEVICES=$GPU bash test.sh experiment=lba_S \
    ++trainer.devices=1 ++trainer.strategy=auto ++data.batch_size=1 \
    ++data.dataset.split_identity_threshold="$THR" \
    ++model.save_file="reproduce/result_lba_$THR.pkl" \
    ckpt_path="$CKPT"
  # 测试指标 test/mse_loss（RMSE=√mse）、test/pearsonr、test/spearmanr 由 adit/test.py 打印;
  # 预测落盘 reproduce/result_lba_<threshold>.pkl(LBA 为单一 train/val/test split,无需三折汇总)。
}

echo "source 本文件后可调用: smoke_lba | full_lba_train [identity_30|identity_60] | full_lba_test <ckpt> [identity_30|identity_60]"
