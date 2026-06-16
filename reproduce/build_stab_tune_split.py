#!/usr/bin/env python
"""
build_stab_tune_split.py  ——  adit-retune-hyperparams 的 HP-search 验证集构建器

目的:在 **不碰 stab_test** 的前提下,把已建好的 `dataset/skempi_stab/stab_train`(120 complexes /
2952 ADiT records)按 **homology cluster 整簇留出 ~20%** 当 HP 搜索的验证集,产出:
  * dataset/skempi_stab_tune/tune_train/  (符号链接 -> stab_train 里的 record)
  * dataset/skempi_stab_tune/tune_val/    (符号链接;= 留出的同源簇)

VAL 簇由 StaB-ddG 的 `super_clusters`(界面同源)在 **train 复合物上做并查集** 得到,再贪心按 #records
留出 ~20%(seed=42),且保证 val 里含足够多 ≥10 突变的 complex(本次 16 个),使 per-interface
Spearman(K≥10)目标稳定。该 VAL 复合物清单已**离线算好、固化在下方常量**,保证 local / ibex 一致、可复现。
(复算见 reproduce 注释脚本;源数据 StaB filtered_skempi.csv + train_pdb.pkl。)

⚠ test(stab_test)全程不参与:tune split 只切分 stab_train。

用法:
  python reproduce/build_stab_tune_split.py            # 建 symlink
  python reproduce/build_stab_tune_split.py --dry-run  # 只统计
"""
import os, glob, argparse, pickle, csv
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAB_TRAIN = os.path.join(REPO, "dataset", "skempi_stab", "stab_train")
TUNE_DIR = os.path.join(REPO, "dataset", "skempi_stab_tune")
TUNE_TRAIN = os.path.join(TUNE_DIR, "tune_train")
TUNE_VAL = os.path.join(TUNE_DIR, "tune_val")
MANIFEST = os.path.join(REPO, "reproduce", "SP-STAB-tune_manifest.csv")

# 离线算好的 homology-aware VAL 复合物(23 簇 / 35 complexes / ~637 records / 16 个 K>=10),seed=42。
VAL_COMPLEXES = {
    "1A22_A_B", "1B2S_A_D", "1B2U_A_D", "1B3S_A_D", "1BP3_A_B", "1BRS_A_D", "1C4Z_ABC_D",
    "1EMV_A_B", "1FFW_A_B", "1IAR_A_B", "1QAB_ABCD_E", "1YCS_A_B", "2B42_A_B", "2C5D_AB_CD",
    "2C5D_A_C", "2DVW_A_B", "2J0T_A_D", "2WPT_A_B", "3AAA_AB_C", "3BT1_A_U", "3EG5_A_B",
    "3EQS_A_B", "3EQY_A_C", "3F1S_A_B", "3MZG_A_B", "3SZK_AB_C", "3VR6_ABCDEF_GH", "4HRN_A_D",
    "4JEU_A_B", "4OFY_A_D", "4PWX_AB_CD", "4RA0_A_C", "5CXB_A_B", "5CYK_A_B", "5E6P_A_B",
}


def complex_id(fname):
    return "_".join(os.path.basename(fname)[:-4].split("_")[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(STAB_TRAIN, "*.pkl")))
    if not files:
        raise SystemExit(f"[err] 没找到 {STAB_TRAIN}/*.pkl —— 先跑 build_stab_split.py 建好 stab_train")

    rows, n_val, n_train = [], 0, 0
    val_comps_seen, train_comps_seen = set(), set()
    to_link = []  # (dst_dir, src_real, name)
    for f in files:
        cid = complex_id(f)
        is_val = cid in VAL_COMPLEXES
        dst_dir = TUNE_VAL if is_val else TUNE_TRAIN
        src_real = os.path.realpath(f)  # stab_train 里可能本就是 symlink,解析到真实 record
        to_link.append((dst_dir, src_real, os.path.basename(f)))
        if is_val:
            n_val += 1; val_comps_seen.add(cid)
        else:
            n_train += 1; train_comps_seen.add(cid)
        rows.append({"file": os.path.basename(f), "complex": cid,
                     "split": "tune_val" if is_val else "tune_train"})

    print(f"[tune] stab_train files={len(files)} -> tune_train={n_train} ({len(train_comps_seen)} complexes) | "
          f"tune_val={n_val} ({len(val_comps_seen)} complexes)")
    missing = VAL_COMPLEXES - val_comps_seen
    if missing:
        print(f"[warn] VAL_COMPLEXES 中有 {len(missing)} 个未在 stab_train 出现(可能因丢 3 complex/撞名): {sorted(missing)}")

    if args.dry_run:
        print("[dry-run] 不建文件。")
        return

    for d in (TUNE_TRAIN, TUNE_VAL):
        os.makedirs(d, exist_ok=True)
    built = 0
    for dst_dir, src_real, name in to_link:
        dst = os.path.join(dst_dir, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src_real, dst)
        built += 1
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "complex", "split"])
        w.writeheader(); w.writerows(rows)
    print(f"[done] 写入 {TUNE_DIR}/(tune_train,tune_val);symlinks={built};manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
