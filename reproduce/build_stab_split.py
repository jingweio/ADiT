#!/usr/bin/env python
"""
build_stab_split.py  ——  Task 2 (adit-ood-analysis) 的 SP-STAB split 构建器

目的:在 **完全继承 StaB-ddG 的 benchmark**(它的 samples / ddG label / 70-30 cluster split)
上准备 ADiT 的输入,用来把 ADiT-S 的结果与 StaB-ddG 论文报告值对比。

数据 (DS-STAB,见 design doc):
  * StaB-ddG/data/SKEMPI/filtered_skempi.csv  (4541 mutants / 201 complexes,含 ddG)
  * StaB-ddG/data/SKEMPI/{train_pdb.pkl,test_pdb.pkl}  (#Pdb 列表:120 train / 81 test)

策略 (决策 B1):ADiT 的 structure preprocessing 需 FoldX(缺失),无法 back-fill ADiT 没有的
3 个复合物 (1NMB_N_LH, 1NCA_N_LH, 1KBH_A_B) => **丢弃这 3 个**,在 198/201 上跑;md 注明缺 ~1.5%。

做法:把每个 StaB 样本 (复合物 #Pdb + mutation set + StaB 的 ddG) 匹配到 ADiT 已有的 record pkl
(join key = (complex_id, frozenset(mutations)),mutation 写法两边一致如 'HA18A')。匹配上的 record:
  * 若 ADiT 自带 ddG 与 StaB ddG 在 tol 内一致 => 直接 symlink(省盘);
  * 否则 => 复制并把 record['ddG'] 覆盖为 StaB 的 ddG(忠实继承 StaB label)。
文件名保持不变(accession),ESM (.pt) 仍按 accession 解析。

输出:
  * dataset/skempi_stab/{stab_train,stab_test}/*.pkl
  * reproduce/SP-STAB_manifest.csv   (sample -> split -> matched? -> ddG_stab/ddG_adit)
  * stdout:matched / unmatched / dropped 统计 + ddG 一致性 + 各 split 样本数

用法:
  python reproduce/build_stab_split.py            # 构建
  python reproduce/build_stab_split.py --dry-run  # 只分析匹配/ddG,不建文件
"""
import os, glob, argparse, pickle, csv
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEMPI_DIR = os.path.join(REPO, "dataset", "skempi")
BYCX_SPLITS = ["split_0", "split_1", "split_2"]
STAB_DIR = os.path.join(REPO, "dataset", "skempi_stab")
STAB_FOLDS = {"train": "stab_train", "test": "stab_test"}
STAB_DATA = "/home/guoj0f/repos/StaB-ddG/data/SKEMPI"
FILTERED = os.path.join(STAB_DATA, "filtered_skempi.csv")
TRAIN_PDB = os.path.join(STAB_DATA, "train_pdb.pkl")
TEST_PDB = os.path.join(STAB_DATA, "test_pdb.pkl")
MANIFEST = os.path.join(REPO, "reproduce", "SP-STAB_manifest.csv")
DDG_TOL = 0.05  # kcal/mol;两边 ddG 差小于它就视为一致 -> symlink


def complex_id(fname):
    return "_".join(os.path.basename(fname)[:-4].split("_")[:3])


def mutset(s):
    return frozenset(t.strip() for t in str(s).split(",") if t.strip())


def index_adit():
    """(complex_id, mutset) -> {'path':abs, 'ddG':float}. 读 record 的 mutation/ddG 字段。"""
    idx = {}
    dup = 0
    for sp in BYCX_SPLITS:
        for f in glob.glob(os.path.join(SKEMPI_DIR, sp, "*.pkl")):
            try:
                rec = pickle.load(open(f, "rb"))
            except Exception:
                continue
            key = (complex_id(f), mutset(rec.get("mutation", "")))
            if key in idx:
                dup += 1
                continue
            idx[key] = {"path": os.path.abspath(f), "ddG": float(rec.get("ddG"))}
    return idx, dup


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stab-data", default=STAB_DATA,
                    help="StaB-ddG SKEMPI 元数据目录(filtered_skempi.csv + train_pdb.pkl + test_pdb.pkl);"
                         "ibex 上需指向同步过去的副本")
    args = ap.parse_args()

    filt = pd.read_csv(os.path.join(args.stab_data, "filtered_skempi.csv"))
    train_pdb = set(pickle.load(open(os.path.join(args.stab_data, "train_pdb.pkl"), "rb")))
    test_pdb = set(pickle.load(open(os.path.join(args.stab_data, "test_pdb.pkl"), "rb")))
    print(f"[stab] {len(filt)} samples / {filt['#Pdb'].nunique()} complexes "
          f"| train_pdb {len(train_pdb)} test_pdb {len(test_pdb)}")

    adit_idx, dup = index_adit()
    print(f"[adit] indexed {len(adit_idx)} (complex,mutset) records (dup skipped={dup})")

    rows, matched, unmatched = [], 0, 0
    ddg_max_abs = 0.0
    override = 0
    unmatched_complexes = defaultdict(int)
    split_counts = defaultdict(int)
    to_build = []  # (split_key, adit_path, stab_ddG, need_override)

    for _, r in filt.iterrows():
        cid = r["#Pdb"]
        ms = mutset(r["Mutation(s)_cleaned"])
        stab_ddg = float(r["ddG"])
        split = "train" if cid in train_pdb else ("test" if cid in test_pdb else None)
        if split is None:
            continue  # 不在 StaB split 里(不应发生)
        hit = adit_idx.get((cid, ms))
        if hit is None:
            unmatched += 1
            unmatched_complexes[cid] += 1
            rows.append({"complex": cid, "mut": ",".join(sorted(ms)), "split": split,
                         "matched": 0, "ddG_stab": stab_ddg, "ddG_adit": ""})
            continue
        matched += 1
        d = abs(hit["ddG"] - stab_ddg)
        ddg_max_abs = max(ddg_max_abs, d)
        need_override = d > DDG_TOL
        override += int(need_override)
        split_counts[split] += 1
        to_build.append((split, hit["path"], stab_ddg, need_override))
        rows.append({"complex": cid, "mut": ",".join(sorted(ms)), "split": split,
                     "matched": 1, "ddG_stab": stab_ddg, "ddG_adit": hit["ddG"]})

    print(f"[match] matched={matched} | unmatched={unmatched}")
    print(f"[match] unmatched complexes: "
          f"{dict(sorted(unmatched_complexes.items(), key=lambda x:-x[1]))}")
    print(f"[ddG] max|ddG_adit-ddG_stab|={ddg_max_abs:.4f} | 需 override 的样本数={override} "
          f"(tol={DDG_TOL})")
    print(f"[split] stab_train={split_counts['train']} | stab_test={split_counts['test']}")

    if args.dry_run:
        print("[dry-run] 不建文件。")
        return

    # 物化
    for k in STAB_FOLDS.values():
        os.makedirs(os.path.join(STAB_DIR, k), exist_ok=True)
    built = 0
    for split, src, stab_ddg, need_override in to_build:
        dst = os.path.join(STAB_DIR, STAB_FOLDS[split], os.path.basename(src))
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        if need_override:
            rec = pickle.load(open(src, "rb"))
            rec["ddG"] = stab_ddg  # 忠实继承 StaB label
            import numpy as np
            rec["ddG"] = np.float32(stab_ddg)
            pickle.dump(rec, open(dst, "wb"))
        else:
            os.symlink(src, dst)
        built += 1
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["complex", "mut", "split", "matched",
                                          "ddG_stab", "ddG_adit"])
        w.writeheader(); w.writerows(rows)
    print(f"[done] SP-STAB 写入 {STAB_DIR}/(stab_train,stab_test);built={built};"
          f"override(copy)={override};manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
