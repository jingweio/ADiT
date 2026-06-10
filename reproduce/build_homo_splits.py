#!/usr/bin/env python
"""
build_homo_splits.py  ——  Task 1 (adit-ood-analysis) 的 SP-HOMO split 构建器

目的:在 ADiT 自己的 SKEMPI 数据 (DS-ADIT, 341 complexes) 上,用 **StaB-ddG 完全相同的
interface-homology clustering 配方** 重建一个 homology-OOD 的 3-fold split (SP-HOMO),
用来与原始 by-complex 3-fold (SP-BYCX, dataset/skempi/split_{0,1,2}) 对比。

关键设计 (见 ibex_records/adit-ood-analysis/adit_ood_analysis_design_*.md):
  * 只重排现有 record pkl(symlink),不复制、不重算 preprocess;ESM 按 filename-stem
    (accession) 索引,文件名不变 => ESM 解析照常。
  * clustering 逐字复刻 StaB-ddG/data/SKEMPI/skempi_splits.ipynb 的 transitive-closure 逻辑
    (token set 两两有交集就 merge;Pr/PI、AB/AG 这类 category code 也作为 merge-bridge 保留;
    每个 complex 用其 Hold_out_proteins 的 first token 归属 cluster)。
  * 整个 cluster 整体分到某一折(绝不跨折);3 折按 **#records 平衡**(greedy 最小堆式)。

输出:
  * dataset/skempi_homo/homo_{0,1,2}/*.pkl  (symlink 到 dataset/skempi/split_*/ 下的原文件)
  * reproduce/SP-HOMO_manifest.csv          (complex -> cluster_id -> homo_fold,留证 & 防混淆)
  * stdout:各折 records/complexes/clusters 统计 + 断言无 cluster 跨折
            + leakage 诊断:有多少 homology cluster 跨越了原始 SP-BYCX 三折

用法:
  python reproduce/build_homo_splits.py            # 构建
  python reproduce/build_homo_splits.py --dry-run  # 只算统计,不建 symlink
"""
import os, glob, argparse, pickle, csv
from collections import defaultdict

# ---- 路径 (相对 ADiT repo 根目录) ----
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEMPI_DIR   = os.path.join(REPO, "dataset", "skempi")            # SP-BYCX 源
BYCX_SPLITS  = ["split_0", "split_1", "split_2"]
HOMO_DIR     = os.path.join(REPO, "dataset", "skempi_homo")       # SP-HOMO 目标
HOMO_FOLDS   = ["homo_0", "homo_1", "homo_2"]
SKEMPI_V2CSV = "/home/guoj0f/repos/StaB-ddG/data/SKEMPI/skempi_v2.csv"  # 含 Hold_out_proteins
MANIFEST     = os.path.join(REPO, "reproduce", "SP-HOMO_manifest.csv")


def complex_id_from_filename(fname):
    """PDB_ChainA_ChainB_mutations_index.pkl -> 'PDB_ChainA_ChainB' (前 3 个 _ token)."""
    toks = os.path.basename(fname)[:-4].split("_")
    return "_".join(toks[:3])


def load_bycx():
    """返回 complex_id -> {'records': [abs pkl paths], 'bycx_fold': split_name}."""
    comp = {}
    for sp in BYCX_SPLITS:
        for f in glob.glob(os.path.join(SKEMPI_DIR, sp, "*.pkl")):
            cid = complex_id_from_filename(f)
            d = comp.setdefault(cid, {"records": [], "bycx_fold": sp})
            d["records"].append(os.path.abspath(f))
            # 同一 complex 的所有 record 都在同一 SP-BYCX 折(已验证),记录其折
            d["bycx_fold"] = sp
    return comp


def build_clusters(complex_ids, strip_category=False, v2csv=SKEMPI_V2CSV):
    """逐字复刻 StaB-ddG skempi_splits.ipynb 的 cluster 构建,但作用于给定 complex 集合。

    strip_category=True 时,丢弃 Hold_out_proteins 里的 functional-category token
    (含 '/' 的,如 Pr/PI、AB/AG、TCR/pMHC),只按真正的 PDB-interface homology 聚类
    (避免一个 category 把整类复合物 merge 成一个 mega-cluster)。

    返回:
      cid_to_cluster: complex_id -> cluster_key (稳定字符串)
    """
    import pandas as pd
    raw = pd.read_csv(v2csv, sep=None, engine="python")
    # #Pdb -> Hold_out_proteins(每个 complex 取第一条非空)
    hop_map = {}
    for pdb, sub in raw.groupby("#Pdb"):
        vals = sub["Hold_out_proteins"].dropna().tolist()
        if vals:
            hop_map[pdb] = vals[0]

    def toks_of(s):
        ts = [t for t in s.split(",") if t]
        if strip_category:
            ts = [t for t in ts if "/" not in t]
        return ts

    # 为每个 ADiT complex 取其 token 列表;查不到或被清空 -> 视为自身单点 cluster
    cid_to_toks = {}
    missing = []
    for cid in complex_ids:
        if cid in hop_map:
            ts = toks_of(hop_map[cid])
            cid_to_toks[cid] = ts if ts else [cid]
        else:
            cid_to_toks[cid] = [cid]  # singleton bridge = 自己
            missing.append(cid)

    # ---- StaB-ddG 的 transitive closure(token set 两两有交集即 merge)----
    hold_out_tok_lists = list(dict.fromkeys(",".join(v) for v in cid_to_toks.values()))
    super_clusters = [set(x.split(",")) for x in hold_out_tok_lists]
    joined = True
    while joined:
        joined = False
        new_sc = []
        while super_clusters:
            cur = super_clusters.pop(0)
            i = 0
            while i < len(super_clusters):
                if cur & super_clusters[i]:
                    cur |= super_clusters[i]
                    super_clusters.pop(i)
                    joined = True
                else:
                    i += 1
            new_sc.append(cur)
        super_clusters = new_sc

    token_to_cluster = {}
    for cluster in super_clusters:
        key = ",".join(sorted(cluster))  # 稳定的 cluster id
        for tok in cluster:
            token_to_cluster[tok] = key

    # 每个 complex 用其 token 列表的 first token 归属(同 StaB 的 x[0])
    cid_to_cluster = {}
    for cid, toks in cid_to_toks.items():
        cid_to_cluster[cid] = token_to_cluster[toks[0]]
    return cid_to_cluster, missing


def greedy_balance(clusters_with_size, k=3):
    """把 cluster 整体分到 k 折,按 #records 平衡(每次给当前最小折)。
    clusters_with_size: list[(cluster_key, n_records)] ;返回 cluster_key -> fold_idx。"""
    order = sorted(clusters_with_size, key=lambda x: (-x[1], x[0]))  # 大到小,稳定
    fold_load = [0] * k
    assign = {}
    for ckey, n in order:
        j = min(range(k), key=lambda i: (fold_load[i], i))
        assign[ckey] = j
        fold_load[j] += n
    return assign, fold_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只算统计,不建 symlink/manifest")
    ap.add_argument("--strip-category", action="store_true",
                    help="聚类时丢弃 Pr/PI、AB/AG 等 functional-category token,只按 PDB-interface homology")
    ap.add_argument("--skempi-v2-csv", default=SKEMPI_V2CSV,
                    help="raw SKEMPI v2 csv(含 Hold_out_proteins);ibex 上需指向同步过去的副本")
    args = ap.parse_args()

    comp = load_bycx()
    print(f"[load] DS-ADIT: {len(comp)} complexes, "
          f"{sum(len(v['records']) for v in comp.values())} records "
          f"| strip_category={args.strip_category}")

    cid_to_cluster, missing = build_clusters(list(comp.keys()),
                                             strip_category=args.strip_category,
                                             v2csv=args.skempi_v2_csv)
    if missing:
        print(f"[warn] {len(missing)} complexes 在 skempi_v2.csv 查不到 Hold_out_proteins,"
              f"按 singleton 处理: {missing}")

    # cluster -> records 数 / complexes
    cluster_records = defaultdict(int)
    cluster_comps = defaultdict(list)
    for cid, ckey in cid_to_cluster.items():
        cluster_records[ckey] += len(comp[cid]["records"])
        cluster_comps[ckey].append(cid)
    print(f"[cluster] {len(cluster_records)} homology clusters over {len(comp)} complexes")

    # ---- leakage 诊断:有多少 cluster 跨越原始 SP-BYCX 三折 ----
    straddle = 0
    for ckey, cids in cluster_comps.items():
        folds = set(comp[c]["bycx_fold"] for c in cids)
        if len(folds) > 1:
            straddle += 1
    print(f"[leakage] {straddle}/{len(cluster_records)} homology clusters 跨越了原始 "
          f"SP-BYCX 三折(即 ADiT by-complex split 默许的 homology 泄漏量)")

    # ---- 3 折按样本数平衡 ----
    assign, fold_load = greedy_balance(list(cluster_records.items()), k=3)
    print("[balance] SP-HOMO 3-fold(按 #records 平衡):")
    for i, fold in enumerate(HOMO_FOLDS):
        ncl = sum(1 for v in assign.values() if v == i)
        ncomp = sum(len(cluster_comps[ck]) for ck, v in assign.items() if v == i)
        print(f"    {fold}: {fold_load[i]} records | {ncomp} complexes | {ncl} clusters")

    if args.dry_run:
        print("[dry-run] 不建 symlink。")
        return

    # ---- 物化 symlink + manifest ----
    rows = []
    for ck, fold_idx in assign.items():
        fold = HOMO_FOLDS[fold_idx]
        out_dir = os.path.join(HOMO_DIR, fold)
        os.makedirs(out_dir, exist_ok=True)
        for cid in cluster_comps[ck]:
            for src in comp[cid]["records"]:
                dst = os.path.join(out_dir, os.path.basename(src))
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            rows.append({"complex": cid, "cluster_id": ck[:60],
                         "homo_fold": fold, "bycx_fold": comp[cid]["bycx_fold"],
                         "n_records": len(comp[cid]["records"])})
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["complex", "cluster_id", "homo_fold",
                                          "bycx_fold", "n_records"])
        w.writeheader(); w.writerows(rows)
    print(f"[done] SP-HOMO 写入 {HOMO_DIR}/(homo_0,1,2 symlink);manifest -> {MANIFEST}")

    # 断言无 cluster 跨折
    fold_of_cluster = defaultdict(set)
    for ck, v in assign.items():
        fold_of_cluster[v].add(ck)
    cross = 0
    seen = {}
    for v, cks in fold_of_cluster.items():
        for ck in cks:
            if ck in seen:
                cross += 1
            seen[ck] = v
    assert cross == 0, "有 cluster 跨折,逻辑错误!"
    print("[assert] OK:没有 cluster 跨折。")


if __name__ == "__main__":
    main()
