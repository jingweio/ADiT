#!/usr/bin/env python
"""算 OOD 分析的 overall + per-interface 指标。
读 ADiT 的 result pkl(每个 = (accession_codes, pred, target),与 scripts/skempi_metric.py 同格式;
pred/target 是 torch tensor,故需 torch 反序列化——在有 GPU 的本地跑,或 CPU 也行)。
complex = accession 前 3 个 '_' token。per-interface Spearman = 对每个 complex(组内样本≥MIN_N)算
spearman 再取均值。
用法: python reproduce/ood_perinterface_metric.py <label> [--min-n K] <pkl1> [pkl2 ...]
注:StaB-ddG 论文口径 = **K≥10**(只算突变数≥10 的 complex,取 per-complex Spearman 的 mean±SEM),
故默认 MIN_N=10 与之对齐。
"""
import sys, pickle
import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch  # noqa: 反序列化 cuda-saved tensor 需要


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x, dtype=float).reshape(-1)


def load(paths):
    accs, preds, targs = [], [], []
    for p in paths:
        with open(p, "rb") as f:
            a, pr, tg = pickle.load(f)
        accs += list(a)
        preds.append(to_np(pr)); targs.append(to_np(tg))
    return accs, np.concatenate(preds), np.concatenate(targs)


def main():
    label = sys.argv[1]
    rest = sys.argv[2:]
    MIN_N = 10  # StaB-ddG 论文口径
    if rest and rest[0] == "--min-n":
        MIN_N = int(rest[1]); rest = rest[2:]
    paths = rest
    accs, pred, target = load(paths)
    n = len(accs)
    # overall pooled
    pe = pearsonr(pred, target)[0]
    sp = spearmanr(pred, target)[0]
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mae = float(np.mean(np.abs(pred - target)))
    # per-interface
    groups = {}
    for a, p, t in zip(accs, pred, target):
        cid = "_".join(a.split("_")[:3])
        groups.setdefault(cid, [[], []])
        groups[cid][0].append(p); groups[cid][1].append(t)
    per_sp, used, skipped_const = [], 0, 0
    for cid, (ps, ts) in groups.items():
        if len(ps) < MIN_N:
            continue
        if len(set(ts)) < 2 or len(set(ps)) < 2:
            skipped_const += 1; continue
        s = spearmanr(ps, ts)[0]
        if np.isfinite(s):
            per_sp.append(s); used += 1
    mean_per_iface = float(np.mean(per_sp)) if per_sp else float("nan")
    median_per_iface = float(np.median(per_sp)) if per_sp else float("nan")
    sem = float(np.std(per_sp, ddof=1) / np.sqrt(len(per_sp))) if len(per_sp) > 1 else float("nan")
    print(f"### {label}  (min_n={MIN_N})")
    print(f"  n_samples={n} | n_complexes={len(groups)} | per-iface used (>= {MIN_N} muts)={used}")
    print(f"  OVERALL: pearson={pe:.4f} spearman={sp:.4f} rmse={rmse:.4f} mae={mae:.4f}")
    print(f"  PER-INTERFACE Spearman: mean={mean_per_iface:.4f} ± {sem:.4f} (SEM) | median={median_per_iface:.4f}")


if __name__ == "__main__":
    main()
