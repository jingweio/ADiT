#!/usr/bin/env python3
"""per-interface(per-structure)口径,按 #mutation-site 分层统计模型指标。
做法:在每个 interface(Name=PDB_伙伴1_伙伴2)内、限定到某个突变位点数分层的样本,
算 within-interface 的 Pearson/Spearman/RMSE/MAE(要求该 interface 该分层样本数 >=K 且 ddG 有方差),
再对所有合格 interface 取(未加权)平均 —— 这是领域标准的 per-structure 指标。"""
import pickle, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

D = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(D, "skempi_per_sample_pred.csv"))

def per_interface(sub, K):
    """对 sub 按 Name 分组,合格 interface(>=K 且 std>0)各算指标,返回跨 interface 的均值。"""
    prs, srs, rmses, maes = [], [], [], []
    n_if = 0; n_cov = 0
    for name, g in sub.groupby("Name"):
        if len(g) < K or g["ddG"].std() == 0 or g["ddG_pred"].std() == 0:
            continue
        p, t = g["ddG_pred"].values, g["ddG"].values
        prs.append(pearsonr(p, t)[0]); srs.append(spearmanr(p, t)[0])
        rmses.append(np.sqrt(np.mean((p - t) ** 2))); maes.append(np.mean(np.abs(p - t)))
        n_if += 1; n_cov += len(g)
    if n_if == 0:
        return None
    return dict(n_interface=n_if, n_samples=n_cov,
               Pearson=np.mean(prs), Spearman=np.mean(srs),
               RMSE=np.mean(rmses), MAE=np.mean(maes))

maxn = int(df["num_mutation_sites"].max())
for K in [10, 5]:
    print("=" * 78)
    print(f"per-interface 指标(阈值:每 interface 该分层 >= {K} 个突变;跨 interface 取均值)")
    print("=" * 78)
    rows = []
    strata = [("n="+str(n), df[df.num_mutation_sites == n]) for n in range(1, maxn + 1)]
    strata += [("single(n=1)", df[df.num_mutation_sites == 1]),
               ("multi(n>1)", df[df.num_mutation_sites > 1]),
               ("overall(all)", df)]
    for label, sub in strata:
        if len(sub) == 0:
            continue
        r = per_interface(sub, K)
        if r is None:
            rows.append({"stratum": label, "n_interface": 0, "n_samples": 0,
                         "Pearson": np.nan, "Spearman": np.nan, "RMSE": np.nan, "MAE": np.nan})
        else:
            rows.append({"stratum": label, **{k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in r.items()}})
    tab = pd.DataFrame(rows)[["stratum","n_interface","n_samples","Pearson","Spearman","RMSE","MAE"]]
    print(tab.to_string(index=False))
    if K == 10:
        tab.to_csv(os.path.join(D, "skempi_per_interface_by_nsites_K10.csv"), index=False)
        print(f"\n[写出] {os.path.join(D,'skempi_per_interface_by_nsites_K10.csv')}")
    print()
