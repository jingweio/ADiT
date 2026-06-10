#!/usr/bin/env python3
"""检验"multi-site 排序更好"是真实建模能力还是 ddG 动态范围/分布的混杂效应。"""
import numpy as np, pandas as pd, os
from scipy.stats import spearmanr, pearsonr

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "skempi_per_sample_pred.csv"))
df["cls"] = np.where(df["num_mutation_sites"] == 1, "single", "multi")

print("="*64)
print("A. 真实 ddG 的分布(range 混杂检验)")
print("="*64)
for c in ["single", "multi"]:
    s = df.loc[df.cls == c, "ddG"]
    print(f"{c:6s} n={len(s):5d} | std={s.std():.3f} | |ddG|mean={s.abs().mean():.3f} | "
          f"min={s.min():.2f} max={s.max():.2f} | IQR={s.quantile(.75)-s.quantile(.25):.3f}")

print("\n" + "="*64)
print("B. 全局 Spearman 的 bootstrap 95% CI(是否显著)")
print("="*64)
rng = np.random.default_rng(0)
def boot_sp(sub, n=1000):
    p, t = sub["ddG_pred"].values, sub["ddG"].values
    out = []
    idx = np.arange(len(sub))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(spearmanr(p[b], t[b])[0])
    return np.percentile(out, [2.5, 50, 97.5])
for c in ["single", "multi"]:
    lo, md, hi = boot_sp(df[df.cls == c])
    print(f"{c:6s} Spearman 中位={md:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")

print("\n" + "="*64)
print("C. range 匹配后再比 Spearman(把 single/multi 限制到相同 ddG 范围)")
print("="*64)
# 取两组共同的 ddG 区间(single 的 1%~99% 分位),都裁剪到该窗内再比
lo_q, hi_q = df.loc[df.cls=="single","ddG"].quantile([.01,.99])
sub = df[(df.ddG>=lo_q)&(df.ddG<=hi_q)]
for c in ["single","multi"]:
    s = sub[sub.cls==c]
    if len(s)>2:
        print(f"{c:6s} n={len(s):5d} | Spearman={spearmanr(s.ddG_pred,s.ddG)[0]:.3f} | "
              f"Pearson={pearsonr(s.ddG_pred,s.ddG)[0]:.3f} | ddG std={s.ddG.std():.3f}")

print("\n" + "="*64)
print("D. 复合物内(within-complex)排序:更接近'生物上有意义的排序'")
print("   对每个 #Pdb 分别在其 single 子集 / multi 子集上算 Spearman(样本≥K),再看分布")
print("="*64)
for K in [8]:
    sp_single, sp_multi = [], []
    for name, g in df.groupby("Name"):
        gs = g[g.cls=="single"]; gm = g[g.cls=="multi"]
        if len(gs)>=K and gs.ddG.std()>0: sp_single.append(spearmanr(gs.ddG_pred,gs.ddG)[0])
        if len(gm)>=K and gm.ddG.std()>0: sp_multi.append(spearmanr(gm.ddG_pred,gm.ddG)[0])
    print(f"K={K}: 满足的复合物数 single={len(sp_single)}, multi={len(sp_multi)}")
    if sp_single: print(f"  within-complex single Spearman: median={np.median(sp_single):.3f} mean={np.mean(sp_single):.3f}")
    if sp_multi:  print(f"  within-complex multi  Spearman: median={np.median(sp_multi):.3f} mean={np.mean(sp_multi):.3f}")
