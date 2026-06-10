#!/usr/bin/env python3
"""Mutation-cliff 数据分析(SKEMPIv2 / ADiT 三折池化预测)。
核心问题:
  (Q1) 同一界面、近似突变(尤其同一位点的不同替换)是否出现 ΔΔG 巨变?是否超过标签噪声下界?
  (Q2) ADiT 对这种 cliff 的预测能力如何(能否捕捉跳变,还是把 landscape 抹平)?
数据:reproduce/skempi_per_sample_pred.csv(列:Name=界面, Mutation, ddG=真值, ddG_pred, num_mutation_sites)
"""
import os, ast, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

D = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(D, "skempi_per_sample_pred.csv"))
df["mut_list"] = df["Mutation"].apply(ast.literal_eval)

# 只看单点突变(最干净的"近似突变"定义)
s = df[df.num_mutation_sites == 1].copy()
s["mut"] = s["mut_list"].apply(lambda l: l[0])
def parse(m):
    return pd.Series({"wt_aa": m[0], "mut_aa": m[-1], "chain": m[1], "pos": m[2:-1], "site": m[1:-1]})
s = pd.concat([s, s["mut"].apply(parse)], axis=1)
print(f"单点突变样本数: {len(s)} | 涉及界面数: {s.Name.nunique()}")

# ===========================================================================
# (A) 标签噪声下界:完全相同的突变(同 Name 同 mut)的重复测量,ddG 的离散程度
# ===========================================================================
print("\n" + "="*70 + "\n(A) 标签噪声下界:完全相同突变的重复测量 ddG 离散度\n" + "="*70)
rep = s.groupby(["Name", "mut"]).agg(n=("ddG", "size"),
        rng=("ddG", lambda x: x.max()-x.min()), sd=("ddG", "std"))
repm = rep[rep.n >= 2]
print(f"有重复测量(≥2 次)的'相同突变'组数: {len(repm)}(覆盖 {int(repm.n.sum())} 条记录)")
if len(repm):
    print(f"  同一突变 ddG range:  median={repm.rng.median():.3f}  mean={repm.rng.mean():.3f}  p90={repm.rng.quantile(.9):.3f}")
    print(f"  → 这≈测量噪声下界(同一突变本应同值);cliff 信号需明显超过它")

# 去重到"distinct 突变"(同 Name 同 mut 取均值),用于 cliff 分析
dst = s.groupby(["Name", "mut"], as_index=False).agg(
    ddG=("ddG", "mean"), ddG_pred=("ddG_pred", "mean"),
    site=("site", "first"), chain=("chain", "first"), pos=("pos", "first"),
    wt_aa=("wt_aa", "first"), mut_aa=("mut_aa", "first"))
print(f"\n去重后 distinct 单点突变数: {len(dst)} | 全局 ddG std = {dst.ddG.std():.3f}")

# ===========================================================================
# (B) 同一位点(同界面+同 chain+pos)不同替换氨基酸 → ddG 的离散度 = cliff 强度
# ===========================================================================
print("\n" + "="*70 + "\n(B) 同界面+同位点、不同替换AA 的 ΔΔG 离散度(cliff 存在性)\n" + "="*70)
site = dst.groupby(["Name", "site"]).agg(
    k=("mut_aa", "nunique"),
    ddg_rng=("ddG", lambda x: x.max()-x.min()), ddg_sd=("ddG", "std"),
    pred_rng=("ddG_pred", lambda x: x.max()-x.min()), pred_sd=("ddG_pred", "std"))
site = site[site.k >= 2]
print(f"同位点深扫(≥2 个不同替换AA)的(界面,位点)组数: {len(site)}")
print(f"  同位点内 ΔΔG range: median={site.ddg_rng.median():.3f} mean={site.ddg_rng.mean():.3f} "
      f"max={site.ddg_rng.max():.3f}")
for thr in [1, 2, 3]:
    f = (site.ddg_rng > thr).mean()
    print(f"  同位点 ΔΔG range > {thr} kcal/mol 的组占比: {100*f:.1f}%")
print(f"  对比:模型预测在同位点内的 range: median={site.pred_rng.median():.3f} mean={site.pred_rng.mean():.3f}")
print(f"  → 若 真值range >> 预测range,说明模型把同位点不同替换'抹平'了(对 cliff 不敏感)")

# ===========================================================================
# (C) 成对分析:同界面+同位点的突变对 → |ΔΔG真| vs |ΔΔG预测|(模型能否捕捉跳变)
# ===========================================================================
print("\n" + "="*70 + "\n(C) 成对 cliff:同界面+同位点突变对,真值跳变 vs 预测跳变\n" + "="*70)
pairs = []
for (name, st), g in dst.groupby(["Name", "site"]):
    if len(g) < 2:
        continue
    rec = g.to_dict("records")
    for a, b in itertools.combinations(rec, 2):
        dT = a["ddG"] - b["ddG"]; dP = a["ddG_pred"] - b["ddG_pred"]
        pairs.append({"Name": name, "site": st,
                      "mutA": a["mut_aa"], "mutB": b["mut_aa"],
                      "ddgA": a["ddG"], "ddgB": b["ddG"],
                      "predA": a["ddG_pred"], "predB": b["ddG_pred"],
                      "dT": dT, "dP": dP, "absT": abs(dT), "absP": abs(dP)})
P = pd.DataFrame(pairs)
print(f"同位点突变对数: {len(P)}")
print(f"  真值跳变 |ΔΔΔG|: median={P.absT.median():.3f} mean={P.absT.mean():.3f} p90={P.absT.quantile(.9):.3f} max={P.absT.max():.3f}")
# 模型能否捕捉跳变方向/幅度
if len(P) >= 3 and P.dT.std() > 0:
    print(f"  ΔΔΔG真 vs ΔΔΔG预测 相关: Pearson={pearsonr(P.dT,P.dP)[0]:.3f} Spearman={spearmanr(P.dT,P.dP)[0]:.3f}")
for thr in [1, 2, 3]:
    cl = P[P.absT > thr]
    if len(cl):
        ratio = (cl.absP / cl.absT).median()
        capt = (cl.absP > thr).mean()  # 模型预测跳变也>thr 的比例
        print(f"  cliff对(|ΔΔΔG真|>{thr}): {len(cl)}对 | 预测跳变|ΔΔΔG预|中位={cl.absP.median():.3f} "
              f"| 预测/真值幅度比中位={ratio:.2f} | 模型也判'大跳变(>{thr})'比例={100*capt:.1f}%")
print("  → 若 幅度比<<1 且 大跳变捕捉比例低,说明模型在 cliff 上系统性低估跳变(抹平)")

# ===========================================================================
# (D) 同位点组内 ranking 能力:k>=3 的组里,pred 与真值的 Spearman
# ===========================================================================
print("\n" + "="*70 + "\n(D) 同位点组内(k≥3)模型对不同替换的排序能力\n" + "="*70)
sp = []
for (name, st), g in dst.groupby(["Name", "site"]):
    if g["mut_aa"].nunique() >= 3 and g.ddG.std() > 0 and g.ddG_pred.std() > 0:
        sp.append(spearmanr(g.ddG_pred, g.ddG)[0])
sp = np.array([x for x in sp if not np.isnan(x)])
print(f"k≥3 的同位点组数: {len(sp)} | 组内 Spearman: median={np.median(sp):.3f} mean={np.mean(sp):.3f} "
      f"| <0 的组占比={100*(sp<0).mean():.1f}%")
print("  → 中位接近 0 / 大量为负 = 模型几乎无法对同一位点的不同替换排序(cliff 上排序失效)")

# ===========================================================================
# (E) 最极端的 cliff 实例(同位点、真值大跳变、模型预测跳变小 → 误差最大)
# ===========================================================================
print("\n" + "="*70 + "\n(E) Top-12 最极端 cliff 实例(同位点大ΔΔG跳变 且 模型低估最严重)\n" + "="*70)
P["miss"] = (P.absT - P.absP)               # 真值跳变 - 预测跳变,越大=模型越没捕捉到
top = P[P.absT > 2].sort_values("miss", ascending=False).head(12)
cols = ["Name", "site", "mutA", "mutB", "ddgA", "ddgB", "dT", "dP"]
print(top[cols].round(2).to_string(index=False))

# 保存 cliff 对
P.sort_values("absT", ascending=False).to_csv(os.path.join(D, "skempi_mutation_cliff_pairs.csv"), index=False)
site.reset_index().to_csv(os.path.join(D, "skempi_same_site_groups.csv"))
print(f"\n[写出] reproduce/skempi_mutation_cliff_pairs.csv ({len(P)} 对) ; reproduce/skempi_same_site_groups.csv ({len(site)} 组)")
