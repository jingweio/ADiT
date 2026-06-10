#!/usr/bin/env python3
"""Mutation-cliff 扩展分析:multi-site 近邻对 + SALI 指数 + 出图。
mutant 用 (Name, {pos:mut_aa}) 表示;同界面任意两 mutant 的"突变距离 d"=两条 mutant 序列
不同残基位置数(缺失位点视为 WT)。d=1 即"差一个突变步"(同位点换AA,或加/减一个突变)。
SALI = |ΔΔΔG| / d(每突变步的 ddG 变化;越大=landscape 越陡=越像 cliff)。
"""
import os, ast, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

D = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(D, "skempi_per_sample_pred.csv"))
df["mut_list"] = df["Mutation"].apply(ast.literal_eval)

def muts_to_dict(lst):
    d = {}
    for m in lst:
        d[m[1:-1]] = m[-1]      # key=chain+pos, val=mut_aa
    return d

df["mdict"] = df["mut_list"].apply(muts_to_dict)
df["mkey"] = df["mdict"].apply(lambda d: frozenset(d.items()))

# 去重到 distinct mutant(同界面同突变集合取均值)
agg = df.groupby(["Name", "mkey"], as_index=False).agg(
    ddG=("ddG", "mean"), ddG_pred=("ddG_pred", "mean"), nsite=("num_mutation_sites", "first"))
agg["mdict"] = agg["mkey"].apply(dict)
print(f"distinct mutant 数: {len(agg)} | 界面数: {agg.Name.nunique()}")

def dist(a, b):
    d = 0
    for p in set(a) | set(b):
        if a.get(p) != b.get(p):
            d += 1
    return d

# 同界面任意配对(界面过大时采样以控规模)
rng = np.random.default_rng(0)
CAP = 250
rows = []
for name, g in agg.groupby("Name"):
    recs = g.to_dict("records")
    if len(recs) > CAP:
        recs = list(rng.choice(recs, CAP, replace=False))
    for a, b in itertools.combinations(recs, 2):
        d = dist(a["mdict"], b["mdict"])
        dT = a["ddG"] - b["ddG"]; dP = a["ddG_pred"] - b["ddG_pred"]
        rows.append((name, d, a["nsite"], b["nsite"], dT, dP, abs(dT), abs(dP), abs(dT)/d))
P = pd.DataFrame(rows, columns=["Name","d","nsiteA","nsiteB","dT","dP","absT","absP","SALI"])
print(f"同界面突变对总数: {len(P)}")

# ---- d 分层:近似程度 vs ΔΔΔG 跳变 / 模型捕捉 ----
print("\n" + "="*78)
print("按突变距离 d 分层(d=1 即差一个突变步,最相似)")
print("="*78)
print(f"{'d':>3} {'#pairs':>8} {'|ΔΔΔG真|中位':>12} {'p90':>7} {'|ΔΔΔG预|中位':>12} {'预测/真值比':>10} {'真>2占比%':>9}")
for d in [1,2,3,4,5]:
    q = P[P.d==d]
    if len(q)==0: continue
    ratio = (q.absP/q.absT.replace(0,np.nan)).median()
    print(f"{d:>3} {len(q):>8} {q.absT.median():>12.3f} {q.absT.quantile(.9):>7.2f} "
          f"{q.absP.median():>12.3f} {ratio:>10.2f} {100*(q.absT>2).mean():>9.1f}")
q = P[P.d>=6]
if len(q): print(f">=6 {len(q):>7} {q.absT.median():>12.3f} {q.absT.quantile(.9):>7.2f} {q.absP.median():>12.3f}")

# ---- d=1 再细分:同位点换AA(both single, same pos) vs 加减一个突变(含 multi) ----
print("\n" + "="*78); print("d=1 细分:single 同位点换AA  vs  multi 近邻(加/减一个突变)"); print("="*78)
d1 = P[P.d==1]
single_swap = d1[(d1.nsiteA==1)&(d1.nsiteB==1)]
multi_nb = d1[(d1.nsiteA>1)|(d1.nsiteB>1)]
for lab, q in [("single 同位点换AA", single_swap), ("multi 近邻(≥1个多突变)", multi_nb)]:
    if len(q):
        print(f"{lab}: {len(q)}对 | |ΔΔΔG真|中位={q.absT.median():.3f} p90={q.absT.quantile(.9):.2f} "
              f"| 真>2占比={100*(q.absT>2).mean():.1f}% | 预测/真值比中位={(q.absP/q.absT.replace(0,np.nan)).median():.2f}")

# ---- SALI cliff ----
print("\n" + "="*78); print("SALI = |ΔΔΔG|/d  cliff 指数"); print("="*78)
print(f"SALI: median={P.SALI.median():.3f} p90={P.SALI.quantile(.9):.3f} p99={P.SALI.quantile(.99):.3f} max={P.SALI.max():.3f}")
hi = P[P.SALI>2]       # 高 SALI = cliff(每步>2 kcal/mol)
print(f"高SALI(>2)对数: {len(hi)} ({100*len(hi)/len(P):.1f}%) | 其中模型预测跳变中位={hi.absP.median():.3f} "
      f"| 预测/真值比中位={(hi.absP/hi.absT).median():.2f}")

# ===================== 出图 =====================
plt.rcParams.update({"figure.dpi":130, "font.size":10})

# 图1:噪声下界 vs 同位点ΔΔG跨度(ECDF)
s = df[df.num_mutation_sites==1].copy()
s["m"]=s["mut_list"].apply(lambda l:l[0])
s["site"]=s["m"].apply(lambda x:x[1:-1])
# 噪声:相同突变(同 Name 同 mut)重复测量的 ddG range,仅取 ≥2 次的组
rg = s.groupby(["Name","m"]).ddG.agg(["size", lambda x: x.max()-x.min()])
rg.columns=["n","rng"]; noise = rg.loc[rg.n>=2,"rng"]
# cliff:去重到 distinct 单点突变后,同位点(同 Name 同 site)≥2 个不同AA 的 ddG range
dstd = s.groupby(["Name","m"],as_index=False).agg(ddG=("ddG","mean"))
dstd["site"]=dstd.m.apply(lambda x:x[1:-1])
sg = dstd.groupby(["Name","site"]).ddG.agg(["size", lambda x: x.max()-x.min()])
sg.columns=["k","rng"]; sgrp = sg.loc[sg.k>=2,"rng"]
fig,ax=plt.subplots(figsize=(5,3.5))
for data,lab in [(noise.values,"same mutation (noise floor)"),(sgrp.values,"same site, diff AA (cliff)")]:
    x=np.sort(data); y=np.arange(1,len(x)+1)/len(x); ax.plot(x,y,label=f"{lab} (n={len(x)})")
ax.set_xlabel("ΔΔG range within group (kcal/mol)"); ax.set_ylabel("ECDF"); ax.set_xlim(0,6)
ax.axvline(2,ls=":",c="grey"); ax.legend(); ax.set_title("Cliffs far exceed label-noise floor")
fig.tight_layout(); fig.savefig(os.path.join(D,"fig_cliff_vs_noise.png")); plt.close(fig)

# 图2:|ΔΔΔG真| vs |ΔΔΔG预测|(d=1 同界面近邻),y=x
fig,ax=plt.subplots(figsize=(4.5,4.2))
ax.scatter(d1.absT,d1.absP,s=5,alpha=.25)
mx=max(d1.absT.max(),d1.absP.max())
ax.plot([0,mx],[0,mx],"r--",lw=1,label="y=x (perfect)")
ax.set_xlabel("|ΔΔΔG| true (d=1 pairs)"); ax.set_ylabel("|ΔΔΔG| predicted")
ax.set_title("Model under-predicts cliff jumps\n(points fall below y=x)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(D,"fig_jump_true_vs_pred.png")); plt.close(fig)

# 图3:预测跳变 按真值跳变分箱(中位+IQR)
fig,ax=plt.subplots(figsize=(5,3.5))
bins=[0,0.5,1,1.5,2,3,4,12]; lab=["0-.5",".5-1","1-1.5","1.5-2","2-3","3-4",">4"]
d1b=d1.copy(); d1b["bin"]=pd.cut(d1b.absT,bins,labels=lab)
med=d1b.groupby("bin").absP.median(); cnt=d1b.groupby("bin").absP.size()
ax.bar(range(len(med)),med.values,color="steelblue")
ax.plot(range(len(med)),[ (bins[i]+bins[i+1])/2 for i in range(len(lab))],"r--",marker="o",label="true jump (bin center)")
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab,rotation=45)
ax.set_xlabel("|ΔΔΔG| true bin"); ax.set_ylabel("predicted |ΔΔΔG| (median)")
ax.set_title("Predicted jump saturates as true jump grows"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(D,"fig_jump_saturation.png")); plt.close(fig)

# 图4:|ΔΔΔG真| 随 d 的箱线(近似度越高 cliff 越突出)
fig,ax=plt.subplots(figsize=(5,3.5))
data=[P[P.d==k].absT.values for k in [1,2,3,4,5]]
ax.boxplot(data,labels=["1","2","3","4","5"],showfliers=False)
ax.set_xlabel("mutation distance d (smaller = more similar)"); ax.set_ylabel("|ΔΔΔG| true")
ax.set_title("Large ΔΔG jumps persist even at d=1"); fig.tight_layout()
fig.savefig(os.path.join(D,"fig_absT_by_distance.png")); plt.close(fig)

P.sort_values("SALI",ascending=False).head(2000).to_csv(os.path.join(D,"skempi_cliff_pairs_SALI_top2000.csv"),index=False)
print("\n[图] reproduce/fig_cliff_vs_noise.png, fig_jump_true_vs_pred.png, fig_jump_saturation.png, fig_absT_by_distance.png")
print("[csv] reproduce/skempi_cliff_pairs_SALI_top2000.csv")
