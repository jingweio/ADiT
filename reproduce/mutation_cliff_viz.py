#!/usr/bin/env python3
"""Mutation-cliff 深度可视化 + ADiT 失效系统验证。
数据:SKEMPIv2 / ADiT-S 三折池化预测 reproduce/skempi_per_sample_pred.csv。
SALI 口径:每个 complex 内 distinct mutant(同突变集合的重复测量先取均值)两两任意配对
  (distinct mutant >250 的大 complex 做 CAP=250 随机采样, seed=0),d=两 mutant 序列逐点比对
  的不同位点数, SALI=|ΔΔΔG|/d;再把所有 complex 的对 pool 成一个 set。
产出(均在 reproduce/mutation_analysis/):
  (2-1) ecdf_dddg_combined.png / ecdf_dddg_by_d.png       —— |ΔΔΔG| 的 ECDF
  (2-2) qmean_dddg_combined.png / qmean_dddg_by_d.png     —— 100 等量分位的均值曲线
  (2-3) pointcloud_dddg_by_d.png                          —— 各 d 的 |ΔΔΔG| 点云
  (3-1) adit_error_by_dddg_bin.png                        —— ADiT 预测随真值跳变增大而恶化
  (3-2) sali_pairs_table.csv                              —— 全 SALI 对的真值/预测跳变表
"""
import os, ast, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "mutation_analysis"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(D, "skempi_per_sample_pred.csv"))
df["mut_list"] = df["Mutation"].apply(ast.literal_eval)

def to_dict(lst):
    d = {}
    for m in lst:
        d[m[1:-1]] = m[-1]          # key=chain+pos, val=mut_aa
    return d
df["mdict"] = df["mut_list"].apply(to_dict)
df["mkey"]  = df["mdict"].apply(lambda d: frozenset(d.items()))
# distinct mutant:同 complex 同突变集合,true/pred ddG 取均值(抹掉重复测量噪声)
agg = df.groupby(["Name","mkey"], as_index=False).agg(
    ddG=("ddG","mean"), ddG_pred=("ddG_pred","mean"))
agg["mdict"] = agg["mkey"].apply(dict)
print(f"distinct mutant: {len(agg)} | complexes: {agg.Name.nunique()}")

def dist(a, b):
    return sum(1 for p in set(a) | set(b) if a.get(p) != b.get(p))
def diffpos(a, b):
    return ",".join(sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p)))

rng = np.random.default_rng(0); CAP = 250; rows = []
for name, g in agg.groupby("Name"):
    recs = g.to_dict("records")
    if len(recs) > CAP:
        recs = list(rng.choice(recs, CAP, replace=False))
    for a, b in itertools.combinations(recs, 2):
        d = dist(a["mdict"], b["mdict"])
        dT = a["ddG"] - b["ddG"]; dP = a["ddG_pred"] - b["ddG_pred"]
        rows.append((name, diffpos(a["mdict"], b["mdict"]), d,
                     a["ddG"], b["ddG"], a["ddG_pred"], b["ddG_pred"],
                     dT, dP, abs(dT), abs(dP), abs(dT)/d))
P = pd.DataFrame(rows, columns=["PDB","site","d","ddG_A","ddG_B","pred_A","pred_B",
                                "true_jump","pred_jump","absT","absP","SALI"])
P["pred_over_true"] = P["absP"] / P["absT"].replace(0, np.nan)
print(f"SALI 对总数: {len(P)} | d 分布: {P.d.value_counts().sort_index().to_dict()}")

DS = [1,2,3,4,5]
COL = plt.cm.viridis(np.linspace(0,0.9,len(DS)))
plt.rcParams.update({"figure.dpi":130, "font.size":10})

# ============================ (2-1) ECDF ============================
def ecdf(x):
    x = np.sort(np.asarray(x)); return x, np.arange(1,len(x)+1)/len(x)

fig,ax=plt.subplots(figsize=(5.2,3.6))
x,y=ecdf(P.absT); ax.plot(x,y,c="firebrick",lw=2,label=f"all pairs (n={len(P)})")
ax.axvline(2,ls=":",c="grey"); ax.set_xlim(0,8)
ax.set_xlabel("|ΔΔΔG|  (kcal/mol)"); ax.set_ylabel("ECDF  (fraction ≤ x)")
ax.set_title("(2-1a) |ΔΔΔG| ECDF — all d combined"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT,"ecdf_dddg_combined.png")); plt.close(fig)

fig,ax=plt.subplots(figsize=(5.2,3.6))
for d,c in zip(DS,COL):
    q=P[P.d==d].absT
    if len(q): x,y=ecdf(q); ax.plot(x,y,c=c,lw=1.6,label=f"d={d} (n={len(q)})")
ax.axvline(2,ls=":",c="grey"); ax.set_xlim(0,8)
ax.set_xlabel("|ΔΔΔG|  (kcal/mol)"); ax.set_ylabel("ECDF  (fraction ≤ x)")
ax.set_title("(2-1b) |ΔΔΔG| ECDF — by mutation distance d"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"ecdf_dddg_by_d.png")); plt.close(fig)

# ============================ (2-2) 100 等量分位的均值曲线 ============================
def qmean(x, nq=100):
    parts = np.array_split(np.sort(np.asarray(x)), nq)
    return np.arange(1,nq+1), np.array([p.mean() for p in parts])

fig,ax=plt.subplots(figsize=(5.2,3.6))
xi,ym=qmean(P.absT); ax.plot(xi,ym,c="firebrick",lw=2,label=f"all pairs (n={len(P)})")
ax.set_xlabel("percentile bin (1–100, |ΔΔΔG| ascending)"); ax.set_ylabel("mean |ΔΔΔG| in bin")
ax.set_title("(2-2a) sorted-mean |ΔΔΔG| — all d combined"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT,"qmean_dddg_combined.png")); plt.close(fig)

fig,ax=plt.subplots(figsize=(5.2,3.6))
for d,c in zip(DS,COL):
    q=P[P.d==d].absT
    if len(q)>=100: xi,ym=qmean(q); ax.plot(xi,ym,c=c,lw=1.6,label=f"d={d} (n={len(q)})")
ax.set_xlabel("percentile bin (1–100, |ΔΔΔG| ascending)"); ax.set_ylabel("mean |ΔΔΔG| in bin")
ax.set_title("(2-2b) sorted-mean |ΔΔΔG| — by distance d"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"qmean_dddg_by_d.png")); plt.close(fig)

# ============================ (2-3) 点云 ============================
fig,ax=plt.subplots(figsize=(5.6,3.8))
rngp=np.random.default_rng(1)
for d,c in zip(DS,COL):
    q=P[P.d==d].absT.values
    if len(q)==0: continue
    show=q if len(q)<=4000 else rngp.choice(q,4000,replace=False)
    jit=d+rngp.uniform(-0.32,0.32,len(show))
    ax.scatter(jit,show,s=4,alpha=0.12,c=[c])
    ax.plot([d-0.36,d+0.36],[np.median(q)]*2,c="k",lw=2)          # 中位
    ax.plot([d-0.36,d+0.36],[np.quantile(q,.9)]*2,c="k",lw=1,ls="--")  # p90
ax.axhline(2,ls=":",c="red",lw=1,label="|ΔΔΔG|=2 (cliff)")
ax.axhline(0.214,ls=":",c="green",lw=1,label="noise median 0.21")
ax.set_xticks(DS); ax.set_xlabel("mutation distance d (smaller = more similar)")
ax.set_ylabel("|ΔΔΔG|  (kcal/mol)"); ax.set_ylim(-0.3,10)
ax.set_title("(2-3) |ΔΔΔG| point cloud by d\n(black=median, dashed=p90)"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"pointcloud_dddg_by_d.png")); plt.close(fig)

# ============================ (3-1) ADiT 预测随真值跳变恶化 ============================
edges=[0,0.5,1,1.5,2,3,4,6,100]
lab=["0-.5",".5-1","1-1.5","1.5-2","2-3","3-4","4-6",">6"]
P["bin"]=pd.cut(P.absT,edges,labels=lab,right=False)
gb=P.groupby("bin",observed=True)
stat=pd.DataFrame({
    "n":gb.size(),
    "true_med":gb.absT.median(),
    "pred_med":gb.absP.median(),
    "capture":gb.apply(lambda g:(g.absP/g.absT.replace(0,np.nan)).median(),include_groups=False),
})
print("\n=== (3-1) ADiT 预测 by 真值跳变分桶 ===")
print(stat.round(3).to_string())

fig,(a1,a2)=plt.subplots(1,2,figsize=(9.4,3.7))
xs=np.arange(len(lab))
a1.bar(xs,stat.pred_med.values,color="steelblue",label="predicted |ΔΔΔG| (median)")
a1.plot(xs,stat.true_med.values,"r--o",lw=1.4,ms=4,label="true |ΔΔΔG| (median)")
a1.set_xticks(xs); a1.set_xticklabels(lab,rotation=45)
a1.set_xlabel("true |ΔΔΔG| bin (kcal/mol)"); a1.set_ylabel("|ΔΔΔG|")
a1.set_title("Predicted jump saturates as true cliff grows"); a1.legend(fontsize=8)
a2.plot(xs,stat.capture.values,"-o",c="darkorange",lw=1.8,ms=5)
a2.set_xticks(xs); a2.set_xticklabels(lab,rotation=45)
a2.set_xlabel("true |ΔΔΔG| bin (kcal/mol)"); a2.set_ylabel("capture ratio  median(|pred|/|true|)")
a2.axhline(1,ls=":",c="grey"); a2.set_ylim(0,1.05)
a2.set_title("ADiT captures less of the jump on bigger cliffs")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"adit_error_by_dddg_bin.png")); plt.close(fig)

# ============================ (3-2) 全 SALI 对表 ============================
tab=P[["PDB","site","d","ddG_A","ddG_B","true_jump","pred_jump","absT","absP","pred_over_true"]].copy()
tab=tab.round({"ddG_A":3,"ddG_B":3,"true_jump":3,"pred_jump":3,"absT":3,"absP":3,"pred_over_true":4})
tab=tab.sort_values(["PDB","d","pred_over_true"],ascending=[True,True,True])
tab.to_csv(os.path.join(OUT,"sali_pairs_table.csv"),index=False)
print(f"\n[写出表] mutation_analysis/sali_pairs_table.csv ({len(tab)} 行)")

# 极端 case:真值大跳变(>4)但模型严重低估(capture 最小)
ext=P[(P.absT>4)].sort_values("pred_over_true").head(10)
print("\n=== (3-2) Top-10 极端 cliff 低估 case (absT>4, 按 capture 升序) ===")
print(ext[["PDB","site","d","ddG_A","ddG_B","absT","absP","pred_over_true"]].round(3).to_string(index=False))

# 关键统计(写 md 用)
print("\n=== STATS (md 用) ===")
def block(x):
    x=np.asarray(x)
    return dict(n=len(x),median=np.median(x),CV=x.std()/x.mean(),
                p90=np.quantile(x,.9),p99=np.quantile(x,.99),mx=x.max(),
                smooth=(x<0.5).mean(),cliff=(x>2).mean())
print("ALL:",{k:(round(v,3) if k!='n' else v) for k,v in block(P.absT).items()})
for d in DS:
    q=P[P.d==d].absT
    if len(q): print(f"d={d}:",{k:(round(v,3) if k!='n' else v) for k,v in block(q).items()})
print(f"corr(d,|ΔΔΔG|): Pearson={pearsonr(P.d,P.absT)[0]:.3f} Spearman={spearmanr(P.d,P.absT)[0]:.3f}")
print(f"SALI>2 占比={100*(P.SALI>2).mean():.1f}% | SALI>3 占比={100*(P.SALI>3).mean():.1f}% | SALI max={P.SALI.max():.1f}")
