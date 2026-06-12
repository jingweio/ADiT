#!/usr/bin/env python3
"""Mutation-cliff 深度可视化 + ADiT 失效系统验证。
数据:SKEMPIv2 / ADiT-S 三折池化预测 reproduce/skempi_per_sample_pred.csv。
口径:每个 complex 内 distinct mutant(同突变集合的重复测量先取均值)两两任意配对
  (distinct mutant >250 的大 complex 做 CAP=250 随机采样, seed=0),d=两 mutant 序列逐点比对
  的不同位点数, |ΔΔΔG|=真值 ddG 差, SALI=|ΔΔΔG|/d(cliff 指数,每突变步的 ddG 变化);
  再把所有 complex 的对 pool 成一个 set。
产出(均在 reproduce/mutation_analysis/):
  §3 cliff 存在性(指标=SALI):
    Evidence1  ecdf_sali_combined.png / ecdf_sali_by_d.png
    Evidence2  qmean_sali_combined.png / qmean_sali_by_d.png
    Evidence3  pointcloud_sali_by_d.png
  §4 ADiT 失效(指标=ddG / |ΔΔΔG|):
    adit_jump_saturation.png   —— 真值 vs 预测 |ΔΔΔG|(mean)随真值分桶
    adit_corr_by_dddg_bin.png  —— 每个真值跳变桶里 真值vs预测 的 Pearson/Spearman
    sali_pairs_table.csv        —— 全突变对表
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
        d[m[1:-1]] = m[-1]
    return d
df["mdict"] = df["mut_list"].apply(to_dict)
df["mkey"]  = df["mdict"].apply(lambda d: frozenset(d.items()))
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
print(f"突变对总数: {len(P)} | d 分布(前5): {P[P.d<=5].d.value_counts().sort_index().to_dict()}")

DS = [1,2,3,4,5]
COL = plt.cm.viridis(np.linspace(0,0.9,len(DS)))
plt.rcParams.update({"figure.dpi":130, "font.size":10})

def ecdf(x):
    x = np.sort(np.asarray(x)); return x, np.arange(1,len(x)+1)/len(x)
def qmean(x, nq=100):
    parts = np.array_split(np.sort(np.asarray(x)), nq)
    return np.arange(1,nq+1), np.array([p.mean() for p in parts])

# ================= §3 Evidence 1: SALI 的 ECDF =================
fig,ax=plt.subplots(figsize=(5.2,3.6))
x,y=ecdf(P.SALI); ax.plot(x,y,c="firebrick",lw=2,label=f"all pairs (n={len(P)})")
ax.axvline(2,ls=":",c="grey"); ax.set_xlim(0,6)
ax.set_xlabel("SALI = |ΔΔΔG| / d  (kcal/mol per step)"); ax.set_ylabel("ECDF (fraction ≤ x)")
ax.set_title("Evidence 1: SALI ECDF — all d combined"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT,"ecdf_sali_combined.png")); plt.close(fig)

fig,ax=plt.subplots(figsize=(5.2,3.6))
for d,c in zip(DS,COL):
    q=P[P.d==d].SALI
    if len(q): x,y=ecdf(q); ax.plot(x,y,c=c,lw=1.6,label=f"d={d} (n={len(q)})")
ax.axvline(2,ls=":",c="grey"); ax.set_xlim(0,6)
ax.set_xlabel("SALI = |ΔΔΔG| / d  (kcal/mol per step)"); ax.set_ylabel("ECDF (fraction ≤ x)")
ax.set_title("Evidence 1: SALI ECDF — by mutation distance d"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"ecdf_sali_by_d.png")); plt.close(fig)

# ================= §3 Evidence 2: SALI 的 100 等量分位均值曲线 =================
fig,ax=plt.subplots(figsize=(5.2,3.6))
xi,ym=qmean(P.SALI); ax.plot(xi,ym,c="firebrick",lw=2,label=f"all pairs (n={len(P)})")
ax.set_xlabel("percentile bin (1–100, SALI ascending)"); ax.set_ylabel("mean SALI in bin")
ax.set_title("Evidence 2: sorted-mean SALI — all d combined"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT,"qmean_sali_combined.png")); plt.close(fig)

fig,ax=plt.subplots(figsize=(5.2,3.6))
for d,c in zip(DS,COL):
    q=P[P.d==d].SALI
    if len(q)>=100: xi,ym=qmean(q); ax.plot(xi,ym,c=c,lw=1.6,label=f"d={d} (n={len(q)})")
ax.set_xlabel("percentile bin (1–100, SALI ascending)"); ax.set_ylabel("mean SALI in bin")
ax.set_title("Evidence 2: sorted-mean SALI — by distance d"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"qmean_sali_by_d.png")); plt.close(fig)

# ================= §3 Evidence 3: SALI 点云 by d =================
fig,ax=plt.subplots(figsize=(5.6,3.8))
rngp=np.random.default_rng(1)
for d,c in zip(DS,COL):
    q=P[P.d==d].SALI.values
    if len(q)==0: continue
    show=q if len(q)<=4000 else rngp.choice(q,4000,replace=False)
    jit=d+rngp.uniform(-0.32,0.32,len(show))
    ax.scatter(jit,show,s=4,alpha=0.12,c=[c])
    ax.plot([d-0.36,d+0.36],[np.median(q)]*2,c="k",lw=2)
    ax.plot([d-0.36,d+0.36],[np.quantile(q,.9)]*2,c="k",lw=1,ls="--")
ax.axhline(2,ls=":",c="red",lw=1,label="SALI=2 (steep cliff)")
ax.set_xticks(DS); ax.set_xlabel("mutation distance d (smaller = more similar)")
ax.set_ylabel("SALI = |ΔΔΔG| / d"); ax.set_ylim(-0.2,7)
ax.set_title("Evidence 3: SALI point cloud by d\n(black=median, dashed=p90)"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"pointcloud_sali_by_d.png")); plt.close(fig)

# SALI 统计(§3 Evidence3 表 + 文字)
print("\n=== §3 SALI 统计(by d + all)===")
def block(x):
    x=np.asarray(x)
    return dict(n=len(x),median=np.median(x),CV=x.std()/x.mean(),
                p90=np.quantile(x,.9),p99=np.quantile(x,.99),mx=x.max(),
                smooth=(x<0.5).mean(),cliff=(x>2).mean())
for d in DS:
    q=P[P.d==d].SALI
    print(f"d={d}:",{k:(round(v,3) if k!='n' else int(v)) for k,v in block(q).items()})
print("ALL:",{k:(round(v,3) if k!='n' else int(v)) for k,v in block(P.SALI).items()})
print(f"corr(d, SALI): Pearson={pearsonr(P.d,P.SALI)[0]:.3f} Spearman={spearmanr(P.d,P.SALI)[0]:.3f}")
print(f"SALI>2 占比={100*(P.SALI>2).mean():.1f}% | >3={100*(P.SALI>3).mean():.1f}% | max={P.SALI.max():.1f}")

# ================= §4 fig1:|ΔΔΔG| 分桶,mean 真值 vs 预测(幅度饱和)=================
edges=[0,0.5,1,1.5,2,3,4,6,100]
lab=["0-.5",".5-1","1-1.5","1.5-2","2-3","3-4","4-6",">6"]
P["bin"]=pd.cut(P.absT,edges,labels=lab,right=False)
g1=P.groupby("bin",observed=True)
s1=pd.DataFrame({"n":g1.size(),"true_mean":g1.absT.mean(),"pred_mean":g1.absP.mean()})
s1["gap"]=s1.true_mean-s1.pred_mean
print("\n=== §4 fig1: |ΔΔΔG| 分桶(mean 真值 vs 预测)===")
print(s1.round(3).to_string())
xs=np.arange(len(lab))
fig,ax=plt.subplots(figsize=(5.4,3.7))
ax.bar(xs,s1.pred_mean.values,color="steelblue",label="predicted |ΔΔΔG| (mean)")
ax.plot(xs,s1.true_mean.values,"r--o",lw=1.4,ms=4,label="true |ΔΔΔG| (mean)")
ax.set_xticks(xs); ax.set_xticklabels(lab,rotation=45)
ax.set_xlabel("true |ΔΔΔG| bin (kcal/mol)"); ax.set_ylabel("|ΔΔΔG| (mean)")
ax.set_title("Predicted jump saturates as true cliff grows\n(gap = true−pred widens)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"adit_jump_saturation.png")); plt.close(fig)

# ================= §4 fig2:按 SALI(cliff 陡峭度)分桶,模型退化 =================
# 预测的 cliff 陡峭度 SALI_pred = |预测跳变| / d;capture = mean(SALI_pred)/mean(SALI_true)
P["SALI_pred"]=P.absP/P.d
sedges=[0,0.25,0.5,0.75,1,1.5,2,100]
slab=["0-.25",".25-.5",".5-.75",".75-1","1-1.5","1.5-2",">2"]
P["sbin"]=pd.cut(P.SALI,sedges,labels=slab,right=False)
def sstat(g):
    pe=pearsonr(g.true_jump,g.pred_jump)[0] if g.true_jump.std()>0 and g.pred_jump.std()>0 else np.nan
    sp=spearmanr(g.true_jump,g.pred_jump)[0]
    return pd.Series({"n":len(g),"SALI_true":g.SALI.mean(),"SALI_pred":g.SALI_pred.mean(),
                      "capture":g.SALI_pred.mean()/g.SALI.mean(),"pearson":pe,"spearman":sp})
s2=P.groupby("sbin",observed=True).apply(sstat,include_groups=False)
print("\n=== §4 fig2: SALI 分桶(capture 下降 / Pearson-Spearman 上升=SNR 假象)===")
print(s2.round(3).to_string())
xs2=np.arange(len(slab))
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,3.8))
a1.bar(xs2,s2.SALI_pred.values,color="steelblue",label="predicted SALI (mean)")
a1.plot(xs2,s2.SALI_true.values,"r--o",lw=1.4,ms=4,label="true SALI (mean)")
a1.set_xticks(xs2); a1.set_xticklabels(slab,rotation=45)
a1.set_xlabel("true SALI bin (kcal/mol per step)"); a1.set_ylabel("SALI (mean)")
a1.set_title("Predicted cliff steepness saturates"); a1.legend(fontsize=8)
a2.plot(xs2,s2.capture.values,"-o",c="darkorange",lw=1.9,ms=5,label="capture = pred/true SALI")
a2.axhline(1,ls=":",c="grey")
a2.set_xticks(xs2); a2.set_xticklabels(slab,rotation=45)
a2.set_xlabel("true SALI bin (kcal/mol per step)"); a2.set_ylabel("magnitude capture ratio")
a2.set_ylim(0,1.6); a2.set_title("ADiT captures less steepness on steeper cliffs"); a2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"adit_sali_degradation.png")); plt.close(fig)

# ================= 表:删 absT/absP/pred_over_true,加 diff_percent =================
P["diff_percent"]=(P.pred_jump-P.true_jump)/P.true_jump.replace(0,np.nan)
tab=P[["PDB","site","d","ddG_A","ddG_B","true_jump","pred_jump","diff_percent"]].copy()
tab=tab.round({"ddG_A":3,"ddG_B":3,"true_jump":3,"pred_jump":3,"diff_percent":4})
tab["_absdp"]=tab.diff_percent.abs()
tab=tab.sort_values(["PDB","d","_absdp"],ascending=[True,True,False],na_position="last").drop(columns="_absdp")
tab.to_csv(os.path.join(OUT,"sali_pairs_table.csv"),index=False)
print(f"\n[写出表] mutation_analysis/sali_pairs_table.csv ({len(tab)} 行)")

# 极端 case:真值大跳变(|true_jump|>4)且模型预测跳变方向/幅度严重错(diff_percent 最负)
ext=P[P.absT>4].sort_values("diff_percent").head(8)
print("\n=== §4.2 极端 cliff 失配 case(|true_jump|>4, diff_percent 最负)===")
print(ext[["PDB","site","d","ddG_A","ddG_B","true_jump","pred_jump","diff_percent"]].round(3).to_string(index=False))
