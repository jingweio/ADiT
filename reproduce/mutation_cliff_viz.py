#!/usr/bin/env python3
"""Mutation-cliff 深度可视化 + ADiT 失效系统验证。
数据:SKEMPIv2 / ADiT-S 三折池化预测 reproduce/skempi_per_sample_pred.csv。
口径:每个 complex 内 distinct mutant(同突变集合的重复测量先取均值)两两任意配对
  (distinct mutant >250 的大 complex 做 CAP=250 随机采样, seed=0),d=两 mutant 序列逐点比对
  的不同位点数, |ΔΔΔG|=真值 ddG 差, SALI=|ΔΔΔG|/d(cliff 指数);再 pool 全部 complex。
产出(reproduce/mutation_analysis/):
  §3  ecdf_sali_*, qmean_sali_*, pointcloud_sali_by_d        —— cliff 存在性(SALI)
  §4  adit_cliff_flatten.png   —— 真值 vs 预测 |ΔΔΔG|(mean)随 cliff 加剧 flatten
      adit_cliff_metrics.png   —— 按 SALI 分桶的 12 个模型指标(a/b/c/d × P/S/RMSE)
      sali_pairs_table.csv      —— 全配对表
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
N = agg.to_dict("records")
print(f"distinct mutant: {len(agg)} | complexes: {agg.Name.nunique()}")

def dist(a, b):
    return sum(1 for p in set(a) | set(b) if a.get(p) != b.get(p))
def diffpos(a, b):
    return ",".join(sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p)))

rng = np.random.default_rng(0); CAP = 250; rows = []
for name, g in agg.groupby("Name"):
    idx = list(g.index)
    if len(idx) > CAP:
        idx = list(rng.choice(idx, CAP, replace=False))
    for ia, ib in itertools.combinations(idx, 2):
        a, b = N[ia], N[ib]
        d = dist(a["mdict"], b["mdict"])
        dT = a["ddG"] - b["ddG"]; dP = a["ddG_pred"] - b["ddG_pred"]
        rows.append((name, diffpos(a["mdict"], b["mdict"]), ia, ib, d,
                     a["ddG"], b["ddG"], a["ddG_pred"], b["ddG_pred"],
                     dT, dP, abs(dT), abs(dP), abs(dT)/d))
P = pd.DataFrame(rows, columns=["PDB","site","iA","iB","d","ddG_A","ddG_B","pred_A","pred_B",
                                "true_jump","pred_jump","absT","absP","SALI"])
print(f"突变对总数: {len(P)}")

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

print("\n=== §3 SALI 统计(by d + all)===")
def block(x):
    x=np.asarray(x)
    return dict(n=len(x),median=np.median(x),CV=x.std()/x.mean(),
                p90=np.quantile(x,.9),p99=np.quantile(x,.99),mx=x.max(),
                smooth=(x<0.5).mean(),cliff=(x>2).mean())
for d in DS:
    print(f"d={d}:",{k:(round(v,3) if k!='n' else int(v)) for k,v in block(P[P.d==d].SALI).items()})
print("ALL:",{k:(round(v,3) if k!='n' else int(v)) for k,v in block(P.SALI).items()})
print(f"corr(d, SALI): Pearson={pearsonr(P.d,P.SALI)[0]:.3f} Spearman={spearmanr(P.d,P.SALI)[0]:.3f}")

# ======================= §4 按 SALI(cliff 严重度)分桶 =======================
sedges=[0,0.5,1,1.5,2,3,4,1e9]
slab=["0-.5",".5-1","1-1.5","1.5-2","2-3","3-4",">4"]
P["sbin"]=pd.cut(P.SALI,sedges,labels=slab,right=False)
xs=np.arange(len(slab))

# ---- §4 fig1:真值 vs 预测 |ΔΔΔG| 的 mean(flatten)----
m_true=[P[P.sbin==L].absT.mean() for L in slab]
m_pred=[P[P.sbin==L].absP.mean() for L in slab]
print("\n=== §4 fig1: 真值 vs 预测 |ΔΔΔG| mean ===")
for L,t,p in zip(slab,m_true,m_pred): print(f"  {L}: true {t:.2f} pred {p:.2f}")
w=0.4
fig,ax=plt.subplots(figsize=(6.0,3.9))
ax.bar(xs-w/2,m_true,w,color="indianred",label="true |ΔΔΔG| (mean)")
ax.bar(xs+w/2,m_pred,w,color="steelblue",label="predicted |ΔΔΔG| (mean)")
ax.set_xticks(xs); ax.set_xticklabels(slab,rotation=45)
ax.set_xlabel("true SALI bin = cliff severity"); ax.set_ylabel("|ΔΔΔG| (mean, kcal/mol)")
ax.set_title("As cliff severity grows, predicted effect lags true (flatten)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"adit_cliff_flatten.png")); plt.close(fig)

# ---- §4 fig2:12 个模型指标(a/b/c/d × Pearson/Spearman/RMSE)----
def rmse(x,y): return float(np.sqrt(np.mean((np.asarray(x)-np.asarray(y))**2)))
def safe(f,x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return np.nan
    return f(x,y)[0]
KP=10
def perif(d,gcol,xc,yc):
    pe,sp,rm=[],[],[]
    for _,gg in d.groupby(gcol):
        if len(gg)<KP: continue
        pe.append(safe(pearsonr,gg[xc].values,gg[yc].values))
        sp.append(safe(spearmanr,gg[xc].values,gg[yc].values))
        rm.append(rmse(gg[xc],gg[yc]))
    f=lambda v:np.nanmean(v) if len(v) else np.nan
    return f(pe),f(sp),f(rm),sum(~np.isnan(pe)) if pe else 0

res={k:{m:[] for m in ["P","S","R"]} for k in "abcd"}
nif={"b":[],"d":[]}
print("\n=== §4 fig2: 12 指标 by SALI 桶 ===")
for L in slab:
    q=P[P.sbin==L]
    res["a"]["P"].append(safe(pearsonr,q.true_jump,q.pred_jump)); res["a"]["S"].append(safe(spearmanr,q.true_jump,q.pred_jump)); res["a"]["R"].append(rmse(q.true_jump,q.pred_jump))
    bp,bs,br,bn=perif(q,"PDB","true_jump","pred_jump"); res["b"]["P"].append(bp); res["b"]["S"].append(bs); res["b"]["R"].append(br); nif["b"].append(bn)
    ids=pd.unique(pd.concat([q.iA,q.iB])); M=agg.loc[ids,["Name","ddG","ddG_pred"]]
    res["c"]["P"].append(safe(pearsonr,M.ddG,M.ddG_pred)); res["c"]["S"].append(safe(spearmanr,M.ddG,M.ddG_pred)); res["c"]["R"].append(rmse(M.ddG,M.ddG_pred))
    dp,ds,dr,dn=perif(M,"Name","ddG","ddG_pred"); res["d"]["P"].append(dp); res["d"]["S"].append(ds); res["d"]["R"].append(dr); nif["d"].append(dn)
    print(f"  {L}: a {res['a']['P'][-1]:.2f}/{res['a']['S'][-1]:.2f}/{res['a']['R'][-1]:.2f} | "
          f"b {bp:.2f}/{bs:.2f}/{br:.2f} | c {res['c']['P'][-1]:.2f}/{res['c']['S'][-1]:.2f}/{res['c']['R'][-1]:.2f} | d {dp:.2f}/{ds:.2f}/{dr:.2f}")

titles={"a":"(a) ΔΔΔG overall","b":"(b) ΔΔΔG per-interface","c":"(c) ΔΔG overall","d":"(d) ΔΔG per-interface"}
fig,axes=plt.subplots(2,2,figsize=(11,7.4))
for ax,k in zip(axes.flat,"abcd"):
    ax.plot(xs,res[k]["P"],"-o",c="crimson",lw=1.7,ms=4,label="Pearson")
    ax.plot(xs,res[k]["S"],"-s",c="navy",lw=1.7,ms=4,label="Spearman")
    ax.set_ylim(0,1); ax.set_xticks(xs); ax.set_xticklabels(slab,rotation=45)
    ax.set_ylabel("correlation (↑=better)"); ax.set_title(titles[k]); ax.axhline(0,ls=":",c="grey")
    axr=ax.twinx(); axr.plot(xs,res[k]["R"],"-^",c="darkorange",lw=1.7,ms=4,label="RMSE")
    axr.set_ylabel("RMSE (↓=better)")
    h1,l1=ax.get_legend_handles_labels(); h2,l2=axr.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,fontsize=7.5,loc="upper left")
    ax.set_xlabel("true SALI bin = cliff severity")
fig.suptitle("Per-SALI-bin model metrics: correlations RISE (SNR artifact), only RMSE rises=worse",fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"adit_cliff_metrics.png")); plt.close(fig)

# ======================= 表:diff_percent =======================
P["diff_percent"]=(P.pred_jump-P.true_jump)/P.true_jump.replace(0,np.nan)
tab=P[["PDB","site","d","ddG_A","ddG_B","true_jump","pred_jump","diff_percent"]].copy()
tab=tab.round({"ddG_A":3,"ddG_B":3,"true_jump":3,"pred_jump":3,"diff_percent":4})
tab["_a"]=tab.diff_percent.abs()
tab=tab.sort_values(["PDB","d","_a"],ascending=[True,True,False],na_position="last").drop(columns="_a")
tab.to_csv(os.path.join(OUT,"sali_pairs_table.csv"),index=False)
print(f"\n[写出表] mutation_analysis/sali_pairs_table.csv ({len(tab)} 行)")

ext=P[P.absT>4].sort_values("diff_percent").head(8)
print("\n=== §4.2 极端 cliff 失配 case(|true_jump|>4, diff_percent 最负)===")
print(ext[["PDB","site","d","ddG_A","ddG_B","true_jump","pred_jump","diff_percent"]].round(3).to_string(index=False))
