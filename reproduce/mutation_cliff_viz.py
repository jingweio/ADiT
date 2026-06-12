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

# ======================= §4 按 SALI 细分桶(step=0.1)=======================
def rmse(x,y): return float(np.sqrt(np.mean((np.asarray(x)-np.asarray(y))**2)))
def safe(f,x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return np.nan
    return f(x,y)[0]
KP=10            # per-interface: 每个 complex 至少 KP 个样本
MINN=40          # 每个 fine bin 至少 MINN 对才计点
MINIF=3          # per-interface: 至少 MINIF 个合格 complex 才计点
def perif(d,gcol,xc,yc):
    pe,sp,rm=[],[],[]
    for _,gg in d.groupby(gcol):
        if len(gg)<KP: continue
        pe.append(safe(pearsonr,gg[xc].values,gg[yc].values))
        sp.append(safe(spearmanr,gg[xc].values,gg[yc].values))
        rm.append(rmse(gg[xc],gg[yc]))
    n=sum(~np.isnan(pe)) if pe else 0
    f=lambda v:(np.nanmean(v) if n>=MINIF else np.nan)
    return f(pe),f(sp),f(rm),n

BW=0.01; TOP=2.0; HIX=2.05      # [0,2) step=0.01 细桶 + 一个 >2 聚合点(画在 x=2.05)
fedges=np.round(np.arange(0,TOP+1e-9,BW),2); ctr=fedges[:-1]+BW/2
P["fb"]=pd.cut(P.SALI,fedges,labels=False,right=False)
groups={int(k):v for k,v in P.dropna(subset=["fb"]).groupby("fb")}
HI=P[P.SALI>=TOP]               # >2 聚合桶
n=len(ctr)
mt=np.full(n,np.nan); mp=np.full(n,np.nan); mdt=np.full(n,np.nan); mdp=np.full(n,np.nan)
res={k:{m:np.full(n,np.nan) for m in "PSR"} for k in "abcd"}
def metrics_of(q):
    out={}
    out["a"]=(safe(pearsonr,q.true_jump,q.pred_jump),safe(spearmanr,q.true_jump,q.pred_jump),rmse(q.true_jump,q.pred_jump))
    out["b"]=perif(q,"PDB","true_jump","pred_jump")[:3]
    ids=pd.unique(pd.concat([q.iA,q.iB])); M=agg.loc[ids,["Name","ddG","ddG_pred"]]
    out["c"]=(safe(pearsonr,M.ddG,M.ddG_pred),safe(spearmanr,M.ddG,M.ddG_pred),rmse(M.ddG,M.ddG_pred))
    out["d"]=perif(M,"Name","ddG","ddG_pred")[:3]
    return out,M
for i in range(n):
    q=groups.get(i)
    if q is None or len(q)<MINN: continue
    mt[i]=q.absT.mean(); mp[i]=q.absP.mean(); mdt[i]=q.absT.median(); mdp[i]=q.absP.median()
    mo,_=metrics_of(q)
    for k in "abcd":
        res[k]["P"][i],res[k]["S"][i],res[k]["R"][i]=mo[k]
hmo,HM=metrics_of(HI)           # >2 聚合
hi_mt,hi_mp,hi_mdt,hi_mdp=HI.absT.mean(),HI.absP.mean(),HI.absT.median(),HI.absP.median()
nval={k:int(np.sum(~np.isnan(res[k]["P"]))) for k in "abcd"}
print(f"\n=== §4 fine bins: {n} 个(step={BW}, 0–{TOP})+>2 | 有效点 a/b/c/d = {nval} ===")

# ---- §4 (i):flatten,mean 与 median 两张图(0–2 细线 + >2 聚合点)----
def addhi(ax,val): ax.plot([HIX],[val],marker="*",ms=12,c="k",zorder=5)
for tag,vt,vp,hvt,hvp in [("mean",mt,mp,hi_mt,hi_mp),("median",mdt,mdp,hi_mdt,hi_mdp)]:
    fig,ax=plt.subplots(figsize=(5.9,3.8))
    ax.plot(ctr,vt,c="indianred",lw=1.7,label=f"true |ΔΔΔG| ({tag})")
    ax.plot(ctr,vp,c="steelblue",lw=1.7,label=f"predicted |ΔΔΔG| ({tag})")
    ax.fill_between(ctr,vp,vt,where=~np.isnan(vt),color="orange",alpha=0.15)
    ax.plot([HIX],[hvt],marker="*",ms=13,c="indianred"); ax.plot([HIX],[hvp],marker="*",ms=13,c="steelblue")
    ax.axvline(TOP,ls=":",c="grey",lw=1)
    ax.set_xlim(0,2.12); ax.set_xticks([0,0.5,1,1.5,2,HIX]); ax.set_xticklabels(["0","0.5","1","1.5","2","★>2"])
    ax.set_xlabel("SALI = cliff severity (|ΔΔΔG|/d)"); ax.set_ylabel(f"|ΔΔΔG| ({tag}, kcal/mol)")
    ax.set_title(f"(i-{tag}) predicted effect flattens as cliff grows"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"adit_cliff_flatten_{tag}.png")); plt.close(fig)

# ---- §4 (ii):12 指标(a/b/c/d × P/S/RMSE),0–2 细线 + >2 聚合点 ----
titles={"a":"(a) ΔΔΔG overall","b":"(b) ΔΔΔG per-interface","c":"(c) ΔΔG overall","d":"(d) ΔΔG per-interface"}
fig,axes=plt.subplots(2,2,figsize=(11,7.4))
for ax,k in zip(axes.flat,"abcd"):
    ax.plot(ctr,res[k]["P"],"-",c="crimson",lw=1.4,label="Pearson")
    ax.plot(ctr,res[k]["S"],"-",c="navy",lw=1.4,label="Spearman")
    ax.plot([HIX],[hmo[k][0]],"*",ms=11,c="crimson"); ax.plot([HIX],[hmo[k][1]],"*",ms=11,c="navy")
    ax.axvline(TOP,ls=":",c="grey",lw=1)
    ax.set_ylim(0,1); ax.set_xlim(0,2.12); ax.set_xticks([0,0.5,1,1.5,2,HIX]); ax.set_xticklabels(["0",".5","1","1.5","2","★>2"])
    ax.set_ylabel("correlation (↑=better)"); ax.set_title(titles[k]); ax.axhline(0,ls=":",c="grey")
    axr=ax.twinx(); axr.plot(ctr,res[k]["R"],"-",c="darkorange",lw=1.4,label="RMSE"); axr.plot([HIX],[hmo[k][2]],"*",ms=11,c="darkorange")
    axr.set_ylabel("RMSE (↓=better)")
    h1,l1=ax.get_legend_handles_labels(); h2,l2=axr.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,fontsize=7.5,loc="center left")
    ax.set_xlabel("SALI = cliff severity")
fig.suptitle("Per-SALI-bin (step=0.01, 0–2) + >2: correlations RISE (SNR artifact), only RMSE rises=worse",fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"adit_cliff_metrics.png")); plt.close(fig)

# ---- coarse 汇总表(md 用): 0-.5,.5-1,1-1.5,1.5-2,>2 ----
cedges=[0,0.5,1,1.5,2,1e9]; clab=["0-.5",".5-1","1-1.5","1.5-2",">2"]
P["cb"]=pd.cut(P.SALI,cedges,labels=clab,right=False)
print("\n=== §4 coarse 汇总(table 用)===")
for L in clab:
    q=P[P.cb==L]; mo,M=metrics_of(q)
    print(f"  {L}: n={len(q)} | true {q.absT.mean():.2f}/{q.absT.median():.2f} pred {q.absP.mean():.2f}/{q.absP.median():.2f} "
          f"| a {mo['a'][0]:.2f}/{mo['a'][1]:.2f}/{mo['a'][2]:.2f} | b {mo['b'][0]:.2f}/{mo['b'][1]:.2f}/{mo['b'][2]:.2f} "
          f"| c {mo['c'][0]:.2f}/{mo['c'][1]:.2f}/{mo['c'][2]:.2f} | d {mo['d'][0]:.2f}/{mo['d'][1]:.2f}/{mo['d'][2]:.2f}")

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
