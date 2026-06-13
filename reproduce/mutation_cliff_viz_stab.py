#!/usr/bin/env python3
"""把 §4(ADiT)的全套 cliff-失效分析,在 StaB-ddG 上复刻一遍。
数据:reproduce/stab_eval_skempi_test.csv(= StaB-ddG fine-tuned 在 SKEMPIv2 test split 上的 eval,
  列 #Pdb / Mutation / ddG(真值,StaB 约定)/ ddG_pred;1491 mutations / 81 complexes)。
仅在 test split 上分析(StaB 只有该 split)。口径与 §4 完全一致:全量配对、SALI=|ΔΔΔG|/d、
per-interface K≥10。产出:reproduce/mutation_analysis_stab/。
"""
import os, ast, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
D=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(D,"mutation_analysis_stab"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"figure.dpi":130,"font.size":10})
df=pd.read_csv(os.path.join(D,"stab_eval_skempi_test.csv"))
df["mut_list"]=df["Mutation"].apply(ast.literal_eval)
def to_dict(l):
    d={}
    for m in l: d[m[1:-1]]=m[-1]
    return d
df["mdict"]=df["mut_list"].apply(to_dict); df["mkey"]=df["mdict"].apply(lambda d:frozenset(d.items()))
agg=df.groupby(["#Pdb","mkey"],as_index=False).agg(ddG=("ddG","mean"),ddG_pred=("ddG_pred","mean"),mut=("mut_list","first"))
agg["mdict"]=agg["mkey"].apply(dict); agg["ns"]=agg["mdict"].apply(len); N=agg.to_dict("records")
def dist(a,b): return sum(1 for p in set(a)|set(b) if a.get(p)!=b.get(p))
rows=[]
for name,g in agg.groupby("#Pdb"):
    idx=list(g.index)
    for ia,ib in itertools.combinations(idx,2):
        a,b=N[ia],N[ib]; d=dist(a["mdict"],b["mdict"])
        rows.append((name,ia,ib,d,a["ns"],b["ns"],a["ddG"],b["ddG"],a["ddG_pred"],b["ddG_pred"],
                     a["ddG"]-b["ddG"],a["ddG_pred"]-b["ddG_pred"]))
P=pd.DataFrame(rows,columns=["PDB","iA","iB","d","nsA","nsB","ddG_A","ddG_B","pred_A","pred_B","true_jump","pred_jump"])
P["absT"]=P.true_jump.abs(); P["absP"]=P.pred_jump.abs(); P["SALI"]=P.absT/P.d
print(f"StaB test: pairs={len(P)} d1={int((P.d==1).sum())}")

def rmse(x,y): return float(np.sqrt(np.mean((np.asarray(x)-np.asarray(y))**2)))
def safe(f,x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return np.nan
    return f(x,y)[0]
KP=10; MINN=20; MINIF=3
def perif(d,gcol,xc,yc):
    pe,sp,rm=[],[],[]
    for _,gg in d.groupby(gcol):
        if len(gg)<KP: continue
        pe.append(safe(pearsonr,gg[xc].values,gg[yc].values)); sp.append(safe(spearmanr,gg[xc].values,gg[yc].values)); rm.append(rmse(gg[xc],gg[yc]))
    nC=sum(~np.isnan(pe)) if pe else 0
    f=lambda v:(np.nanmean(v) if nC>=MINIF else np.nan)
    return f(pe),f(sp),f(rm),nC

# ===== §4.1 按 SALI 细分桶 [0,2) step 0.1 + >2 =====
BW=0.1; TOP=2.0; HIX=2.05
fedges=np.round(np.arange(0,TOP+1e-9,BW),2); ctr=fedges[:-1]+BW/2
P["fb"]=pd.cut(P.SALI,fedges,labels=False,right=False)
groups={int(k):v for k,v in P.dropna(subset=["fb"]).groupby("fb")}
HI=P[P.SALI>=TOP]; nbin=len(ctr)
mt=np.full(nbin,np.nan); mp=np.full(nbin,np.nan); mdt=np.full(nbin,np.nan); mdp=np.full(nbin,np.nan)
res={k:{m:np.full(nbin,np.nan) for m in "PSR"} for k in "abcd"}
def metrics_of(q):
    o={}; o["a"]=(safe(pearsonr,q.true_jump,q.pred_jump),safe(spearmanr,q.true_jump,q.pred_jump),rmse(q.true_jump,q.pred_jump))
    o["b"]=perif(q,"PDB","true_jump","pred_jump")[:3]
    ids=pd.unique(pd.concat([q.iA,q.iB])); M=agg.loc[ids,["#Pdb","ddG","ddG_pred"]]
    o["c"]=(safe(pearsonr,M.ddG,M.ddG_pred),safe(spearmanr,M.ddG,M.ddG_pred),rmse(M.ddG,M.ddG_pred))
    o["d"]=perif(M,"#Pdb","ddG","ddG_pred")[:3]; return o
for i in range(nbin):
    q=groups.get(i)
    if q is None or len(q)<MINN: continue
    mt[i]=q.absT.mean(); mp[i]=q.absP.mean(); mdt[i]=q.absT.median(); mdp[i]=q.absP.median()
    mo=metrics_of(q)
    for k in "abcd": res[k]["P"][i],res[k]["S"][i],res[k]["R"][i]=mo[k]
hmo=metrics_of(HI); hi_mt,hi_mp,hi_mdt,hi_mdp=HI.absT.mean(),HI.absP.mean(),HI.absT.median(),HI.absP.median()
print("\n[S4.1 flatten coarse 0-.5/.5-1/1-1.5/1.5-2/>2]")
for lo,hi in [(0,.5),(.5,1),(1,1.5),(1.5,2),(2,1e9)]:
    q=P[(P.SALI>=lo)&(P.SALI<hi)]
    if len(q): print(f"  SALI[{lo},{hi}): n={len(q)} true {q.absT.mean():.2f}/{q.absT.median():.2f} pred {q.absP.mean():.2f}/{q.absP.median():.2f}")
print("[S4.1 RMSE coarse]")
for lo,hi in [(0,.5),(.5,1),(1,1.5),(1.5,2),(2,1e9)]:
    q=P[(P.SALI>=lo)&(P.SALI<hi)]
    if len(q)<5: continue
    mo=metrics_of(q); print(f"  SALI[{lo},{hi}): a-RMSE {mo['a'][2]:.2f} b-RMSE {mo['b'][2]} c-RMSE {mo['c'][2]:.2f} d-RMSE {mo['d'][2]}")
def addhi(ax,val,c): ax.plot([HIX],[val],marker="*",ms=13,c=c)
# flatten mean / median
for tag,vt,vp,hvt,hvp in [("mean",mt,mp,hi_mt,hi_mp),("median",mdt,mdp,hi_mdt,hi_mdp)]:
    fig,ax=plt.subplots(figsize=(6.0,3.8))
    ax.plot(ctr,vt,c="indianred",lw=1.7,label=f"true |ΔΔΔG| ({tag})"); ax.plot(ctr,vp,c="steelblue",lw=1.7,label=f"predicted |ΔΔΔG| ({tag})")
    ax.fill_between(ctr,vp,vt,where=~np.isnan(vt),color="orange",alpha=0.15)
    addhi(ax,hvt,"indianred"); addhi(ax,hvp,"steelblue"); ax.axvline(TOP,ls=":",c="grey",lw=1)
    ax.set_xlim(0,2.12); ax.set_xticks([0,0.5,1,1.5,2,HIX]); ax.set_xticklabels(["0","0.5","1","1.5","2","★>2"])
    ax.set_xlabel("SALI = cliff severity (|ΔΔΔG|/d)"); ax.set_ylabel(f"|ΔΔΔG| ({tag}, kcal/mol)")
    ax.set_title(f"StaB-ddG (i-{tag}): predicted effect vs cliff severity"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"stab_flatten_{tag}.png")); plt.close(fig)
# RMSE dddg / ddg
for tag,(ko,kp,col) in {"dddg":("a","b","crimson"),"ddg":("c","d","navy")}.items():
    lbl="ΔΔΔG" if tag=="dddg" else "ΔΔG"
    fig,ax=plt.subplots(figsize=(6.2,4.0))
    ax.plot(ctr,res[ko]["R"],"-",c=col,lw=1.9,label=f"{lbl} overall")
    ax.plot(ctr,res[kp]["R"],":",c=col,lw=1.9,label=f"{lbl} per-interface")
    ax.plot([HIX],[hmo[ko][2]],"*",ms=13,c=col)
    if not np.isnan(hmo[kp][2]): ax.plot([HIX],[hmo[kp][2]],"*",ms=13,c=col)
    ax.axvline(TOP,ls=":",c="grey",lw=1); ax.set_xlim(0,2.12)
    ax.set_xticks([0,0.5,1,1.5,2,HIX]); ax.set_xticklabels(["0","0.5","1","1.5","2","★>2"])
    ax.set_xlabel("SALI = cliff severity"); ax.set_ylabel(f"{lbl} RMSE (kcal/mol, ↓ better)")
    ax.set_title(f"StaB-ddG: {lbl} RMSE vs cliff severity"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"stab_rmse_{tag}.png")); plt.close(fig)

# ===== §4.2 extreme cases: single-site & d=1 =====
ss=P[(P.nsA==1)&(P.nsB==1)&(P.d==1)].copy(); ss["jerr"]=(ss.pred_jump-ss.true_jump).abs()
def lab2(rec_iA,rec_iB):
    a=N[rec_iA]["mut"][0]; b=N[rec_iB]["mut"][0]; return a[1:-1],a[0]+"→"+a[-1],b[0]+"→"+b[-1]
print("\n[S4.2 single-site&d=1 cases] n=",len(ss))
def show(q,k):
    out=[]
    for _,r in q.iterrows():
        site,A,B=lab2(int(r.iA),int(r.iB)); out.append((r.PDB,site,A,B,r.ddG_A,r.ddG_B,r.true_jump,r.pred_jump))
    return out
norm=ss[(ss.true_jump.abs()<0.5)&(ss.jerr<0.3)].drop_duplicates("PDB").head(5)
ab=ss[ss.true_jump.abs()>2].sort_values("jerr",ascending=False).drop_duplicates("PDB").head(5)
print("normal:");  [print("|",*[f"{v}" for v in r],"|") for r in show(norm,5)]
print("abnormal:");[print("|",*[f"{v}" for v in r],"|") for r in show(ab,5)]

# ===== §4.2 d=1 误差直方图 [0,5) step 0.5 + >5 (percentage) =====
d1=P[P.d==1].copy(); d1["jerr"]=(d1.pred_jump-d1.true_jump).abs()
print(f"\n[S4.2 d=1 jerr] n={len(d1)} median={d1.jerr.median():.2f} p90={d1.jerr.quantile(.9):.2f} >5pct={100*(d1.jerr>5).mean():.1f}%")
hedges=[round(0.5*i,1) for i in range(5)]+[1e9]   # [0,2) step 0.5 + >2
XT=[0,1,2,3,4]; XL=["0","0.5","1","1.5",">2"]; FS=(8.8,4.6)
hpct=100*np.histogram(d1.jerr.values,bins=hedges)[0]/len(d1); hxs=np.arange(len(hpct))
fig,ax=plt.subplots(figsize=FS)
ax.bar(hxs,hpct,width=0.9,color="steelblue",edgecolor="white",linewidth=0.3)
for x,v in zip(hxs,hpct): ax.text(x,v+0.5,f"{v:.1f}",ha="center",va="bottom",fontsize=11)
ax.set_ylim(0,np.nanmax(hpct)*1.13)
ax.set_xticks(XT); ax.set_xticklabels(XL)
ax.set_xlabel("|predicted jump − true jump| (kcal/mol, d=1; bin=0.5 in [0,2], last=>2)")
ax.set_ylabel("percentage of d=1 pairs (%)"); ax.set_title(f"StaB-ddG d=1 jump-error (n={len(d1)}; median={d1.jerr.median():.2f})")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"stab_d1_jumperr_hist.png")); plt.close(fig)
# sign-flip by |error|
d1["eb"]=pd.cut(d1.jerr,hedges,labels=False,right=False)
sf=np.full(len(hpct),np.nan)
for k,g in d1.groupby("eb"): sf[int(k)]=100*(np.sign(g.true_jump)!=np.sign(g.pred_jump)).mean()
print("[S4.2 signflip by err]", {i:round(v) for i,v in enumerate(sf) if not np.isnan(v)})
fig,ax=plt.subplots(figsize=FS)
ax.bar(hxs,sf,width=0.9,color="indianred",edgecolor="white",linewidth=0.3)
for x,v in zip(hxs,sf):
    if not np.isnan(v): ax.text(x,v+1.5,f"{v:.0f}",ha="center",va="bottom",fontsize=11)
ax.set_xticks(XT); ax.set_xticklabels(XL); ax.set_ylim(0,100)
ax.set_xlabel("|predicted jump − true jump| bin (kcal/mol, d=1; bin=0.5 in [0,2], last=>2)"); ax.set_ylabel("% sign-opposite within bin")
ax.set_title(f"StaB-ddG d=1: sign-flip vs jump-error (n={len(d1)})")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"stab_d1_signflip_by_err.png")); plt.close(fig)

# ===== §4.2 按 |true jump| 分桶 [0,2) step 0.5 + >2 =====
tj=d1.true_jump.abs(); tb=pd.cut(tj,hedges,labels=False,right=False)
tpct=np.full(len(hpct),np.nan); terr=np.full(len(hpct),np.nan); tflip=np.full(len(hpct),np.nan)
for k,idx in d1.assign(_t=tb).groupby("_t").groups.items():
    k=int(k); g=d1.loc[idx]; tpct[k]=100*len(g)/len(d1); terr[k]=(g.pred_jump-g.true_jump).abs().mean(); tflip[k]=100*(np.sign(g.true_jump)!=np.sign(g.pred_jump)).mean()
print("[S4.2 by |true jump|] pct:",{i:round(v,1) for i,v in enumerate(tpct) if not np.isnan(v)})
print("  meanErr:",{i:round(v,2) for i,v in enumerate(terr) if not np.isnan(v)})
print("  flip:",{i:round(v) for i,v in enumerate(tflip) if not np.isnan(v)})
def _xt(ax): ax.set_xticks(XT); ax.set_xticklabels(XL); ax.set_xlabel("|true jump| bin (kcal/mol, d=1; bin=0.5 in [0,2], last=>2)")
fig,ax=plt.subplots(figsize=FS); ax.bar(hxs,tpct,width=0.9,color="steelblue",edgecolor="white",linewidth=0.3); _xt(ax)
for x,v in zip(hxs,tpct):
    if not np.isnan(v): ax.text(x,v+0.5,f"{v:.1f}",ha="center",va="bottom",fontsize=11)
ax.set_ylim(0,np.nanmax(tpct)*1.13); ax.set_ylabel("percentage of d=1 pairs (%)"); ax.set_title(f"StaB-ddG d=1: distribution of |true jump| (n={len(d1)})")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"stab_byTrueJump_pct.png")); plt.close(fig)
fig,ax=plt.subplots(figsize=FS); ax.bar(hxs,terr,width=0.9,color="seagreen",edgecolor="white",linewidth=0.3); _xt(ax)
for x,v in zip(hxs,terr):
    if not np.isnan(v): ax.text(x,v+0.04,f"{v:.2f}",ha="center",va="bottom",fontsize=11)
ax.set_ylim(0,np.nanmax(terr)*1.13); ax.set_ylabel("mean |pred − true jump| (kcal/mol)"); ax.set_title("StaB-ddG d=1: error vs true cliff magnitude")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"stab_byTrueJump_err.png")); plt.close(fig)
fig,ax=plt.subplots(figsize=FS); ax.bar(hxs,tflip,width=0.9,color="indianred",edgecolor="white",linewidth=0.3); _xt(ax)
for x,v in zip(hxs,tflip):
    if not np.isnan(v): ax.text(x,v+1.5,f"{v:.0f}",ha="center",va="bottom",fontsize=11)
ax.set_ylim(0,100); ax.set_ylabel("% sign-opposite within bin"); ax.set_title("StaB-ddG d=1: sign-flip vs true cliff magnitude")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"stab_byTrueJump_signflip.png")); plt.close(fig)

# ===== overview 统计(同位点 + 全体配对) =====
print("\n[overview]")
print(f"  overall(per-mutant,distinct): Pearson {pearsonr(agg.ddG,agg.ddG_pred)[0]:.3f} Spearman {spearmanr(agg.ddG,agg.ddG_pred)[0]:.3f} RMSE {rmse(agg.ddG,agg.ddG_pred):.3f}")
print(f"  jump(全体对): Pearson {safe(pearsonr,P.true_jump,P.pred_jump):.3f} Spearman {safe(spearmanr,P.true_jump,P.pred_jump):.3f}")
cl=P[P.absT>2]; print(f"  cliff对(|true|>2) n={len(cl)} capture(median |pred|/|true|)={(cl.absP/cl.absT).median():.2f}")
print("[figs]", sorted(os.listdir(OUT)))
