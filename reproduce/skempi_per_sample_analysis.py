#!/usr/bin/env python3
"""把 skempi 三折测试的逐样本预测拆出来写成 CSV(格式参考 demo_only.csv),
并按"突变位点数(number of mutation sites)"统计数据分布与模型指标。
数据源:reproduce/result_split_{0,1,2}.pkl —— 这就是三折测试时模型保存的逐样本推理结果。
"""
import pickle, os
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr as _pr, spearmanr as _sr
    def pearson(a, b): return float(_pr(a, b)[0])
    def spearman(a, b): return float(_sr(a, b)[0])
except Exception:
    def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])
    def spearman(a, b):
        ra = pd.Series(a).rank().values; rb = pd.Series(b).rank().values
        return float(np.corrcoef(ra, rb)[0, 1])

REPRO = os.path.dirname(os.path.abspath(__file__))


def parse_code(code):
    """'3QIB_ABP_CD_QA57A_1905' -> name='3QIB_ABP_CD', mut='QA57A', rec='1905'
    突变 token 不含下划线(多突变用逗号),记录号在末尾,故 split('_') 后:
    [:-2]=name, [-2]=mutation, [-1]=record."""
    parts = code.split("_")
    name = "_".join(parts[:-2])
    mutation = parts[-2]
    record = parts[-1]
    sites = mutation.split(",")
    return name, sites, record


rows = []
for s in [0, 1, 2]:
    with open(os.path.join(REPRO, f"result_split_{s}.pkl"), "rb") as f:
        codes, pred, target = pickle.load(f)
    pred = np.asarray(pred.cpu() if hasattr(pred, "cpu") else pred, dtype=float).reshape(-1)
    target = np.asarray(target.cpu() if hasattr(target, "cpu") else target, dtype=float).reshape(-1)
    for i, code in enumerate(codes):
        name, sites, record = parse_code(code)
        rows.append({
            "Name": name,
            "Mutation": str(sites),          # 形如 ['QA57A'] / ['DA11A','MA14V',...]
            "ddG": target[i],                # 真值
            "ddG_pred": pred[i],             # 模型预测
            "#Pdb": name,
            "num_mutation_sites": len(sites),
            "split": f"split_{s}",
            "accession_code": code,
        })

df = pd.DataFrame(rows)
df.index.name = "Unnamed: 0"
out_csv = os.path.join(REPRO, "skempi_per_sample_pred.csv")
df.to_csv(out_csv)
print(f"[写出] {out_csv}  共 {len(df)} 个样本\n")

# ---------- (1) 按突变位点数的数据分布 ----------
print("=" * 60)
print("(1) 数据分布(按突变位点数)")
print("=" * 60)
total = len(df)
maxn = int(df["num_mutation_sites"].max())
dist_rows = []
for n in range(1, maxn + 1):
    c = int((df["num_mutation_sites"] == n).sum())
    dist_rows.append({"n_sites": n, "count": c, "pct(%)": round(100 * c / total, 2)})
dist = pd.DataFrame(dist_rows)
print(dist.to_string(index=False))
n_single = int((df["num_mutation_sites"] == 1).sum())
n_multi = total - n_single
print(f"\n汇总: single(n=1) = {n_single} ({100*n_single/total:.2f}%) | "
      f"multi(n>1) = {n_multi} ({100*n_multi/total:.2f}%) | total = {total}")

# ---------- (2) 按突变位点数的模型指标 ----------
def metrics(sub):
    p = sub["ddG_pred"].values; t = sub["ddG"].values
    if len(sub) < 2 or np.std(p) == 0 or np.std(t) == 0:
        pr = sr = float("nan")
    else:
        pr = pearson(p, t); sr = spearman(p, t)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    return len(sub), pr, sr, rmse, mae

print("\n" + "=" * 60)
print("(2) 模型指标(按突变位点数)")
print("=" * 60)
met_rows = []
for n in range(1, maxn + 1):
    sub = df[df["num_mutation_sites"] == n]
    if len(sub) == 0:
        continue
    cnt, pr, sr, rmse, mae = metrics(sub)
    met_rows.append({"n_sites": n, "count": cnt,
                     "Pearsonr": round(pr, 4), "Spearmanr": round(sr, 4),
                     "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
# single / multi / overall
for label, sub in [("single(n=1)", df[df["num_mutation_sites"] == 1]),
                   ("multi(n>1)", df[df["num_mutation_sites"] > 1]),
                   ("overall(all)", df)]:
    cnt, pr, sr, rmse, mae = metrics(sub)
    met_rows.append({"n_sites": label, "count": cnt,
                     "Pearsonr": round(pr, 4), "Spearmanr": round(sr, 4),
                     "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
met = pd.DataFrame(met_rows)
print(met.to_string(index=False))
met.to_csv(os.path.join(REPRO, "skempi_metrics_by_nsites.csv"), index=False)
print(f"\n[写出] {os.path.join(REPRO, 'skempi_metrics_by_nsites.csv')}")
