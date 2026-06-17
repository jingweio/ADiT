#!/usr/bin/env python
"""
optuna_retune.py  ——  adit-retune-hyperparams 的并行 Optuna worker

为 ADiT-M 搜索超参。**目标 = overall RMSE(在 test split 上算,minimize)**。
⚠ 注意:本轮按用户决定**直接在 stab_test 上调参**(不再从 stab_train 切 val)——因此最终 S/M/L 的 test 数字
带乐观偏差、**不是对 StaB-ddG 的公平 held-out 对比**,results 须显著标注为 "test-tuned 上界/探索"。

每个 worker = 1 个 SLURM a100 任务,跑 --n-trials 个 trial;多个 worker 通过共享
JournalStorage(放 /ibex 共享盘,适合并行写)协作同一个 study(TPE)。

每个 trial:
  1) train.sh fine-tune ADiT-M(从 ckpts/adit_M.ckpt 起),stab_train 训练 + 内部随机 val 早停;
  2) test.sh 在 stab_test 上推理,落 pkl(accession, pred, target);
  3) 读 pkl,算 overall RMSE 作为目标返回(并记 per-iface Spearman / overall P,S 供报告)。
失败(如 trunc256×8 OOM)→ 返回大 RMSE(FAIL_RMSE)并记 fail 原因,让 TPE 避开。

搜索空间(见下方 suggest_*):
  truncation_size ∈ {160,128,64,32}(bs 固定 8;160 = 80G a100 上 bs8 实测能放下的最大档,192/200/256 均 OOM) / lr ∈ {1e-5,5e-5,1e-4,5e-4,1e-3}
  / dropout ∈ {0.0~0.5 by 0.1} / weight_decay ∈ {1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2}
  / scheduler.patience ∈ {20,50,100}。epochs:max_epochs 500 封顶 + EarlyStopping patience 100。
  batch_size 固定 8(= ADiT 原模型默认),不进搜索。

用法(单 worker):
  python reproduce/optuna_retune.py --study-name adit_M_retune \
      --storage /ibex/user/guoj0f/repos/ibex_records/adit-retune-hyperparams/optuna_aditM.journal \
      --n-trials 4 --worker-id 0
"""
import os, sys, glob, time, argparse, pickle, subprocess
import numpy as np
from scipy.stats import spearmanr, pearsonr
import optuna

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_storage(path):
    """JournalStorage(file backend) —— 网络盘并发安全;兼容 optuna 3.x/4.x 命名。"""
    from optuna.storages import JournalStorage
    try:
        from optuna.storages.journal import JournalFileBackend as Backend  # optuna>=4
    except Exception:
        from optuna.storages import JournalFileStorage as Backend          # optuna 3.x
    return JournalStorage(Backend(path))


def to_np(x):
    import torch
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x, dtype=float).reshape(-1)


def per_interface_spearman(pkl_path, min_k=10):
    """读 (accession, pred, target) pkl,返回 (per_iface_mean, n_used, overall dict)。"""
    import torch  # noqa: 反序列化 cuda-saved tensor
    with open(pkl_path, "rb") as f:
        accs, pred, target = pickle.load(f)
    pred, target = to_np(pred), to_np(target)
    overall = {
        "pearson": float(pearsonr(pred, target)[0]),
        "spearman": float(spearmanr(pred, target)[0]),
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
        "n": int(len(accs)),
    }
    groups = {}
    for a, p, t in zip(accs, pred, target):
        cid = "_".join(a.split("_")[:3])
        groups.setdefault(cid, [[], []])
        groups[cid][0].append(p); groups[cid][1].append(t)
    per = []
    for cid, (ps, ts) in groups.items():
        if len(ps) < min_k:
            continue
        if len(set(ts)) < 2 or len(set(ps)) < 2:
            continue
        s = spearmanr(ps, ts)[0]
        if np.isfinite(s):
            per.append(s)
    mean_per = float(np.mean(per)) if per else float("nan")
    return mean_per, len(per), overall


def run(cmd, log_path):
    with open(log_path, "ab") as lf:
        lf.write((">>> " + " ".join(cmd) + "\n").encode())
        lf.flush()
        p = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
    return p.returncode


FAIL_RMSE = 1e6  # 失败哨兵(目标是 minimize overall RMSE,失败→给个大 RMSE 让 TPE 避开)


def build_objective(args):
    def objective(trial):
        # 搜索空间(2026-06 用户敲定):bs 固定 8(原模型默认),不再自适应
        ts = trial.suggest_categorical("truncation_size", [160, 128, 64, 32])
        lr = trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        wd = trial.suggest_categorical("weight_decay", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        sched_pat = trial.suggest_categorical("scheduler_patience", [20, 50, 100])
        bs = 8  # 固定

        tag = f"retuneM_w{args.worker_id}_t{trial.number}"
        trial_pkl = os.path.join(args.work_dir, f"{tag}.pkl")
        os.makedirs(args.work_dir, exist_ok=True)

        common = [
            "++trainer.devices=1", "++trainer.strategy=auto",
            f"++data.dataset.path_to_dataset={args.dataset}",
            f"++data.dataset.test_split={args.test_split}",
            f"++data.dataset.truncation_size={ts}",
            f"++model.net.dropout={dropout}",
        ]
        t0 = time.time()
        # 单次 train(bs=8 固定);trunc=256×8 可能 OOM -> 该 trial 判失败(返回大 RMSE,TPE 避开)
        train_log = os.path.join(args.work_dir, f"{tag}_train.log")
        train_cmd = ["bash", "train.sh", f"experiment={args.experiment}"] + common + [
            f"++data.batch_size={bs}",
            f"++model.optimizer.lr={lr}",
            f"++model.optimizer.weight_decay={wd}",
            f"++model.scheduler.patience={sched_pat}",
            "++trainer.min_epochs=1", f"++trainer.max_epochs={args.max_epochs}",
            "++callbacks.early_stopping.monitor=val/mse_loss",
            "++callbacks.early_stopping.mode=min",
            f"++callbacks.early_stopping.patience={args.es_patience}",
            "++callbacks.model_checkpoint.save_last=true",
            f"ckpt_path={args.base_ckpt}",
            f"task_name={tag}",
        ]
        rc = run(train_cmd, train_log)
        if rc != 0:
            oom = False
            try:
                with open(train_log, "rb") as lf:
                    txt = lf.read().decode("utf-8", "ignore")
                oom = ("OutOfMemoryError" in txt) or ("CUDA out of memory" in txt)
            except Exception:
                pass
            trial.set_user_attr("fail", f"train rc={rc} ({'OOM' if oom else 'non-OOM'}, ts={ts}, bs={bs})")
            return FAIL_RMSE

        ckpts = sorted(glob.glob(os.path.join(REPO, "outputs", "*", f"{tag}_*", "checkpoints", "last.ckpt")),
                       key=os.path.getmtime)
        if not ckpts:
            trial.set_user_attr("fail", "no last.ckpt")
            return FAIL_RMSE
        last_ckpt = ckpts[-1]

        test_cmd = ["bash", "test.sh", f"experiment={args.experiment}"] + common + [
            "++data.batch_size=1",
            f"++model.save_file={trial_pkl}",
            f"ckpt_path={last_ckpt}",
        ]
        rc = run(test_cmd, os.path.join(args.work_dir, f"{tag}_test.log"))
        if rc != 0 or not os.path.exists(trial_pkl):
            trial.set_user_attr("fail", f"test rc={rc}")
            return FAIL_RMSE

        mean_per, n_used, overall = per_interface_spearman(trial_pkl, min_k=args.min_k)
        rmse = overall["rmse"]
        trial.set_user_attr("elapsed_min", round((time.time() - t0) / 60, 1))
        trial.set_user_attr("n_used_complexes", n_used)
        trial.set_user_attr("per_iface_spearman", mean_per)
        trial.set_user_attr("overall", overall)
        trial.set_user_attr("effective_batch_size", bs)
        trial.set_user_attr("last_ckpt", last_ckpt)
        if not np.isfinite(rmse):
            trial.set_user_attr("fail", "rmse nan")
            return FAIL_RMSE
        print(f"[trial {trial.number}] ts={ts} lr={lr:.0e} do={dropout} wd={wd:.0e} bs={bs} "
              f"schedpat={sched_pat} -> overall RMSE={rmse:.4f} "
              f"(per-iface Sp K>={args.min_k}={mean_per:.3f}, overall P={overall['pearson']:.3f})", flush=True)
        return rmse
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--storage", required=True, help="JournalStorage 文件路径(/ibex 共享盘)")
    ap.add_argument("--n-trials", type=int, default=4)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--experiment", default="skempi_M")
    ap.add_argument("--base-ckpt", default="ckpts/adit_M.ckpt")
    ap.add_argument("--dataset", default="dataset/skempi_stab")
    ap.add_argument("--test-split", default="stab_test")  # 直接在 test split 上调参(用户决定;results 须标注)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--es-patience", type=int, default=100)
    ap.add_argument("--min-k", type=int, default=10)
    ap.add_argument("--work-dir", default=os.path.join(REPO, "reproduce", "retune_trials"))
    args = ap.parse_args()

    storage = make_storage(args.storage)
    sampler = optuna.samplers.TPESampler(seed=1000 + args.worker_id * 97, n_startup_trials=10)
    study = optuna.create_study(study_name=args.study_name, storage=storage,
                                direction="minimize", sampler=sampler, load_if_exists=True)
    print(f"[worker {args.worker_id}] study='{args.study_name}' trials so far={len(study.trials)} "
          f"-> running {args.n_trials} trials (objective=overall RMSE, minimize)", flush=True)
    study.optimize(build_objective(args), n_trials=args.n_trials, catch=(Exception,))
    print(f"[worker {args.worker_id}] done. best RMSE={study.best_value:.4f} params={study.best_params}", flush=True)


if __name__ == "__main__":
    main()
