#!/usr/bin/env python
"""
optuna_retune.py  ——  adit-retune-hyperparams 的并行 Optuna worker

在 **stab_train 内部的 homology-aware 留出验证集(tune_val)** 上,为 ADiT-M 搜索超参;
目标 = val per-interface Spearman(K>=10,maximize)。test(stab_test)全程不碰。

每个 worker = 1 个 SLURM a100 任务,跑 --n-trials 个 trial;多个 worker 通过共享
JournalStorage(放 /ibex 共享盘,适合并行写)协作同一个 study(TPE)。

每个 trial:
  1) train.sh fine-tune ADiT-M(从 ckpts/adit_M.ckpt 起),tune_train 训练 + 内部随机 val 早停;
  2) test.sh 在 tune_val 上推理,落 pkl(accession, pred, target);
  3) 读 pkl,按 complex(前 3 个 '_' token)分组算 per-interface Spearman(K>=10)的均值,作为目标返回。
失败(如大 truncation OOM)→ 返回 -1.0 并记 fail 原因,让 TPE 自行避开。

搜索空间(范围/步长见下方 suggest_*):truncation_size / lr / dropout / weight_decay /
scheduler.patience。epochs 用早停自适应(max_epochs 封顶)。batch_size 不进搜索:每个 trial
从 8 起遇 CUDA OOM 自动减半(8→4→2→1)取最大可行值(因 trunc×batch 显存耦合,trunc256×8 在 80G a100 也 OOM),
记入 user_attr['effective_batch_size']。

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


def build_objective(args):
    def objective(trial):
        ts = trial.suggest_categorical("truncation_size", [50, 100, 150, 200, 256])
        lr = trial.suggest_float("lr", 1e-6, 3e-4, log=True)
        dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
        wd = trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-2])
        sched_pat = trial.suggest_categorical("scheduler_patience", [5, 10])

        tag = f"retuneM_w{args.worker_id}_t{trial.number}"
        trial_pkl = os.path.join(args.work_dir, f"{tag}.pkl")
        log_path = os.path.join(args.work_dir, f"{tag}.log")
        os.makedirs(args.work_dir, exist_ok=True)

        common = [
            "++trainer.devices=1", "++trainer.strategy=auto",
            f"++data.dataset.path_to_dataset={args.dataset}",
            f"++data.dataset.test_split={args.test_split}",
            f"++data.dataset.truncation_size={ts}",
            f"++model.net.dropout={dropout}",
        ]
        # 显存自适应:从 batch=8 起,遇 CUDA OOM 自动减半(8→4→2→1),保证每个 truncation 都训得起来
        t0 = time.time()
        eff_bs = None
        for bs in (8, 4, 2, 1):
            attempt_log = os.path.join(args.work_dir, f"{tag}_bs{bs}.log")
            if os.path.exists(attempt_log):
                os.remove(attempt_log)
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
            rc = run(train_cmd, attempt_log)
            if rc == 0:
                eff_bs = bs
                break
            try:
                with open(attempt_log, "rb") as lf:
                    txt = lf.read().decode("utf-8", "ignore")
                oom = ("OutOfMemoryError" in txt) or ("CUDA out of memory" in txt)
            except Exception:
                oom = False
            if not oom:
                trial.set_user_attr("fail", f"train rc={rc} (non-OOM, bs={bs})")
                return -1.0
            trial.set_user_attr(f"oom_at_bs{bs}", 1)  # OOM -> 试更小 batch
        if eff_bs is None:
            trial.set_user_attr("fail", "OOM even at batch=1")
            return -1.0
        trial.set_user_attr("effective_batch_size", eff_bs)

        ckpts = sorted(glob.glob(os.path.join(REPO, "outputs", "*", f"{tag}_*", "checkpoints", "last.ckpt")),
                       key=os.path.getmtime)
        if not ckpts:
            trial.set_user_attr("fail", "no last.ckpt")
            return -1.0
        last_ckpt = ckpts[-1]

        test_cmd = ["bash", "test.sh", f"experiment={args.experiment}"] + common + [
            "++data.batch_size=1",
            f"++model.save_file={trial_pkl}",
            f"ckpt_path={last_ckpt}",
        ]
        rc = run(test_cmd, os.path.join(args.work_dir, f"{tag}_test.log"))
        if rc != 0 or not os.path.exists(trial_pkl):
            trial.set_user_attr("fail", f"test rc={rc}")
            return -1.0

        mean_per, n_used, overall = per_interface_spearman(trial_pkl, min_k=args.min_k)
        trial.set_user_attr("elapsed_min", round((time.time() - t0) / 60, 1))
        trial.set_user_attr("n_used_complexes", n_used)
        trial.set_user_attr("overall", overall)
        trial.set_user_attr("last_ckpt", last_ckpt)
        if not np.isfinite(mean_per):
            trial.set_user_attr("fail", f"per-iface nan (n_used={n_used})")
            return -1.0
        print(f"[trial {trial.number}] ts={ts} lr={lr:.2e} do={dropout} wd={wd} bs={eff_bs}(auto) "
              f"schedpat={sched_pat} -> per-iface Sp(K>={args.min_k})={mean_per:.4f} "
              f"(n_used={n_used}, overall P={overall['pearson']:.3f}/S={overall['spearman']:.3f})", flush=True)
        return mean_per
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--storage", required=True, help="JournalStorage 文件路径(/ibex 共享盘)")
    ap.add_argument("--n-trials", type=int, default=4)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--experiment", default="skempi_M")
    ap.add_argument("--base-ckpt", default="ckpts/adit_M.ckpt")
    ap.add_argument("--dataset", default="dataset/skempi_stab_tune")
    ap.add_argument("--test-split", default="tune_val")
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--es-patience", type=int, default=12)
    ap.add_argument("--min-k", type=int, default=10)
    ap.add_argument("--work-dir", default=os.path.join(REPO, "reproduce", "retune_trials"))
    args = ap.parse_args()

    storage = make_storage(args.storage)
    sampler = optuna.samplers.TPESampler(seed=1000 + args.worker_id * 97, n_startup_trials=10)
    study = optuna.create_study(study_name=args.study_name, storage=storage,
                                direction="maximize", sampler=sampler, load_if_exists=True)
    print(f"[worker {args.worker_id}] study='{args.study_name}' trials so far={len(study.trials)} "
          f"-> running {args.n_trials} trials", flush=True)
    study.optimize(build_objective(args), n_trials=args.n_trials, catch=(Exception,))
    print(f"[worker {args.worker_id}] done. best={study.best_value:.4f} params={study.best_params}", flush=True)


if __name__ == "__main__":
    main()
