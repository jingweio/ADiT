#!/usr/bin/env python
"""
optuna_report.py  ——  读 adit-retune-hyperparams 的 Optuna study,汇报最优/全部 trial。
用法:
  python reproduce/optuna_report.py --study-name adit_M_retune \
      --storage /ibex/.../optuna_aditM.journal [--top 10]
"""
import argparse
import numpy as np
import optuna
from optuna.trial import TrialState


def make_storage(path):
    from optuna.storages import JournalStorage
    try:
        from optuna.storages.journal import JournalFileBackend as Backend
    except Exception:
        from optuna.storages import JournalFileStorage as Backend
    return JournalStorage(Backend(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--storage", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=make_storage(args.storage))
    trials = study.trials
    done = [t for t in trials if t.state == TrialState.COMPLETE]
    # 目标 = overall RMSE(minimize);失败 trial 返回 FAIL_RMSE=1e6,用 <1e5 过滤
    valid = [t for t in done if t.value is not None and np.isfinite(t.value) and t.value < 1e5]
    print(f"study='{args.study_name}'  total={len(trials)} complete={len(done)} valid={len(valid)} "
          f"(fail={len(done) - len(valid)})  [objective=overall RMSE, lower=better]")
    if not valid:
        print("没有有效 trial。"); return

    valid.sort(key=lambda t: t.value)  # RMSE 升序,最小最好
    print(f"\n=== Top {min(args.top, len(valid))} trials (by overall RMSE, lower=better) ===")
    hdr = f"{'rank':>4} {'trial':>5} {'RMSE':>8} {'trunc':>6} {'lr':>9} {'drop':>5} {'wd':>8} {'bs':>3} {'schP':>5} {'perIfSp':>7} {'ovrP':>6} {'min':>6}"
    print(hdr)
    for i, t in enumerate(valid[:args.top], 1):
        p = t.params
        ov = t.user_attrs.get("overall", {})
        lr = p.get('lr'); wd = p.get('weight_decay')
        print(f"{i:>4} {t.number:>5} {t.value:>8.4f} {str(p.get('truncation_size')):>6} "
              f"{(f'{lr:.0e}' if lr is not None else '?'):>9} {str(p.get('dropout')):>5} "
              f"{(f'{wd:.0e}' if wd is not None else '?'):>8} "
              f"{str(t.user_attrs.get('effective_batch_size','?')):>3} {str(p.get('scheduler_patience')):>5} "
              f"{t.user_attrs.get('per_iface_spearman', float('nan')):>7.3f} "
              f"{ov.get('pearson', float('nan')):>6.3f} {str(t.user_attrs.get('elapsed_min','?')):>6}")

    b = study.best_trial
    print(f"\n=== BEST (trial {b.number}, overall RMSE = {b.value:.4f}) ===")
    for k, v in b.params.items():
        print(f"  {k} = {v}")
    ov = b.user_attrs.get("overall", {})
    print(f"  [stab_test] RMSE={ov.get('rmse')} P={ov.get('pearson')} S={ov.get('spearman')} "
          f"per-iface Sp(K>=10)={b.user_attrs.get('per_iface_spearman')}")
    print("\n⚠ 本轮在 stab_test 上调参(test-tuned 上界);最终 S/M/L 用此配置出数时须如此标注。")


if __name__ == "__main__":
    main()
