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
    valid = [t for t in done if t.value is not None and np.isfinite(t.value) and t.value > -1.0]
    print(f"study='{args.study_name}'  total={len(trials)} complete={len(done)} valid={len(valid)} "
          f"(fail/-1={len(done) - len(valid)})")
    if not valid:
        print("没有有效 trial。"); return

    valid.sort(key=lambda t: t.value, reverse=True)
    print(f"\n=== Top {min(args.top, len(valid))} trials (by val per-interface Spearman K>=10) ===")
    hdr = f"{'rank':>4} {'trial':>5} {'perIfaceSp':>10} {'trunc':>6} {'lr':>9} {'drop':>5} {'wd':>7} {'efbs':>3} {'schP':>5} {'n_used':>6} {'ovrP':>6} {'ovrS':>6} {'min':>5}"
    print(hdr)
    for i, t in enumerate(valid[:args.top], 1):
        p = t.params
        ov = t.user_attrs.get("overall", {})
        eff_bs = t.user_attrs.get('effective_batch_size', '?')  # batch_size 不再是搜索维,改读自适应后的实际值
        lr = p.get('lr')
        print(f"{i:>4} {t.number:>5} {t.value:>10.4f} {str(p.get('truncation_size')):>6} "
              f"{(f'{lr:.2e}' if lr is not None else '?'):>9} {str(p.get('dropout')):>5} {str(p.get('weight_decay')):>7} "
              f"{str(eff_bs):>3} {str(p.get('scheduler_patience')):>5} "
              f"{str(t.user_attrs.get('n_used_complexes','?')):>6} "
              f"{ov.get('pearson', float('nan')):>6.3f} {ov.get('spearman', float('nan')):>6.3f} "
              f"{str(t.user_attrs.get('elapsed_min','?')):>5}")

    b = study.best_trial
    print(f"\n=== BEST (trial {b.number}, val per-iface Spearman K>=10 = {b.value:.4f}) ===")
    for k, v in b.params.items():
        print(f"  {k} = {v}")
    ov = b.user_attrs.get("overall", {})
    print(f"  [tune_val overall] P={ov.get('pearson')} S={ov.get('spearman')} RMSE={ov.get('rmse')} "
          f"n_used_complexes={b.user_attrs.get('n_used_complexes')}")
    print("\n建议把上述 best 配置用于 S/M/L 的最终 train+test(在 stab_train -> stab_test 上)。")


if __name__ == "__main__":
    main()
