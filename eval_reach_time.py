"""评估各方法到达目标准确率所需的累计时间。

用法：
    python eval_reach_time.py --src results_new            # 读取已有结果（基线）
    python eval_reach_time.py --run --out results_rl_v2    # 重新跑实验并保存、打印

到达时间 = 该方法逐轮时延累加到「准确率曲线首次 >= 目标精度」那一轮为止，
定义与 euro/plotting.py 的 plot_reach_time_bar 一致。从未达标记为 inf。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from euro.config import (DATASETS, SimConfig, apply_dataset_params, derive_seed,
                         load_accuracy, set_global_seed)
from euro.plotting import first_reach_index
from euro.simulator import (run_inter_experiment, run_intra_experiment,
                            run_joint_experiment)

METHODS = ["ours", "euro_afl", "fair", "average", "random"]


def reach_times(accuracy, times, targets):
    cum = np.cumsum(np.asarray(times, dtype=np.float64))
    out = []
    for tg in targets:
        idx = first_reach_index(accuracy, tg)
        out.append(float(cum[idx]) if idx is not None else float("inf"))
    return out


def report(ds, exp, times_by_method, accuracy, targets):
    print(f"\n[{ds} | {exp}]  目标精度 = {list(targets)}")
    print(f"  {'方法':<10}" + "".join(f"{'到达%g%%' % t:>12}" for t in targets) + f"{'50轮总时延':>12}")
    for m in METHODS:
        if m not in times_by_method:
            continue
        rt = reach_times(accuracy, times_by_method[m], targets)
        total = float(np.sum(times_by_method[m]))
        print(f"  {m:<10}" + "".join(f"{v:>12.1f}" for v in rt) + f"{total:>12.1f}")


def from_dir(src, datasets):
    for ds in datasets:
        cfg = DATASETS[ds]
        acc = load_accuracy(ds)
        for exp in ("intra", "inter", "joint"):
            d = os.path.join(src, cfg.display, exp)
            if not os.path.isdir(d):
                continue
            tm = {f.split("_time_list")[0]: np.load(os.path.join(d, f))
                  for f in os.listdir(d) if f.endswith("_time_list.npy")}
            report(ds, exp, tm, acc, cfg.target_accuracies)


def run_and_save(out, datasets, seed, budget):
    set_global_seed(seed)
    torch.set_num_threads(1)
    sim0 = SimConfig(seed=seed, total_budget=budget)
    for ds in datasets:
        cfg = DATASETS[ds]
        acc = load_accuracy(ds)
        target = float(cfg.target_accuracies[-1])
        for exp, fn in (("intra", run_intra_experiment),
                        ("inter", run_inter_experiment),
                        ("joint", run_joint_experiment)):
            s = apply_dataset_params(SimConfig(**sim0.__dict__), ds)
            s.seed = derive_seed(seed, ds)
            sd = derive_seed(seed, ds, exp)
            set_global_seed(sd)
            tm = fn(acc, s, target, np.random.default_rng(sd))
            d = os.path.join(out, cfg.display, exp)
            os.makedirs(d, exist_ok=True)
            for m, series in tm.items():
                np.save(os.path.join(d, f"{m}_time_list.npy"), np.asarray(series))
            report(ds, exp, tm, acc, cfg.target_accuracies)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--src", default="results_new")
    p.add_argument("--run", action="store_true")
    p.add_argument("--out", default="results_rl_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget", type=float, default=10000.0)
    a = p.parse_args()
    if a.run:
        run_and_save(a.out, a.datasets, a.seed, a.budget)
    else:
        from_dir(a.src, a.datasets)


if __name__ == "__main__":
    main()
