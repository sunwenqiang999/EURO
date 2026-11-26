"""逐轮对比表：每一轮各方法的轮间预算与该轮时延（轮间实验）。

用法：
    python show_round_table.py                       # 四个数据集
    python show_round_table.py --datasets cifar100
输出：results_rl_v2/<数据集>/inter_round_table.csv 与 results_rl_v2/inter_round_tables.md
参数与 eval_reach_time.py 完全一致，因此时延与之前的结果逐位相同。
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from euro.config import (DATASETS, SimConfig, apply_dataset_params, derive_seed,
                         load_accuracy, set_global_seed)
from euro.simulator import run_inter_experiment, run_joint_experiment

# inter：仅轮间策略不同（轮内统一用解析 IRBSB）；joint：每种方法用自己完整的「轮间+轮内」。
CONFIG = {
    "inter": (run_inter_experiment, ["ours", "euro_afl", "average", "random"],
              {"ours": "IRBAI(ours)", "euro_afl": "InterBMA(RL)",
               "average": "FedAvg", "random": "Random"}),
    "joint": (run_joint_experiment, ["ours", "euro_afl", "fair", "average", "random"],
              {"ours": "IRBAI+IRBSB(ours)", "euro_afl": "InterBMA+IntraBMA(RL)",
               "fair": "均分+Fair", "average": "均分+均分(FedAvg)", "random": "随机+随机"}),
}


def build(ds, seed, budget, exp):
    cfg = DATASETS[ds]
    acc = load_accuracy(ds)
    s = apply_dataset_params(SimConfig(seed=seed, total_budget=budget), ds)
    s.seed = derive_seed(seed, ds)
    sd = derive_seed(seed, ds, exp)
    set_global_seed(sd)
    bud = {}
    times = CONFIG[exp][0](acc, s, float(cfg.target_accuracies[-1]),
                                 np.random.default_rng(sd), budgets_out=bud)
    return cfg, acc, bud, times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget", type=float, default=10000.0)
    p.add_argument("--out", default="results_rl_v2")
    p.add_argument("--exp", choices=list(CONFIG), default="inter")
    a = p.parse_args()
    torch.set_num_threads(1)
    md = []
    ORDER, NAME = CONFIG[a.exp][1], CONFIG[a.exp][2]
    for ds in a.datasets:
        cfg, acc, bud, times = build(ds, a.seed, a.budget, a.exp)
        T = len(acc)
        header = (["轮次", "准确率%"] + [f"预算_{NAME[m]}" for m in ORDER]
                  + [f"时间_{NAME[m]}" for m in ORDER])
        rows = []
        for r in range(T):
            rows.append([r + 1, f"{acc[r]:.2f}"]
                        + [f"{bud[m][r]:.1f}" for m in ORDER]
                        + [f"{times[m][r]:.2f}" for m in ORDER])
        total = (["合计", ""] + [f"{sum(bud[m]):.0f}" for m in ORDER]
                 + [f"{sum(times[m]):.1f}" for m in ORDER])
        d = os.path.join(a.out, cfg.display)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{a.exp}_round_table.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows); w.writerow(total)
        md.append(f"## {cfg.display}（总预算 {a.budget:g}，目标精度 {list(cfg.target_accuracies)}）\n")
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        for r_ in rows + [total]:
            md.append("| " + " | ".join(str(x) for x in r_) + " |")
        md.append("")
    path = os.path.join(a.out, f"{a.exp}_round_tables.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"已写入 {path} 及各数据集 {a.exp}_round_table.csv")


if __name__ == "__main__":
    main()
