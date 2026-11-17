"""EURO 实验主入口。

用法：
    python run_experiments.py                 # 运行全部数据集的轮内+轮间实验
    python run_experiments.py --datasets mnist fmnist
    python run_experiments.py --only intra    # 只跑轮内
    python run_experiments.py --sensitivity   # 额外跑预算/ψ 敏感性

产物：
    results_new/<dataset>/*.npy    各方法逐轮时延
    figures_new/intra/<dataset>_*  轮内对比图（折线 + 柱状）
    figures_new/inter/<dataset>_*  轮间对比图
    figures_new/sensitivity/*      敏感性图
    results_new/models/*.pth       DQN 模型（如启用保存）
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from euro import plotting
from euro.config import (DATASETS, FIGURES_DIR, RESULTS_DIR, SimConfig,
                         apply_dataset_params, ensure_dirs, load_accuracy,
                         set_global_seed)
from euro.simulator import (run_budget_sensitivity, run_inter_experiment,
                            run_intra_experiment, run_joint_experiment,
                            run_psi_sensitivity)


def _save_times(out_dir: str, times: dict) -> None:
    ensure_dirs(out_dir)
    for method, series in times.items():
        np.save(os.path.join(out_dir, f"{method}_time_list.npy"), np.asarray(series))


def run_dataset(ds_key: str, sim: SimConfig, run_intra: bool, run_inter: bool,
                run_joint: bool = False, full_curve: bool = False) -> None:
    cfg = DATASETS[ds_key]
    accuracy = load_accuracy(ds_key)
    # 目标精度取该数据集最高目标。
    target = float(cfg.target_accuracies[-1])

    def ds_sim() -> SimConfig:
        # 拷贝配置并按数据集注入客户端参数范围（mnist/fmnist 小、cifar 大）。
        return apply_dataset_params(SimConfig(**sim.__dict__), ds_key)

    pr = cfg.param_range
    print(f"\n=== 数据集 {cfg.display} | target_acc={target} | threshold={cfg.acc_threshold} "
          f"| params d/c/a={pr['d']} T_up={pr['tup']} ===")

    if run_intra:
        # 每个实验用独立 Generator，保证各方法共享同轮客户端且可复现。
        rng = np.random.default_rng(sim.seed)
        times = run_intra_experiment(accuracy, ds_sim(), target, rng)
        _save_times(os.path.join(RESULTS_DIR, cfg.display, "intra"), times)
        intra_fig = os.path.join(FIGURES_DIR, "intra")
        plotting.plot_acc_vs_time(accuracy, times, plotting.INTRA_ORDER,
                                  cfg.acc_threshold, intra_fig,
                                  f"{cfg.display}_accVstime",
                                  full_curve=full_curve,
                                  labels=plotting.INTRA_LABEL,
                                  legend_loc="lower right")
        plotting.plot_reach_time_bar(accuracy, times, plotting.INTRA_ORDER,
                                     cfg.target_accuracies, intra_fig,
                                     f"{cfg.display}_reachtime",
                                     labels=plotting.INTRA_LABEL)
        print(f"  [intra] 轮内对比图已生成 -> {intra_fig}")

    if run_inter:
        rng = np.random.default_rng(sim.seed + 1)
        times = run_inter_experiment(accuracy, ds_sim(), target, rng)
        _save_times(os.path.join(RESULTS_DIR, cfg.display, "inter"), times)
        inter_fig = os.path.join(FIGURES_DIR, "inter")
        plotting.plot_acc_vs_time(accuracy, times, plotting.INTER_ORDER,
                                  cfg.acc_threshold, inter_fig,
                                  f"{cfg.display}_accVstime",
                                  full_curve=full_curve,
                                  labels=plotting.INTER_LABEL,
                                  legend_loc="upper left")
        plotting.plot_reach_time_bar(accuracy, times, plotting.INTER_ORDER,
                                     cfg.target_accuracies, inter_fig,
                                     f"{cfg.display}_reachtime",
                                     labels=plotting.INTER_LABEL)
        print(f"  [inter] 轮间对比图已生成 -> {inter_fig}")

    if run_joint:
        rng = np.random.default_rng(sim.seed + 4)
        times = run_joint_experiment(accuracy, ds_sim(), target, rng)
        _save_times(os.path.join(RESULTS_DIR, cfg.display, "joint"), times)
        joint_fig = os.path.join(FIGURES_DIR, "joint")
        plotting.plot_acc_vs_time(accuracy, times, plotting.JOINT_ORDER,
                                  cfg.acc_threshold, joint_fig,
                                  f"{cfg.display}_accVstime",
                                  full_curve=full_curve,
                                  labels=plotting.JOINT_LABEL,
                                  legend_loc="lower right")
        plotting.plot_reach_time_bar(accuracy, times, plotting.JOINT_ORDER,
                                     cfg.target_accuracies, joint_fig,
                                     f"{cfg.display}_reachtime",
                                     labels=plotting.JOINT_LABEL)
        print(f"  [joint] 轮间+轮内联合对比图已生成 -> {joint_fig}")


def run_sensitivity(ds_key: str, sim: SimConfig) -> None:
    cfg = DATASETS[ds_key]
    accuracy = load_accuracy(ds_key)
    target = float(cfg.target_accuracies[-1])
    sens_fig = os.path.join(FIGURES_DIR, "sensitivity")

    rng = np.random.default_rng(sim.seed + 2)
    budgets = [5000.0, 10000.0, 20000.0, 30000.0]
    bud_series = run_budget_sensitivity(
        accuracy, apply_dataset_params(SimConfig(**sim.__dict__), ds_key), target, budgets, rng)
    # 与原实验图一致：图例以千为单位标注（B=5 / B=10 / B=20 / B=30）。
    bud_series = {b / 1000.0: s for b, s in bud_series.items()}
    plotting.plot_sensitivity(accuracy, bud_series, "B={:.0f}", cfg.acc_threshold,
                              sens_fig, f"{cfg.display}_budget")

    rng = np.random.default_rng(sim.seed + 3)
    psis = [1.0, 10.0, 100.0, 500.0, 1000.0]
    psi_series = run_psi_sensitivity(
        accuracy, apply_dataset_params(SimConfig(**sim.__dict__), ds_key), target, psis, rng)
    plotting.plot_sensitivity(accuracy, psi_series, r"$\psi$={:.0f}", cfg.acc_threshold,
                              sens_fig, f"{cfg.display}_psi")
    print(f"  [sensitivity] 敏感性图已生成 -> {sens_fig}")


def main():
    parser = argparse.ArgumentParser(description="EURO 两阶段联邦学习预算分配实验")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    parser.add_argument("--only", choices=["intra", "inter", "joint", "both", "all"],
                        default="all",
                        help="intra/inter/joint 单独运行；both=轮内+轮间；all=三者全部")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=float, default=10000.0)
    parser.add_argument("--full-curve", action="store_true",
                        help="折线图画完整 50 轮（横轴为全程累计时间），而非按阈值截断")
    args = parser.parse_args()

    ensure_dirs(RESULTS_DIR, FIGURES_DIR)
    set_global_seed(args.seed)

    sim = SimConfig(seed=args.seed, total_budget=args.budget)
    run_intra = args.only in ("intra", "both", "all")
    run_inter = args.only in ("inter", "both", "all")
    run_joint = args.only in ("joint", "all")

    for ds in args.datasets:
        run_dataset(ds, sim, run_intra, run_inter, run_joint, full_curve=args.full_curve)
        if args.sensitivity:
            run_sensitivity(ds, sim)

    print("\n全部实验完成。")


if __name__ == "__main__":
    main()
