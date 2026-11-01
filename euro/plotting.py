"""绘图：精度-累计时间折线图 + 到达目标精度柱状图。

关键修复（相对原代码）：
    - ``first_reach_index`` 正确处理「从未达标」的情况，不再被 ``np.argmax``
      误判为第 0 轮达标。
    - 统一相对输出路径，移除 Windows 硬编码路径。
    - 每张图独立 ``figure``，避免多图叠加。
    - eps 嵌入 TrueType 字体，文件名拼写统一。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也可保存图片
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 方法 -> (图例标签, 颜色)
METHOD_STYLE = {
    "ours": ("GBSFL (Ours)", "b"),
    "euro_afl": ("EURO-AFL", "orange"),
    "random": ("Random", "g"),
    "average": ("FedAvg", "r"),
    "fair": ("Fair", "purple"),
}

# 绘图顺序（保证图例稳定）。
INTRA_ORDER = ["ours", "euro_afl", "random", "average", "fair"]
INTER_ORDER = ["ours", "euro_afl", "random", "average"]


def first_reach_index(accuracy: Sequence[float], target: float) -> Optional[int]:
    """返回首次达到 target 的轮次索引；从未达标返回 None。"""
    acc = np.asarray(accuracy, dtype=np.float64)
    reached = np.nonzero(acc >= target)[0]
    return int(reached[0]) if reached.size > 0 else None


def _cumulative_time(times: Sequence[float], upto_index: int) -> float:
    return float(np.sum(np.asarray(times, dtype=np.float64)[: upto_index + 1]))


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.eps"), bbox_inches="tight")
    plt.close(fig)


def plot_acc_vs_time(accuracy: Sequence[float],
                     method_times: Dict[str, List[float]],
                     order: Sequence[str],
                     threshold: float,
                     out_dir: str, name: str,
                     title: str = "") -> None:
    """精度随累计时间变化折线图（按阈值截断）。"""
    acc = np.asarray(accuracy, dtype=np.float64)
    idx = first_reach_index(acc, threshold)
    end = len(acc) - 1 if idx is None else idx  # 未达标则画完整曲线

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in order:
        if m not in method_times:
            continue
        label, color = METHOD_STYLE[m]
        cum = np.cumsum(np.asarray(method_times[m], dtype=np.float64))
        ax.plot(cum[: end + 1], acc[: end + 1], label=label, color=color, marker="o", markersize=3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy (%)")
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, name)


def plot_reach_time_bar(accuracy: Sequence[float],
                        method_times: Dict[str, List[float]],
                        order: Sequence[str],
                        target_accuracies: Sequence[float],
                        out_dir: str, name: str,
                        title: str = "") -> None:
    """到达各目标精度所需累计时间的分组柱状图。

    从未达到某目标精度的方法，该柱子记为 0（并在图例/日志中可识别）。
    """
    acc = np.asarray(accuracy, dtype=np.float64)
    methods = [m for m in order if m in method_times]
    n_targets = len(target_accuracies)
    bar_width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_targets)
    for i, m in enumerate(methods):
        label, color = METHOD_STYLE[m]
        heights = []
        for target in target_accuracies:
            idx = first_reach_index(acc, target)
            heights.append(0.0 if idx is None else _cumulative_time(method_times[m], idx))
        ax.bar(x + i * bar_width, heights, bar_width, label=label, color=color)

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Time (s)")
    if title:
        ax.set_title(title)
    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels([str(t) for t in target_accuracies])
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, name)


def plot_sensitivity(accuracy: Sequence[float],
                     series: Dict,
                     label_fmt: str,
                     threshold: float,
                     out_dir: str, name: str,
                     title: str = "") -> None:
    """敏感性折线图：series 为 {参数值: 逐轮时延}。"""
    acc = np.asarray(accuracy, dtype=np.float64)
    idx = first_reach_index(acc, threshold)
    end = len(acc) - 1 if idx is None else idx

    colors = ["orange", "g", "r", "purple", "b", "brown"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (key, times) in enumerate(series.items()):
        cum = np.cumsum(np.asarray(times, dtype=np.float64))
        ax.plot(cum[: end + 1], acc[: end + 1],
                label=label_fmt.format(key), color=colors[i % len(colors)], marker="o", markersize=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy (%)")
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, name)
