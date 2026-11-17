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

# 方法 -> 颜色（与原实验图一致：蓝/橙/绿/红/紫）。
METHOD_COLOR = {
    "ours": "b",
    "euro_afl": "orange",
    "random": "g",
    "average": "r",
    "fair": "purple",
}

# 图例文案按实验类型区分，与原实验图保持一致：
#   joint —— 整体方法名；intra —— 固定轮间(IRBAI)、变轮内；inter —— 固定轮内(IRBSB)、变轮间。
JOINT_LABEL = {
    "ours": "GBSFL",
    "euro_afl": "EURO-AFL",
    "random": "Random",
    "average": "FedAvg",
    "fair": "Fair",
}
INTRA_LABEL = {
    "ours": "IRBAI+IRBSB",
    "euro_afl": "IRBAI+IntraBMA",
    "random": "IRBAI+Random",
    "average": "IRBAI+FedAvg",
    "fair": "IRBAI+Fair",
}
INTER_LABEL = {
    "ours": "IRBAI+IRBSB",
    "euro_afl": "InterBMA+IRBSB",
    "random": "Random+IRBSB",
    "average": "FedAvg+IRBSB",
}

# 绘图顺序（保证图例稳定）。
INTRA_ORDER = ["ours", "euro_afl", "random", "average", "fair"]
INTER_ORDER = ["ours", "euro_afl", "random", "average"]
JOINT_ORDER = ["ours", "euro_afl", "random", "average", "fair"]

# 原实验图的通用样式常量。
LINE_FIGSIZE = (10, 6)   # 折线图
BAR_FIGSIZE = (6.4, 4.8) # 柱状图（matplotlib 默认比例）
BAR_WIDTH = 0.15
SAVE_DPI = 300


def first_reach_index(accuracy: Sequence[float], target: float) -> Optional[int]:
    """返回首次达到 target 的轮次索引；从未达标返回 None。"""
    acc = np.asarray(accuracy, dtype=np.float64)
    reached = np.nonzero(acc >= target)[0]
    return int(reached[0]) if reached.size > 0 else None


def _cumulative_time(times: Sequence[float], upto_index: int) -> float:
    return float(np.sum(np.asarray(times, dtype=np.float64)[: upto_index + 1]))


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.eps"), format="eps", dpi=SAVE_DPI,
                bbox_inches="tight")
    plt.close(fig)


def plot_acc_vs_time(accuracy: Sequence[float],
                     method_times: Dict[str, List[float]],
                     order: Sequence[str],
                     threshold: float,
                     out_dir: str, name: str,
                     title: str = "",
                     full_curve: bool = False,
                     labels: Optional[Dict[str, str]] = None,
                     legend_loc: str = "best") -> None:
    """精度随累计时间变化折线图。

    Args:
        full_curve: True 时画完整 50 轮（横轴为全程累计时间，与旧图量级一致）；
                    False 时按 threshold 首次达标轮次截断。
        labels: 方法 -> 图例文案，默认用 JOINT_LABEL。
        legend_loc: 图例位置，与原实验图一致默认自动选择。
    """
    acc = np.asarray(accuracy, dtype=np.float64)
    label_map = labels or JOINT_LABEL
    if full_curve:
        end = len(acc) - 1
    else:
        idx = first_reach_index(acc, threshold)
        end = len(acc) - 1 if idx is None else idx  # 未达标则画完整曲线

    fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
    for m in order:
        if m not in method_times:
            continue
        cum = np.cumsum(np.asarray(method_times[m], dtype=np.float64))
        ax.plot(cum[: end + 1], acc[: end + 1],
                label=label_map.get(m, m), color=METHOD_COLOR[m])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy (%)")
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=legend_loc)
    fig.tight_layout()
    _save(fig, out_dir, name)


def plot_reach_time_bar(accuracy: Sequence[float],
                        method_times: Dict[str, List[float]],
                        order: Sequence[str],
                        target_accuracies: Sequence[float],
                        out_dir: str, name: str,
                        title: str = "",
                        labels: Optional[Dict[str, str]] = None) -> None:
    """到达各目标精度所需累计时间的分组柱状图。

    从未达到某目标精度的方法，该柱子记为 0（并在图例/日志中可识别）。
    """
    acc = np.asarray(accuracy, dtype=np.float64)
    label_map = labels or JOINT_LABEL
    methods = [m for m in order if m in method_times]
    n_targets = len(target_accuracies)

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    x = np.arange(n_targets)
    for i, m in enumerate(methods):
        heights = []
        for target in target_accuracies:
            idx = first_reach_index(acc, target)
            heights.append(0.0 if idx is None else _cumulative_time(method_times[m], idx))
        ax.bar(x + i * BAR_WIDTH, heights, BAR_WIDTH,
               label=label_map.get(m, m), color=METHOD_COLOR[m])

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Time (s)")
    if title:
        ax.set_title(title)
    # 与原实验图一致：刻度落在第 3 个柱子（索引 2）的位置。
    ax.set_xticks(x + BAR_WIDTH * 2)
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
    fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
    for i, (key, times) in enumerate(series.items()):
        cum = np.cumsum(np.asarray(times, dtype=np.float64))
        ax.plot(cum[: end + 1], acc[: end + 1],
                label=label_fmt.format(key), color=colors[i % len(colors)])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy (%)")
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, name)
