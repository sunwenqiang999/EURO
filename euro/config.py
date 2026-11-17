"""全局配置：路径、随机种子、数据集与客户端异构参数采样。

所有路径都基于项目根目录动态计算，彻底移除原代码中的 Windows 硬编码路径
(``E:\\pycharm\\EURO\\...``)，保证在 macOS / Linux / Windows 均可运行。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# 路径配置
# --------------------------------------------------------------------------- #
# config.py 位于 <PROJECT_ROOT>/euro/ 之下，向上一级即项目根目录。
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

# 预生成准确率曲线所在目录。
TIMELIST_DIR = os.path.join(PROJECT_ROOT, "timelist")

# 重构后统一的输出目录（与旧目录隔离，避免覆盖历史结果）。
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_new")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures_new")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")


def ensure_dirs(*paths: str) -> None:
    """确保给定目录存在。"""
    for p in paths:
        os.makedirs(p, exist_ok=True)


# --------------------------------------------------------------------------- #
# 数据集配置
# --------------------------------------------------------------------------- #
# 磁盘上真实文件名（注意 FMNIST 首字母大写），统一通过本映射引用，
# 修复原代码 ``fminst`` / ``Fminst`` / ``minst`` 大小写不一致问题。
@dataclass(frozen=True)
class DatasetConfig:
    key: str                 # 内部标识，如 "mnist"
    display: str             # 图例 / 文件名使用的名称
    acc_file: str            # timelist 下的 .npy 文件名
    target_accuracies: tuple # 到达时间柱状图使用的四个目标精度
    acc_threshold: float     # accVstime 折线图的截断阈值
    # 客户端异构参数采样区间 (d, c, a, T_up) 的 (low, high)。
    # mnist/fmnist 使用较小范围；cifar10/cifar100 使用较大范围，
    # 使各数据集累计时延量级与原实验图对齐。
    param_range: dict = None


# 两套客户端异构参数范围。
_SMALL_PARAMS = {  # mnist / fmnist
    "d": (1.0, 2.5), "c": (1.0, 2.5), "a": (1.0, 2.5), "tup": (3.0, 8.0),
}
_LARGE_PARAMS = {  # cifar10 / cifar100
    "d": (1.0, 3.5), "c": (1.0, 3.5), "a": (1.0, 3.5), "tup": (5.0, 15.0),
}


DATASETS: dict[str, DatasetConfig] = {
    "mnist": DatasetConfig(
        key="mnist",
        display="mnist",
        acc_file="minstcnn_trainacc.npy",
        target_accuracies=(96, 97, 98, 99),
        acc_threshold=99.0,
        param_range=_SMALL_PARAMS,
    ),
    "fmnist": DatasetConfig(
        key="fmnist",
        display="fmnist",
        acc_file="Fminstcnn_trainacc.npy",
        target_accuracies=(80, 85, 88, 90),
        acc_threshold=91.5,
        param_range=_SMALL_PARAMS,
    ),
    "cifar10": DatasetConfig(
        key="cifar10",
        display="cifar10",
        acc_file="cifar10_trainacc.npy",
        target_accuracies=(80, 85, 90, 95),
        acc_threshold=97.0,
        param_range=_LARGE_PARAMS,
    ),
    "cifar100": DatasetConfig(
        key="cifar100",
        display="cifar100",
        acc_file="cifar100_trainacc.npy",
        target_accuracies=(60, 70, 80, 90),
        acc_threshold=95.0,
        param_range=_LARGE_PARAMS,
    ),
}


def load_accuracy(dataset_key: str) -> np.ndarray:
    """加载指定数据集的预生成准确率曲线（0-100 百分制，长度 50）。"""
    cfg = DATASETS[dataset_key]
    path = os.path.join(TIMELIST_DIR, cfg.acc_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"准确率文件不存在: {path}")
    return np.asarray(np.load(path), dtype=np.float64)


# --------------------------------------------------------------------------- #
# 仿真全局参数
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    num_clients: int = 20          # 参与客户端数量 K
    num_rounds: int = 50           # 通信轮数 T
    total_budget: float = 10000.0  # 全局预算 B
    sigma: int = 3                 # 本地训练轮数 σ
    seed: int = 42                 # 全局随机种子

    # IRBAI（收敛感知轮间）参数。
    irbai_psi: float = 10.0        # ψ, 控制精度增量对预算的影响强度
    # IRBAI 的收敛判定阈值 Q，默认取数据集目标精度（运行时覆盖）。
    irbai_Q: float = 100.0

    # 客户端异构参数采样区间（均匀分布）。
    # d_i: 数据量相关；c_i: 单样本计算周期；a_i: 能耗系数；T_up: 上行通信时间。
    # 默认使用较小范围（mnist/fmnist）；运行时可用 apply_dataset_params()
    # 按数据集覆盖（cifar 使用较大范围，使累计时延量级对齐原实验图）。
    d_low: float = 1.0
    d_high: float = 2.5
    c_low: float = 1.0
    c_high: float = 2.5
    a_low: float = 1.0
    a_high: float = 2.5
    tup_low: float = 3.0
    tup_high: float = 8.0


def apply_dataset_params(sim: "SimConfig", ds_key: str) -> "SimConfig":
    """按数据集把客户端参数范围写入 SimConfig（就地修改并返回）。

    mnist / fmnist 使用较小范围，cifar10 / cifar100 使用较大范围。
    """
    pr = DATASETS[ds_key].param_range or _SMALL_PARAMS
    sim.d_low, sim.d_high = pr["d"]
    sim.c_low, sim.c_high = pr["c"]
    sim.a_low, sim.a_high = pr["a"]
    sim.tup_low, sim.tup_high = pr["tup"]
    return sim


def set_global_seed(seed: int) -> None:
    """统一设置 random / numpy / torch 随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class ClientParams:
    """一轮中 K 个客户端的异构参数（向量化）。"""

    sigma: int
    d_i: np.ndarray
    c_i: np.ndarray
    a_i: np.ndarray
    T_up: np.ndarray

    def __len__(self) -> int:
        return len(self.d_i)


def sample_clients(sim: SimConfig, rng: np.random.Generator | None = None) -> ClientParams:
    """为一轮采样 K 个异构客户端参数。

    使用传入的 ``rng`` (numpy Generator) 保证可控随机性；若为 None 则使用全局
    numpy 随机状态。
    """
    draw = rng.uniform if rng is not None else np.random.uniform
    k = sim.num_clients
    return ClientParams(
        sigma=sim.sigma,
        d_i=draw(sim.d_low, sim.d_high, size=k),
        c_i=draw(sim.c_low, sim.c_high, size=k),
        a_i=draw(sim.a_low, sim.a_high, size=k),
        T_up=draw(sim.tup_low, sim.tup_high, size=k),
    )
