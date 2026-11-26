"""FedAvg 基线训练的默认配置（独立于原有仿真代码）。"""
import os
import random

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据与模型
DATASET = "mnist"  # 可选: mnist / fmnist / cifar10 / cifar100
MODEL = "simplecnn"  # 可选: simplecnn / lenet5
DATA_ROOT = os.path.join(BASE_DIR, "data")

# 联邦训练超参数（与原仿真器的设定对齐）
NUM_CLIENTS = 20   # 客户端数量，对应仿真器 num_clients=20
NUM_ROUNDS = 50    # 全局通信轮数，对应仿真器 num_rounds=50
LOCAL_EPOCHS = 3   # 本地 epoch，对应仿真器 sigma=3（本地训练轮数）
BATCH_SIZE = 32
LR = 0.01          # 本地 SGD 学习率
MOMENTUM = 0.9

SEED = 42

# 输出目录（准确率曲线 .npy 与最终模型 .pth）
OUT_DIR = os.path.join(BASE_DIR, "results")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def set_seed(seed: int = SEED) -> None:
    """统一设置 random / numpy / torch 随机种子并开启确定性算法。

    覆盖范围：
        - ``PYTHONHASHSEED``：消除 set/dict 遍历顺序的隐式随机性；
        - Python ``random``、``numpy``、torch CPU / CUDA / MPS；
        - cuDNN 关闭算法自动搜索（``benchmark=False``）并开启 ``deterministic``，
          否则同一卷积在不同次运行可能选到不同算法，产生浮点差异。

    注意：在 CUDA / MPS 后端上，部分算子（如 ``scatter_add``、某些池化反向）
    本身不保证位级确定；若要求严格逐位复现，请用 ``DEVICE=cpu`` 运行。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed"):
        try:
            mps.manual_seed(seed)
        except Exception:  # pragma: no cover - 部分 torch 版本无 MPS 运行时
            pass
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - 某些后端不支持
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``num_workers>0`` 时的 worker 初始化函数。

    每个 worker 进程有独立的 ``random`` / ``numpy`` 状态，必须显式播种，
    否则多进程加载会引入不可复现的随机性。
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
