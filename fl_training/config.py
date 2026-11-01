"""FedAvg 基线训练的默认配置（独立于原有仿真代码）。"""
import os
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
