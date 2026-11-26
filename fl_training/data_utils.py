"""数据集下载与 IID 平均划分。"""
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ArrayImageDataset(Dataset):
    """以内存中的 uint8 数组为后端的图像数据集。

    行为与 ``torchvision.datasets.CIFAR10`` 完全一致：``__getitem__`` 先把
    ``(H, W, C)`` 的 uint8 数组转成 PIL Image，再套用 ``transform``，因此
    ToTensor / Normalize 等变换的数值结果与 torchvision 原生实现相同。

    用于加载 ``prepare_cifar.py`` 生成的 npz —— CIFAR 官方源站在国内带宽极低
    （实测约 80 KB/s），改由 HuggingFace 镜像取 parquet 并在本地解码成数组。

    Args:
        data: 形状 ``(N, H, W, C)`` 的 uint8 数组。
        targets: 长度 N 的整型标签数组。
        transform: 作用于 PIL Image 的变换。
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        if len(data) != len(targets):
            raise ValueError(f"样本数不一致: data={len(data)}, targets={len(targets)}")
        self.data = data
        self.targets = [int(t) for t in targets]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]


def _load_cifar_from_npz(dataset: str, data_root: str, transform):
    """从 ``<data_root>/<dataset>_arrays.npz`` 构建 (train_set, test_set)。"""
    path = os.path.join(data_root, f"{dataset}_arrays.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"缺少 {path}；请先运行 python fl_training/prepare_cifar.py 生成"
        )
    with np.load(path) as z:
        train_set = ArrayImageDataset(z["train_data"], z["train_labels"], transform)
        test_set = ArrayImageDataset(z["test_data"], z["test_labels"], transform)
    return train_set, test_set


def load_dataset(dataset: str, data_root: str):
    """下载（或从本地 npz 加载）并返回 (train_set, test_set)。"""
    if dataset in ("mnist", "fmnist"):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        train_set = cls(data_root, train=True, download=True, transform=tf)
        test_set = cls(data_root, train=False, download=True, transform=tf)
    elif dataset in ("cifar10", "cifar100"):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        # 优先使用 torchvision 原生目录（若已手工下载 tar.gz 并解压）；
        # 否则回退到 prepare_cifar.py 生成的 npz 数组。
        native_dir = os.path.join(
            data_root,
            "cifar-10-batches-py" if dataset == "cifar10" else "cifar-100-python",
        )
        if os.path.isdir(native_dir):
            cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
            train_set = cls(data_root, train=True, download=False, transform=tf)
            test_set = cls(data_root, train=False, download=False, transform=tf)
        else:
            train_set, test_set = _load_cifar_from_npz(dataset, data_root, tf)
    else:
        raise ValueError(f"未知数据集: {dataset}")
    return train_set, test_set


def split_data_iid(train_set, num_clients: int, seed: int = 42):
    """将训练集随机打乱后【平均】划分给各客户端（IID）。

    返回 list[list[int]]，每个元素是一个客户端持有的样本索引；
    不能整除时，前几个客户端多分 1 条样本。
    """
    n = len(train_set)
    base = n // num_clients
    rem = n % num_clients
    indices = np.random.RandomState(seed).permutation(n).tolist()
    client_indices = []
    start = 0
    for i in range(num_clients):
        size = base + (1 if i < rem else 0)
        client_indices.append(indices[start:start + size])
        start += size
    return client_indices
