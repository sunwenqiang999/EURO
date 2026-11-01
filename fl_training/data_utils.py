"""数据集下载与 IID 平均划分。"""
import numpy as np
from torchvision import datasets, transforms


def load_dataset(dataset: str, data_root: str):
    """下载并返回 (train_set, test_set)。"""
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
        cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
        train_set = cls(data_root, train=True, download=True, transform=tf)
        test_set = cls(data_root, train=False, download=True, transform=tf)
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
