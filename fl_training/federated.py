"""客户端本地训练与服务器端 FedAvg 聚合。"""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import seed_worker


class Client:
    """联邦客户端：持有本地数据子集，执行本地训练。

    Args:
        seed: 该客户端 DataLoader 的 shuffle 种子。给定后绑定独立的
              ``torch.Generator``，不再依赖 torch 全局 RNG——否则每轮每个
              客户端的 batch 顺序会随「之前消耗了多少随机数」而变化，
              导致训练结果不可复现。
    """

    def __init__(self, client_id, dataset, indices, batch_size, device,
                 seed=None, num_workers=0):
        self.client_id = client_id
        self.device = device
        self.num_samples = len(indices)
        generator = None
        if seed is not None:
            generator = torch.Generator()
            # 不同客户端用不同但确定的种子，避免所有客户端共用同一打乱顺序。
            generator.manual_seed(int(seed) + int(client_id))
        self.loader = DataLoader(
            Subset(dataset, indices), batch_size=batch_size, shuffle=True,
            generator=generator, num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None,
        )

    def local_train(self, global_model, local_epochs, lr, momentum):
        """拷贝全局模型，在本地数据上训练 local_epochs 轮，返回 CPU 上的模型权重。"""
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        for _ in range(local_epochs):
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class FedAvgServer:
    """服务器：维护全局模型，按客户端数据量加权聚合（FedAvg）并评估。"""

    def __init__(self, model, device):
        self.device = device
        self.global_model = model.to(device)

    def aggregate(self, client_weights, client_sizes):
        """FedAvg: w <- sum_k (n_k / n) * w_k

        对 BatchNorm 的整型缓冲区（``num_batches_tracked``，dtype=int64）做特殊
        处理：加权平均会把它变成浮点数，再经 ``load_state_dict`` 写回 int64 时
        被截断（如 234.999 -> 234），造成计数持续偏小。这里改为直接取各客户端
        的最大值，语义上等价于「全局模型已见过的 batch 数」，且保持整型。
        ResNet-18 等含 BN 的模型会命中该分支；SimpleCNN / LeNet-5 无 BN，不受影响。
        """
        total = sum(client_sizes)
        averaged = {}
        for key in client_weights[0]:
            ref = client_weights[0][key]
            if not ref.dtype.is_floating_point:
                # 整型缓冲区（num_batches_tracked 等）不做加权平均。
                averaged[key] = torch.stack([w[key] for w in client_weights]).max(dim=0).values
            else:
                averaged[key] = sum(
                    w[key] * (n / total) for w, n in zip(client_weights, client_sizes)
                )
        self.global_model.load_state_dict(averaged)
        return self.global_model

    @torch.no_grad()
    def evaluate(self, data_loader):
        """在给定数据集上评估当前全局模型的准确率。"""
        self.global_model.eval()
        correct, total = 0, 0
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            pred = self.global_model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        return correct / total
