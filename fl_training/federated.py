"""客户端本地训练与服务器端 FedAvg 聚合。"""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class Client:
    """联邦客户端：持有本地数据子集，执行本地训练。"""

    def __init__(self, client_id, dataset, indices, batch_size, device):
        self.client_id = client_id
        self.device = device
        self.num_samples = len(indices)
        self.loader = DataLoader(
            Subset(dataset, indices), batch_size=batch_size, shuffle=True
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
        """FedAvg: w <- sum_k (n_k / n) * w_k"""
        total = sum(client_sizes)
        averaged = {}
        for key in client_weights[0]:
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
