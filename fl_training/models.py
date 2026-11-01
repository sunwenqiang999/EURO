"""FedAvg 基线可选模型：SimpleCNN 与 LeNet-5。"""
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """输入 28x28 (MNIST/FMNIST) 或 32x32 (CIFAR) 的简单 CNN。

    参数规模: MNIST/FMNIST 约 1.66M，CIFAR-10 约 2.16M，CIFAR-100 约 2.20M。
    """

    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        reduced = input_size // 4  # 两次 2x2 池化: 28->7, 32->8
        self.fc1 = nn.Linear(64 * reduced * reduced, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LeNet5(nn.Module):
    """经典 LeNet-5，使用 ReLU/最大池化的现代等价实现。

    支持 28x28（MNIST/Fashion-MNIST）和 32x32（CIFAR）图像。
    """

    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        reduced = (input_size - 4) // 2  # 第一次卷积和池化后的边长
        reduced = (reduced - 4) // 2     # 第二次卷积和池化后的边长
        self.fc1 = nn.Linear(16 * reduced * reduced, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def build_model(dataset: str, model_name: str = "simplecnn") -> nn.Module:
    """按数据集和模型名称构建分类模型。"""
    if dataset in ("mnist", "fmnist"):
        in_channels, num_classes, input_size = 1, 10, 28
    elif dataset == "cifar10":
        in_channels, num_classes, input_size = 3, 10, 32
    elif dataset == "cifar100":
        in_channels, num_classes, input_size = 3, 100, 32
    else:
        raise ValueError(f"未知数据集: {dataset}")

    if model_name == "simplecnn":
        return SimpleCNN(in_channels, num_classes, input_size)
    if model_name == "lenet5":
        return LeNet5(in_channels, num_classes, input_size)
    raise ValueError(f"未知模型: {model_name}")
