"""FedAvg 基线可选模型：SimpleCNN、LeNet-5 与 ResNet-18。"""
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


def build_resnet18(in_channels: int = 3, num_classes: int = 10,
                   input_size: int = 32) -> nn.Module:
    """CIFAR 版 ResNet-18（约 11.2M 参数）。

    torchvision 原版 ResNet-18 是为 224x224 的 ImageNet 设计的：首层是
    7x7/stride2 卷积，其后还接 3x3/stride2 的 MaxPool。若直接喂 32x32 的 CIFAR
    图像，经这两步就降到 8x8，再过 layer2/3/4 只剩 1x1，空间信息几乎丢光，
    精度通常比正常实现低 10 个百分点以上。

    因此采用学术界通行的「CIFAR 版 ResNet-18」改法（He et al. 原文针对 CIFAR
    的处理方式，亦为后续绝大多数论文的标准做法）：
        - 首层换成 3x3 / stride=1 / padding=1 卷积；
        - 去掉首层后的 MaxPool（替换为 Identity）。
    这样特征图尺寸为 32 -> 32 -> 16 -> 8 -> 4，与原论文在 CIFAR 上的设定一致。

    Args:
        in_channels: 输入通道数（灰度图为 1，RGB 为 3）。
        num_classes: 分类类别数。
        input_size: 输入边长；大于等于 64 时保留 ImageNet 原始首层结构。
    """
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    if input_size < 64:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                                padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
    return model


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
    if model_name == "resnet18":
        return build_resnet18(in_channels, num_classes, input_size)
    raise ValueError(f"未知模型: {model_name}")
