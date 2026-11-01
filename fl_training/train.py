"""FedAvg 基线联邦训练入口脚本。

流程：数据 IID 平均划分到客户端 -> 客户端本地训练 -> 服务器整体聚合（FedAvg）。

用法示例:
    python fl_training/train.py                        # 默认: MNIST, 20 客户端, 50 轮, 本地 3 epoch
    python fl_training/train.py --dataset cifar10 --num-rounds 50 --lr 0.01
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data_utils import load_dataset, split_data_iid
from federated import Client, FedAvgServer
from models import build_model


def parse_args():
    p = argparse.ArgumentParser(description="FedAvg baseline federated training")
    p.add_argument("--dataset", default=config.DATASET,
                   choices=["mnist", "fmnist", "cifar10", "cifar100"])
    p.add_argument("--model", default=config.MODEL,
                   choices=["simplecnn", "lenet5"])
    p.add_argument("--num-clients", type=int, default=config.NUM_CLIENTS)
    p.add_argument("--num-rounds", type=int, default=config.NUM_ROUNDS)
    p.add_argument("--local-epochs", type=int, default=config.LOCAL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--momentum", type=float, default=config.MOMENTUM)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--data-root", default=config.DATA_ROOT)
    p.add_argument("--out-dir", default=config.OUT_DIR)
    p.add_argument(
        "--run-name", default=None,
        help="本次实验的自定义名称；提供后结果保存到独立子目录，避免覆盖。",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = config.DEVICE
    run_name = args.run_name or (
        f"{args.dataset}_{args.model}_clients{args.num_clients}_"
        f"rounds{args.num_rounds}_epochs{args.local_epochs}_lr{args.lr:g}_"
        f"seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"device = {device}")
    print(f"结果目录 = {run_dir}")
    print(f"dataset = {args.dataset}, model = {args.model}, clients = {args.num_clients}, "
          f"rounds = {args.num_rounds}, local_epochs = {args.local_epochs}")

    # 1) 数据：下载 + IID 平均划分到客户端
    train_set, test_set = load_dataset(args.dataset, args.data_root)
    client_indices = split_data_iid(train_set, args.num_clients, args.seed)
    clients = [
        Client(i, train_set, idx, args.batch_size, device)
        for i, idx in enumerate(client_indices)
    ]
    train_eval_loader = DataLoader(train_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    print(f"每个客户端样本数: {len(train_set) // args.num_clients} "
          f"(共 {len(train_set)} 条训练样本)")

    # 2) 服务器：初始化全局模型
    server = FedAvgServer(build_model(args.dataset, args.model), device)
    num_params = sum(p.numel() for p in server.global_model.parameters())
    print(f"model = {args.model}, params = {num_params / 1e6:.2f}M")

    # 3) 联邦主循环：全体客户端本地训练 -> FedAvg 聚合 -> 全局测试评估
    train_acc_curve, test_acc_curve = [], []
    start = time.time()
    for r in range(args.num_rounds):
        weights, sizes = [], []
        for client in clients:
            w = client.local_train(server.global_model, args.local_epochs,
                                   args.lr, args.momentum)
            weights.append(w)
            sizes.append(client.num_samples)
        server.aggregate(weights, sizes)
        train_acc = server.evaluate(train_eval_loader)
        test_acc = server.evaluate(test_loader)
        train_acc_curve.append(train_acc)
        test_acc_curve.append(test_acc)
        print(
            f"Round {r + 1:3d}/{args.num_rounds}  "
            f"train_acc = {train_acc:.4f}  test_acc = {test_acc:.4f}"
        )

    elapsed = time.time() - start
    print(f"训练完成，总耗时 {elapsed / 60:.1f} 分钟")

    # 4) 保存准确率曲线（.npy，格式与项目 timelist/ 一致）与最终模型
    train_acc_path = os.path.join(run_dir, "trainacc.npy")
    test_acc_path = os.path.join(run_dir, "testacc.npy")
    np.save(train_acc_path, np.asarray(train_acc_curve, dtype=np.float32) * 100)
    np.save(test_acc_path, np.asarray(test_acc_curve, dtype=np.float32) * 100)
    model_path = os.path.join(run_dir, "fedavg_model.pth")
    torch.save(server.global_model.state_dict(), model_path)

    # 5) 生成训练集/测试集准确率随全局通信轮次变化的图片
    rounds = np.arange(1, args.num_rounds + 1)
    train_acc_percent = np.asarray(train_acc_curve) * 100
    test_acc_percent = np.asarray(test_acc_curve) * 100
    plt.figure(figsize=(9, 5))
    plt.plot(rounds, train_acc_percent, color="tab:blue", linewidth=1.8,
             label="Training accuracy")
    plt.plot(rounds, test_acc_percent, color="tab:orange", linewidth=1.8,
             label="Test accuracy")
    plt.xlabel("Global communication round")
    plt.ylabel("Accuracy (%)")
    plt.title(f"FedAvg on {args.dataset.upper()} ({args.model.upper()})")
    plt.xlim(1, max(args.num_rounds, 2))
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, color="gray", linestyle="-", linewidth=0.8, alpha=0.45)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    figure_path = os.path.join(run_dir, "accuracy_by_round.png")
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"训练集准确率已保存: {train_acc_path}")
    print(f"测试集准确率已保存: {test_acc_path}")
    print(f"模型已保存: {model_path}")
    print(f"准确率曲线图片已保存: {figure_path}")


if __name__ == "__main__":
    main()
