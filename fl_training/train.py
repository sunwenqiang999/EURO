"""FedAvg 基线联邦训练入口脚本。

流程：数据 IID 平均划分到客户端 -> 客户端本地训练 -> 服务器整体聚合（FedAvg）。

用法示例:
    python fl_training/train.py                        # 默认: MNIST, 20 客户端, 50 轮, 本地 3 epoch
    python fl_training/train.py --dataset cifar10 --num-rounds 50 --lr 0.01
"""
import argparse
import json
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
                   choices=["simplecnn", "lenet5", "resnet18"])
    p.add_argument("--num-clients", type=int, default=config.NUM_CLIENTS)
    p.add_argument("--num-rounds", type=int, default=config.NUM_ROUNDS)
    p.add_argument("--local-epochs", type=int, default=config.LOCAL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--momentum", type=float, default=config.MOMENTUM)
    p.add_argument("--seed", type=int, default=config.SEED,
                   help="全局随机种子；相同种子 + 相同设备必定复现相同结果")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch CPU 线程数，默认 1 以保证逐位复现（0 表示不限制）")
    p.add_argument("--data-root", default=config.DATA_ROOT)
    p.add_argument("--out-dir", default=config.OUT_DIR)
    p.add_argument(
        "--run-name", default=None,
        help="本次实验的自定义名称；提供后结果保存到独立子目录，避免覆盖。",
    )
    p.add_argument(
        "--timelist-dir", default=None,
        help="额外以规范名 <dataset>_trainacc.npy / <dataset>_testacc.npy 把"
             "准确率曲线写入该目录（供 euro 仿真直接读取）。每轮增量覆盖写入，"
             "长任务中途中断也能保留已完成轮次的结果。",
    )
    p.add_argument(
        "--model-dir", default=None,
        help="额外以规范名 <dataset>_<model>.pth 把训练好的全局模型保存到该目录，"
             "并同时写出 <dataset>_<model>_meta.json 记录超参与最终准确率。",
    )
    p.add_argument(
        "--no-figure", action="store_true",
        help="不生成准确率曲线图片，只产出 .npy 数据。",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # 统一播种：random / numpy / torch(CPU, CUDA, MPS) + cuDNN 确定性。
    config.set_seed(args.seed)
    # 固定 CPU 线程数，消除多线程浮点规约顺序带来的微小差异。
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = config.DEVICE
    run_name = args.run_name or (
        f"{args.dataset}_{args.model}_clients{args.num_clients}_"
        f"rounds{args.num_rounds}_epochs{args.local_epochs}_lr{args.lr:g}_"
        f"seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    # 记录完整运行配置，便于日后精确复现本次实验。
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "device": device,
                   "torch_version": torch.__version__},
                  f, ensure_ascii=False, indent=2)

    print(f"device = {device}")
    print(f"结果目录 = {run_dir}")
    print(f"dataset = {args.dataset}, model = {args.model}, clients = {args.num_clients}, "
          f"rounds = {args.num_rounds}, local_epochs = {args.local_epochs}")

    # 1) 数据：下载 + IID 平均划分到客户端
    train_set, test_set = load_dataset(args.dataset, args.data_root)
    client_indices = split_data_iid(train_set, args.num_clients, args.seed)
    clients = [
        Client(i, train_set, idx, args.batch_size, device, seed=args.seed)
        for i, idx in enumerate(client_indices)
    ]
    train_eval_loader = DataLoader(train_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    print(f"每个客户端样本数: {len(train_set) // args.num_clients} "
          f"(共 {len(train_set)} 条训练样本)")

    # 2) 服务器：初始化全局模型
    # 显式重设 torch 种子，使全局模型初始权重只依赖 seed，
    # 而与前面数据加载/DataLoader 构造消耗了多少随机数无关。
    torch.manual_seed(args.seed)
    server = FedAvgServer(build_model(args.dataset, args.model), device)
    num_params = sum(p.numel() for p in server.global_model.parameters())
    print(f"model = {args.model}, params = {num_params / 1e6:.2f}M")

    # 3) 联邦主循环：全体客户端本地训练 -> FedAvg 聚合 -> 全局测试评估
    train_acc_curve, test_acc_curve = [], []
    # 规范命名的曲线输出路径（供 euro 仿真直接读取）。
    timelist_paths = None
    if args.timelist_dir:
        os.makedirs(args.timelist_dir, exist_ok=True)
        timelist_paths = (
            os.path.join(args.timelist_dir, f"{args.dataset}_trainacc.npy"),
            os.path.join(args.timelist_dir, f"{args.dataset}_testacc.npy"),
        )
        print(f"timelist 输出 = {timelist_paths[0]} / {timelist_paths[1]}")

    def flush_curves():
        """把当前已完成轮次的曲线落盘（百分制，与 timelist/ 既有格式一致）。

        每轮都覆盖写入，使长时间任务即便中途中断也能保留已有结果。
        """
        tr = np.asarray(train_acc_curve, dtype=np.float32) * 100
        te = np.asarray(test_acc_curve, dtype=np.float32) * 100
        np.save(os.path.join(run_dir, "trainacc.npy"), tr)
        np.save(os.path.join(run_dir, "testacc.npy"), te)
        if timelist_paths:
            np.save(timelist_paths[0], tr)
            np.save(timelist_paths[1], te)

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
        flush_curves()
        elapsed_so_far = time.time() - start
        eta = elapsed_so_far / (r + 1) * (args.num_rounds - r - 1)
        print(
            f"Round {r + 1:3d}/{args.num_rounds}  "
            f"train_acc = {train_acc:.4f}  test_acc = {test_acc:.4f}  "
            f"[已用 {elapsed_so_far / 60:.1f}min, 预计剩余 {eta / 60:.1f}min]",
            flush=True,
        )

    elapsed = time.time() - start
    print(f"训练完成，总耗时 {elapsed / 60:.1f} 分钟")

    # 4) 保存准确率曲线（.npy，格式与项目 timelist/ 一致）与最终模型
    flush_curves()
    train_acc_path = os.path.join(run_dir, "trainacc.npy")
    test_acc_path = os.path.join(run_dir, "testacc.npy")
    model_path = os.path.join(run_dir, "fedavg_model.pth")
    torch.save(server.global_model.state_dict(), model_path)

    print(f"训练集准确率已保存: {train_acc_path}")
    print(f"测试集准确率已保存: {test_acc_path}")
    if timelist_paths:
        print(f"timelist 曲线已保存: {timelist_paths[0]}")
        print(f"timelist 曲线已保存: {timelist_paths[1]}")
    print(f"模型已保存: {model_path}")

    # 以规范名把模型另存一份到统一目录，便于后续直接按 <数据集>_<模型> 取用。
    if args.model_dir:
        os.makedirs(args.model_dir, exist_ok=True)
        tag = f"{args.dataset}_{args.model}"
        named_model = os.path.join(args.model_dir, f"{tag}.pth")
        torch.save(server.global_model.state_dict(), named_model)
        meta = {
            **vars(args),
            "device": device,
            "torch_version": torch.__version__,
            "num_params": num_params,
            "rounds_completed": len(train_acc_curve),
            "final_train_acc": float(train_acc_curve[-1] * 100),
            "final_test_acc": float(test_acc_curve[-1] * 100),
            "best_test_acc": float(max(test_acc_curve) * 100),
            "best_test_round": int(np.argmax(test_acc_curve)) + 1,
            "elapsed_minutes": round(elapsed / 60, 2),
        }
        with open(os.path.join(args.model_dir, f"{tag}_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"模型已另存: {named_model}")
        print(f"模型元信息已保存: {os.path.join(args.model_dir, f'{tag}_meta.json')}")

    # 5) 生成训练集/测试集准确率随全局通信轮次变化的图片（可用 --no-figure 跳过）
    if args.no_figure:
        return
    rounds = np.arange(1, len(train_acc_curve) + 1)
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
    print(f"准确率曲线图片已保存: {figure_path}")


if __name__ == "__main__":
    main()
