"""把 HuggingFace 的 CIFAR parquet 转换为本项目使用的 npz 数组文件。

背景：CIFAR 官方源站 ``www.cs.toronto.edu`` 在国内带宽极低（实测约 80 KB/s，
下载 339 MB 需 80 分钟且易断流），而 HuggingFace 镜像 ``hf-mirror.com`` 可达
1.8 MB/s。因此改为下载 HF 的 parquet（内含 PNG 字节流），在本地解码成
``(N, 32, 32, 3)`` 的 uint8 数组后存为 npz，供 ``data_utils.load_dataset`` 读取。

产物（与 torchvision 的 HWC uint8 布局一致，可直接喂给 PIL + transform）：
    <data_root>/cifar10_arrays.npz   train_data/train_labels/test_data/test_labels
    <data_root>/cifar100_arrays.npz  同上（标签取 fine_label，100 类）

用法：
    python fl_training/prepare_cifar.py --parquet-dir /tmp/cifar_pq
"""
import argparse
import io
import os

import numpy as np

# 每个数据集需要的 (parquet 文件名前缀, 标签列名)。
_SPECS = {
    "cifar10": {"label_col": "label", "prefix": "c10"},
    "cifar100": {"label_col": "fine_label", "prefix": "c100"},
}


def _decode_split(parquet_path: str, label_col: str):
    """读取单个 parquet，返回 (data uint8 (N,32,32,3), labels int64 (N,))。"""
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet_path)
    img_struct = table.column("img").to_pylist()
    labels = np.asarray(table.column(label_col).to_pylist(), dtype=np.int64)

    data = np.empty((len(img_struct), 32, 32, 3), dtype=np.uint8)
    for i, cell in enumerate(img_struct):
        img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        if arr.shape != (32, 32, 3):
            raise ValueError(f"{parquet_path} 第 {i} 张图尺寸异常: {arr.shape}")
        data[i] = arr
    return data, labels


def convert(dataset: str, parquet_dir: str, data_root: str) -> str:
    """转换一个数据集，返回生成的 npz 路径。"""
    spec = _SPECS[dataset]
    out_path = os.path.join(data_root, f"{dataset}_arrays.npz")
    splits = {}
    for split in ("train", "test"):
        src = os.path.join(parquet_dir, f"{spec['prefix']}_{split}.parquet")
        if not os.path.exists(src):
            raise FileNotFoundError(f"缺少 parquet: {src}")
        data, labels = _decode_split(src, spec["label_col"])
        splits[f"{split}_data"] = data
        splits[f"{split}_labels"] = labels
        print(f"  {dataset} {split}: {data.shape}, {len(np.unique(labels))} 类")

    os.makedirs(data_root, exist_ok=True)
    np.savez(out_path, **splits)
    print(f"  已保存 -> {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)")
    return out_path


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="CIFAR parquet -> npz 转换")
    p.add_argument("--parquet-dir", default="/tmp/cifar_pq",
                   help="存放 c10_train.parquet 等文件的目录")
    p.add_argument("--data-root", default=os.path.join(base_dir, "data"))
    p.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"],
                   choices=list(_SPECS.keys()))
    args = p.parse_args()

    for ds in args.datasets:
        print(f"转换 {ds} ...")
        convert(ds, args.parquet_dir, args.data_root)
    print("全部转换完成。")


if __name__ == "__main__":
    main()
