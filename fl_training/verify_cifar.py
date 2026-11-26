"""校验本地 npz 的 CIFAR 像素是否与官方 tar.gz 逐字节一致。

做法：对官方 cifar-10-python.tar.gz 做「流式」解压（无需下满 170MB），取出
data_batch_1，反序列化得到官方的 (10000, 3072) uint8 原始像素，再与
prepare_cifar.py 生成的 npz 做逐元素比对。

官方 pickle 布局（与 torchvision 读取方式一致）：
    entry['data']   : (10000, 3072) uint8，按 R 通道 1024 + G 1024 + B 1024 排列
    entry['labels'] : 长度 10000 的 int 列表
reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) 后即为 HWC 布局，可与 npz 直接比对。

用法：
    python fl_training/verify_cifar.py --tar /tmp/verify/official_c10_head.tar.gz
"""
import argparse
import os
import pickle
import sys
import tarfile

import numpy as np


def extract_official_batch(tar_path: str, member_name: str = "any"):
    """流式读取（可能被截断的）tar.gz，返回首个批次的 (data_hwc, labels, name)。

    官方归档内的成员顺序并非字母序（实测第一个数据批次是 ``data_batch_4``），
    因此默认匹配任意 ``data_batch_*`` / ``test_batch``，取最先遇到的一个即可。
    """
    seen = []
    # mode="r|gz" 为流式模式，不要求文件完整或可随机寻址。
    with tarfile.open(tar_path, mode="r|gz") as tar:
        try:
            for member in tar:
                seen.append(member.name)
                base = os.path.basename(member.name)
                if member_name == "any":
                    matched = base.startswith("data_batch") or base == "test_batch"
                else:
                    matched = base == member_name
                if not matched:
                    continue
                payload = tar.extractfile(member).read()
                entry = pickle.loads(payload, encoding="latin1")
                data = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                labels = np.asarray(entry["labels"], dtype=np.int64)
                return data, labels, base
        except (tarfile.ReadError, EOFError, OSError) as exc:
            # 文件被截断属预期情况：若目标成员已在截断点之前即可正常返回。
            print(f"  [提示] 归档在读取中截断（{type(exc).__name__}），已遍历成员: {seen}")
    return None, None, seen


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="校验本地 CIFAR npz 与官方数据是否一致")
    p.add_argument("--tar", required=True, help="官方 cifar-10-python.tar.gz（可为部分下载）")
    p.add_argument("--npz", default=os.path.join(base_dir, "data", "cifar10_arrays.npz"))
    p.add_argument("--member", default="any",
                   help="要比对的归档成员名；默认 any 表示取最先遇到的批次")
    args = p.parse_args()

    print(f"流式解压官方归档: {args.tar}")
    official, off_labels, name = extract_official_batch(args.tar, args.member)
    if official is None:
        print(f"  未能取到数据批次（下载量不足）。已见成员: {name}")
        return 2
    print(f"  官方成员 {name}: {official.shape} {official.dtype}, "
          f"标签前 10 项 = {off_labels[:10].tolist()}")

    with np.load(args.npz) as z:
        local = z["train_data"]
        loc_labels = z["train_labels"]
    print(f"  本地 npz train_data: {local.shape} {local.dtype}")

    n = len(official)
    print("\n=== 比对结果 ===")
    # 官方 5 个训练批次各 10000 张，拼接顺序为 batch_1..batch_5；
    # 若本地顺序与官方一致，则该批次应出现在对应的 10000 张切片上。
    exact_slice = None
    for b in range(5):
        seg = local[b * n:(b + 1) * n]
        if len(seg) == n and np.array_equal(seg, official):
            exact_slice = b + 1
            break
    print(f"  [1] 像素与官方某个 10000 张切片逐字节相同 : "
          f"{'是，位于第 ' + str(exact_slice) + ' 段' if exact_slice else '否'}")
    if exact_slice:
        seg_lab = loc_labels[(exact_slice - 1) * n: exact_slice * n]
        print(f"  [2] 该段标签与官方完全相同               : {np.array_equal(seg_lab, off_labels)}")

    # 无论顺序如何，核验「像素内容集合」是否被本地完整覆盖。
    h_off = [hash(official[i].tobytes()) for i in range(n)]
    h_loc = {hash(local[i].tobytes()) for i in range(len(local))}
    inter = sum(1 for h in h_off if h in h_loc)
    print(f"  [3] 官方该批 {n} 张图在本地 {len(local)} 张中的命中数 : "
          f"{inter}/{n} ({inter / n * 100:.2f}%)")
    if inter == n:
        print("      -> 100% 命中：像素内容与官方完全一致（无有损压缩/色彩偏差）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
