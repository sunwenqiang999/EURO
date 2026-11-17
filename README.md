# EURO：两阶段联邦学习预算分配仿真框架

本仓库研究**全局预算约束下的联邦学习训练时延优化**，将预算决策分解为
**轮间（inter-round）**与**轮内（intra-round）**两层，并在 MNIST / FMNIST /
CIFAR-10 / CIFAR-100 的预生成准确率曲线上，仿真不同预算机制达到目标精度所需的
累计训练时间。

> 理论内容对应中期答辩 PPT《两阶段联邦学习优化：轮间预算管理与轮内客户激励》。

## 方法总览

两阶段预算分配：

- **阶段一（轮间）** 决定每一轮的预算 `B_t`。
- **阶段二（轮内）** 在 `B_t` 内把预算支付给异构客户端，最小化最慢客户端完成时间。

对比方法：

| 方法 | 轮间 | 轮内 | 说明 |
|---|---|---|---|
| **Ours (GBSFL)** | IRBAI 收敛感知解析 | IRBSB Stackelberg 二分解析 | 本文方法 |
| **EURO-AFL** | InterBMA (DQN) | IntraBMA (策略网络) | 强化学习方法 |
| **Random** | 随机 | 随机 | 无结构基线 |
| **FedAvg** | 均匀 | 均匀 | 均匀预算基线 |
| **Fair** | — | 按 bid 比例 | 成本/质量导向（仅轮内） |

## 客户端时延模型

```
工作量         h_i = σ · c_i · d_i
计算成本系数   κ_i = a_i · h_i
客户端最优频率 f_i = sqrt(b_i / (2 κ_i))      # 给定支付 b_i
完成时间       T_i = h_i / f_i + T_up_i
轮次同步时延   = max_i T_i                      # 最慢客户端决定
```

轮内解析方法 (IRBSB) 二分共同目标时间 `M`，使所有客户端在 `M` 内完成且总支付 = 预算。

## 目录结构

```
euro/
  config.py       全局配置：相对路径、随机种子、数据集与客户端采样
  algorithms.py   轮间/轮内的解析式、随机、均匀、Fair 策略
  agents.py       轮内 IntraBMA 策略网络 + 轮间 InterBMA DQN (EURO-AFL)
  simulator.py    联邦轮次时延仿真器 + 敏感性实验
  plotting.py     accVstime 折线图 + 到达目标精度柱状图
run_experiments.py  实验主入口
timelist/           预生成准确率曲线 (.npy)
results_new/        逐轮时延结果 (.npy)
figures_new/        生成的对比图 (png + eps)
```

## 运行

```bash
pip install -r requirements.txt

# 运行全部数据集的轮内 + 轮间实验
python run_experiments.py

# 指定数据集 / 只跑某一层 / 附带敏感性实验
python run_experiments.py --datasets mnist fmnist --only intra
python run_experiments.py --sensitivity
```

## 产物

- `figures_new/intra/<dataset>_accVstime.{png,eps}` — 轮内方法精度-时间曲线
- `figures_new/intra/<dataset>_reachtime.{png,eps}` — 轮内到达目标精度时间柱状
- `figures_new/inter/<dataset>_*` — 轮间方法对比
- `figures_new/sensitivity/<dataset>_budget.*` — 总预算 B 敏感性
- `figures_new/sensitivity/<dataset>_psi.*` — 参数 ψ 敏感性

## 相比旧代码的主要修复

1. 移除全部 Windows 硬编码路径 `E:\pycharm\EURO\...`，改为相对路径，跨平台可运行。
2. 统一数据文件名大小写（`Fminstcnn` / `minstcnn`）。
3. 修复 `simulate_round` 参数错位（轮次编号被误当作客户端参数）。
4. 修复轮内二分搜索可能死循环 / 除零，改为稳健的指数扩张 + 二分。
5. 修复绘图 `np.argmax` 在「从未达标」时错误返回第 0 轮的 bug。
6. 统一准确率量纲（0–100）、DQN 学习率（1 → 1e-3）。
7. 固定随机种子，保证可复现；同一轮所有方法共享同组客户端参数。
8. 轮内强化学习改为可微策略网络，稳定优于随机/均匀基线。

旧脚本（`train*.py`、`eyy.py`、`eu.py`、`inter_*`、`generate_picture/` 等）及其产物
（`*.pth`、`simulation_results_*.json`、`figures/`、`time/`、`model/`）已全部删除，
当前流程完全由 `euro/` 包 + `run_experiments.py` 驱动。
