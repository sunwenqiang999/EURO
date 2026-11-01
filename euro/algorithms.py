"""轮间 (inter-round) 与轮内 (intra-round) 预算分配算法。

统一的客户端时延模型（对应中期答辩 PPT 系统模型）：

    工作量         h_i   = σ · c_i · d_i
    计算成本系数   κ_i   = a_i · h_i          (a_i 为原始能耗系数)
    客户端最优频率 f_i   = sqrt(b_i / (2 κ_i))  (给定实际支付 b_i)
    完成时间       T_i   = h_i / f_i + T_up_i
    实际支付       b_i   = 2 κ_i f_i²

轮次同步时延 = max_i T_i（最慢客户端决定）。

给定「本轮分给客户端 i 的预算 b_i」，可直接得到：

    f_i = sqrt(b_i / (2 κ_i))
    T_i = h_i / f_i + T_up_i
        = h_i / sqrt(b_i / (2 κ_i)) + T_up_i

下面的轮内方法都先确定每个客户端的预算 b_i，再用上式算完成时间。
轮内解析方法 (IRBSB) 则反过来：二分共同目标时间 M，反推 b_i。
"""

from __future__ import annotations

import numpy as np

# 数值稳定性常量。
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# 公共时延计算
# --------------------------------------------------------------------------- #
def completion_times(budgets: np.ndarray, d_i, c_i, a_i, sigma, T_up) -> np.ndarray:
    """给定每个客户端的实际支付，返回其完成时间向量。

    若某客户端支付为 0，其频率为 0、完成时间视为无穷大（无法完成）。
    """
    budgets = np.asarray(budgets, dtype=np.float64)
    h = sigma * c_i * d_i                 # 工作量
    kappa = a_i * h                        # 计算成本系数 κ = a · h
    # f = sqrt(b / (2κ))
    freq = np.sqrt(np.maximum(budgets, 0.0) / (2.0 * kappa + _EPS))
    times = np.where(freq > _EPS, h / (freq + _EPS) + T_up, np.inf)
    return times


def round_time(budgets: np.ndarray, d_i, c_i, a_i, sigma, T_up) -> float:
    """轮次同步时延 = 最慢客户端完成时间。"""
    return float(np.max(completion_times(budgets, d_i, c_i, a_i, sigma, T_up)))


# --------------------------------------------------------------------------- #
# 轮间 (inter-round) 预算分配
# --------------------------------------------------------------------------- #
def inter_ours(accuracy: np.ndarray, B: float, T: int, psi: float, Q: float) -> np.ndarray:
    """IRBAI：收敛感知轮间预算分配（本文方法，对应 PPT 阶段一）。

    思想：训练早期准确率提升快（∇S_k 大），应分配更多预算；随着收敛放缓
    预算随之降低。``psi >= 1`` 控制精度增量的影响强度，``psi == 1`` 退化为均匀。

    Args:
        accuracy: 预生成准确率曲线 (0-100)，长度应为 T。
        B: 全局预算。
        T: 总轮数。
        psi: 可调参数 ψ (>= 1)。
        Q: 收敛判定阈值；准确率超过 Q 后按剩余均值分配。

    Returns:
        长度为 T 的每轮预算数组。
    """
    accuracy = np.asarray(accuracy, dtype=np.float64)
    budgets = []
    # 首轮给予两倍均值的启动预算。
    b0 = B / T * 2.0
    budgets.append(b0)
    spent = b0
    prev_mean = 0.0
    running_sum = 0.0

    for i in range(len(accuracy) - 1):
        remaining_rounds = T - i - 1
        avg_remaining = (B - spent) / max(remaining_rounds, 1)
        if accuracy[i] < Q and i != len(accuracy) - 2:
            # 准确率增量（相对历史均值），归一化到 [0, 0.5]。
            s = max((accuracy[i] - prev_mean) * 0.01, 0.0)
            s = min(s, 0.5)
            p = avg_remaining * (psi ** s)
        else:
            p = avg_remaining
        running_sum += accuracy[i]
        prev_mean = running_sum / (i + 2)
        spent += p
        budgets.append(round(p, 4))

    return np.asarray(budgets, dtype=np.float64)


def inter_average(B: float, T: int) -> np.ndarray:
    """FedAvg 轮间：每轮分配相同预算 B/T。"""
    return np.full(T, round(B / T, 4), dtype=np.float64)


def inter_random(B: float, T: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """随机轮间：随机权重归一化后乘以 B。"""
    draw = rng.random if rng is not None else np.random.random
    r = draw(T) * 10.0 + 0.5
    r = r / r.sum() * B
    return np.round(r, 4)


# --------------------------------------------------------------------------- #
# 轮内 (intra-round) 预算分配
# --------------------------------------------------------------------------- #
def intra_average(P: float, d_i, c_i, a_i, sigma, T_up):
    """FedAvg 轮内：预算在客户端间均分。"""
    k = len(d_i)
    budgets = np.full(k, P / k, dtype=np.float64)
    times = completion_times(budgets, d_i, c_i, a_i, sigma, T_up)
    return float(np.max(times)), budgets, times


def intra_random(P: float, d_i, c_i, a_i, sigma, T_up, rng: np.random.Generator | None = None):
    """随机轮内：随机权重归一化后乘以 P。"""
    k = len(d_i)
    draw = rng.random if rng is not None else np.random.random
    r = draw(k) + 0.25
    budgets = r / r.sum() * P
    times = completion_times(budgets, d_i, c_i, a_i, sigma, T_up)
    return float(np.max(times)), budgets, times


def intra_fair(P: float, d_i, c_i, a_i, sigma, T_up):
    """Fair 轮内：按客户端出价 bid 比例分配预算。

    bid 由计算成本与上传时间构造，刻画客户端自身属性（对应 PPT Fair 基线）。
    """
    bid = sigma * c_i * d_i + T_up
    budgets = bid / np.sum(bid) * P
    times = completion_times(budgets, d_i, c_i, a_i, sigma, T_up)
    return float(np.max(times)), budgets, times


def intra_ours(P: float, d_i, c_i, a_i, sigma, T_up,
               tol: float = 1e-6, max_iter: int = 200):
    """IRBSB：Stackelberg 博弈 + 二分搜索解析轮内分配（本文方法，PPT 阶段二）。

    通过二分寻找共同目标完成时间 M，使所有客户端在 M 内完成且总支付等于预算 P。

    支付与目标时间关系（无频率上下限的连续模型）：
        要求 T_i = h_i / f_i + T_up_i <= M
        最紧时 f_i = h_i / (M - T_up_i)
        b_i = 2 κ_i f_i² = 2 κ_i h_i² / (M - T_up_i)²
    总支付 F(M) = Σ b_i 随 M 单调递减，可二分。

    Returns:
        (M, budgets, times)
    """
    d_i = np.asarray(d_i, dtype=np.float64)
    c_i = np.asarray(c_i, dtype=np.float64)
    a_i = np.asarray(a_i, dtype=np.float64)
    T_up = np.asarray(T_up, dtype=np.float64)

    h = sigma * c_i * d_i          # 工作量 h_i
    kappa = a_i * h                 # κ_i = a_i · h_i
    tau = T_up                      # 上传时间

    def required_budget(M: float) -> np.ndarray:
        """给定目标时间 M，各客户端最低所需支付。"""
        slack = M - tau
        # M 必须大于所有上传时间，否则该客户端不可行（支付 +inf）。
        f = np.where(slack > _EPS, h / slack, np.inf)
        return 2.0 * kappa * f * f

    # 二分区间：
    # 上界 R 需使 F(R) <= P（时间足够长，预算充裕）。
    # 下界 L 需使 F(L) >= P（时间很短，预算不足）。
    # 时间下限必须严格大于最大上传时间。
    tau_max = float(np.max(tau))
    L = tau_max + 1e-6               # 极短时间 -> 需要极大预算 -> F(L) 很大
    # 指数扩张寻找可行上界 R。
    R = L + 1.0
    for _ in range(100):
        if np.all(np.isfinite(required_budget(R))) and required_budget(R).sum() <= P:
            break
        R *= 2.0
    else:
        # 预算极度不足，无法在任何合理时间内完成：返回退化的均分结果。
        return intra_average(P, d_i, c_i, a_i, sigma, T_up)

    budgets = required_budget(R)
    M = R
    for _ in range(max_iter):
        if R - L <= tol:
            break
        mid = 0.5 * (L + R)
        need = required_budget(mid)
        if not np.all(np.isfinite(need)):
            # mid 太小，部分客户端不可行 -> 放宽时间。
            L = mid
            continue
        total = need.sum()
        if total > P:          # 预算不足 -> 放宽时间
            L = mid
        else:                  # 预算充裕 -> 收紧时间（保存可行解）
            R = mid
            M = mid
            budgets = need

    times = completion_times(budgets, d_i, c_i, a_i, sigma, T_up)
    return float(M), budgets, times
