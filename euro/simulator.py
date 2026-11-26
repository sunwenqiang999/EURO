"""联邦学习轮次时延仿真器。

核心原则（修复原代码参数错位 bug）：
    - 同一轮内，所有对比方法共享同一组客户端异构参数，保证横向比较公平。
    - 通过显式的 numpy Generator 控制随机性，保证可复现。
    - 智能体（IntraBMA / InterBMA）各自持有由 ``sim.seed`` 稳定派生的独立随机
      种子，网络初始化、ε-探索、经验回放均不触碰全局 RNG。因此单跑一个实验
      与批量跑多个实验/多个数据集，结果完全一致。

提供的实验：
    run_intra_experiment : 固定轮间策略（IRBAI），对比 5 种轮内方法（轮内消融）。
    run_inter_experiment : 固定轮内策略（IRBSB 解析），对比 4 种轮间方法（轮间消融）。
    run_joint_experiment : 五种方法各用完整两层组合的整体对比。
    run_budget_sensitivity / run_psi_sensitivity : 预算 B / 参数 ψ 敏感性。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import algorithms as alg
from .agents import IntraBMAAgent, InterBMAAgent
from .config import SimConfig, derive_seed, sample_clients


# --------------------------------------------------------------------------- #
# 轮内对比实验
# --------------------------------------------------------------------------- #
INTRA_WARMUP_EPOCHS = 300     # 轮内策略预训练 epoch 数（每 epoch = num_rounds 个样本）
INTER_PRETRAIN_EPISODES = 200  # 轮间 DQN 预训练 episode 数


def _warmup_intra_agent(agent: IntraBMAAgent, round_budgets, sim: SimConfig,
                        warm_seed: int, epochs: int = INTRA_WARMUP_EPOCHS) -> None:
    """用纯强化学习（REINFORCE）预训练轮内策略。

    每步：采样一组客户端 -> 智能体带噪声给出预算分配 -> 环境返回本轮时延
    -> 奖励 ``1 - T / T_uniform`` -> 策略梯度更新。智能体全程不接触时延公式。

    预算在评估阶段会落入 [0.5, 2.5] 倍的 IRBAI 预算范围内（联合实验里由轮间
    DQN 决定），因此预训练时在该范围随机取预算，避免训练/评估分布不一致。
    使用独立随机流，不污染评估阶段的客户端采样序列。
    """
    warm_rng = np.random.default_rng(warm_seed)
    std_hi, std_lo = 0.6, 0.1
    for ep in range(epochs):
        agent.explore_std = std_hi + (std_lo - std_hi) * ep / max(epochs - 1, 1)
        for r in range(sim.num_rounds):
            budget = float(round_budgets[r]) * float(warm_rng.uniform(0.5, 2.5))
            cp = sample_clients(sim, warm_rng)
            state = agent.build_state(budget, cp)
            _, payments = agent.act(state, budget)
            t = float(np.max(alg.completion_times(payments, cp.d_i, cp.c_i, cp.a_i,
                                                  cp.sigma, cp.T_up)))
            t_ref, _, _ = alg.intra_average(budget, cp.d_i, cp.c_i, cp.a_i,
                                            cp.sigma, cp.T_up)
            reward = float(np.clip(1.0 - t / t_ref, -3.0, 1.0))
            agent.learn(state, agent.last_logits, reward)
    agent.set_eval()


def _pretrain_inter_agent(agent: InterBMAAgent, accuracy: np.ndarray, sim: SimConfig,
                          target_accuracy: float, seed: int, time_fn,
                          episodes: int = INTER_PRETRAIN_EPISODES) -> None:
    """多 episode 预训练轮间 DQN。

    每个 episode 重新采样客户端，完整走 num_rounds 轮；时延由 ``time_fn(budget, cp)``
    给出（轮间实验用解析轮内分配，联合实验用已训练好的轮内策略）。
    训练结束后重置历史并关闭探索，评估阶段不再更新网络。
    """
    rng = np.random.default_rng(seed)
    T, B = sim.num_rounds, sim.total_budget

    # 标定时延量纲：均分预算下的平均轮时延。
    ref = []
    for _ in range(20):
        cp = sample_clients(sim, rng)
        t, _, _ = alg.intra_average(B / T, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)
        ref.append(t)
    agent.time_scale = float(np.mean(ref))

    for _ in range(episodes):
        agent.reset_history()
        remaining = B
        for r in range(T):
            cp = sample_clients(sim, rng)
            acc = float(accuracy[r])
            state = agent.build_state(r, remaining, T, B)
            action = agent.select_action(state)
            budget = agent.action_to_budget(action, remaining, r, T, acc, target_accuracy)
            t, payments, tlist = time_fn(budget, cp)
            prev_acc = agent.history["accuracies"][-1] if agent.history["accuracies"] else 0.0
            reward = agent.calculate_reward(acc, prev_acc, t)
            remaining -= float(np.sum(payments))
            agent.update_history(payments, tlist, acc)
            next_state = agent.build_state(r + 1, remaining, T, B)
            done = (r >= T - 1) or (remaining <= 0)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            if done:
                break
    agent.reset_history()
    agent.set_eval()


def _analytic_time_fn(budget, cp):
    return alg.intra_ours(budget, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)


def _policy_time_fn(intra_agent: IntraBMAAgent):
    def fn(budget, cp):
        _, payments = intra_agent.act(intra_agent.build_state(budget, cp), budget)
        tl = alg.completion_times(payments, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)
        return float(np.max(tl)), payments, tl
    return fn


def run_intra_experiment(accuracy: np.ndarray, sim: SimConfig,
                         target_accuracy: float,
                         rng: np.random.Generator) -> Dict[str, List[float]]:
    """固定轮间 IRBAI，比较 5 种轮内预算分配方法的逐轮时延。

    Returns:
        {method: [逐轮时延]} ，method ∈
        {random, average, fair, euro_afl, ours}
    """
    sim.irbai_Q = target_accuracy
    round_budgets = alg.inter_ours(accuracy, sim.total_budget, sim.num_rounds,
                                   sim.irbai_psi, target_accuracy)

    agent = IntraBMAAgent(sim.num_clients,
                          seed=derive_seed(sim.seed, "intra", "intra_agent"))
    # 预训练：纯强化学习（只用时延奖励），评估阶段固定策略。
    _warmup_intra_agent(agent, round_budgets, sim,
                        derive_seed(sim.seed, "intra", "warmup"))
    times = {k: [] for k in ("random", "average", "fair", "euro_afl", "ours")}

    for r in range(sim.num_rounds):
        budget = float(round_budgets[r])
        cp = sample_clients(sim, rng)        # 本轮所有方法共享的客户端参数

        t_avg, _, _ = alg.intra_average(budget, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)
        t_rand, _, _ = alg.intra_random(budget, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up, rng)
        t_fair, _, _ = alg.intra_fair(budget, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)
        t_ours, _, _ = alg.intra_ours(budget, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)

        # EURO-AFL（轮内策略网络）
        state = agent.build_state(budget, cp)
        weights, payments = agent.act(state, budget)
        rl_times = alg.completion_times(payments, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)
        t_rl = float(np.max(rl_times))

        times["random"].append(t_rand)
        times["average"].append(t_avg)
        times["fair"].append(t_fair)
        times["euro_afl"].append(t_rl)
        times["ours"].append(t_ours)

    return times


# --------------------------------------------------------------------------- #
# 轮间对比实验
# --------------------------------------------------------------------------- #
def run_inter_experiment(accuracy: np.ndarray, sim: SimConfig,
                         target_accuracy: float,
                         rng: np.random.Generator,
                         budgets_out: dict | None = None) -> Dict[str, List[float]]:
    """固定轮内 IRBSB 解析分配，比较不同轮间策略的逐轮时延。

    Returns:
        {method: [逐轮时延]}，method ∈
        {random, average, euro_afl, ours}
        （对应 PPT 轮间消融：IRBAI / InterBMA / Random / FedAvg）

    若传入 ``budgets_out``（dict），会把各方法每轮实际投入的预算写入其中
    （键同上，值为长度 T 的列表），仅用于分析展示，不影响实验结果。
    """
    sim.irbai_Q = target_accuracy
    budgets_ours = alg.inter_ours(accuracy, sim.total_budget, sim.num_rounds,
                                  sim.irbai_psi, target_accuracy)
    budgets_avg = alg.inter_average(sim.total_budget, sim.num_rounds)
    budgets_rand = alg.inter_random(sim.total_budget, sim.num_rounds, rng)

    inter_agent = InterBMAAgent(num_clients=sim.num_clients,
                                seed=derive_seed(sim.seed, "inter", "inter_agent"))
    _pretrain_inter_agent(inter_agent, accuracy, sim, target_accuracy,
                          derive_seed(sim.seed, "inter", "pretrain"), _analytic_time_fn)
    remaining = sim.total_budget

    times = {k: [] for k in ("random", "average", "euro_afl", "ours")}
    if budgets_out is not None:
        budgets_out.update({"ours": [float(x) for x in budgets_ours],
                            "average": [float(x) for x in budgets_avg],
                            "random": [float(x) for x in budgets_rand],
                            "euro_afl": []})

    for r in range(sim.num_rounds):
        cp = sample_clients(sim, rng)        # 本轮共享客户端参数
        acc = float(accuracy[r])

        def intra_time(b: float) -> tuple:
            m, budgets, t = alg.intra_ours(b, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)
            return m, budgets, t

        t_ours, _, _ = intra_time(float(budgets_ours[r]))
        t_avg, _, _ = intra_time(float(budgets_avg[r]))
        t_rand, _, _ = intra_time(float(budgets_rand[r]))

        # EURO-AFL 轮间 DQN 决定本轮预算，再用解析轮内分配。
        state = inter_agent.build_state(r, remaining, sim.num_rounds, sim.total_budget)
        action = inter_agent.select_action(state)
        rl_budget = inter_agent.action_to_budget(action, remaining, r,
                                                 sim.num_rounds, acc, target_accuracy)
        t_rl, rl_payments, rl_tlist = intra_time(rl_budget)
        if budgets_out is not None:
            budgets_out["euro_afl"].append(float(rl_budget))

        remaining -= float(np.sum(rl_payments))
        inter_agent.update_history(rl_payments, rl_tlist, acc)   # 评估阶段只更新历史，不再学习

        times["random"].append(t_rand)
        times["average"].append(t_avg)
        times["euro_afl"].append(t_rl)
        times["ours"].append(t_ours)

    return times


# --------------------------------------------------------------------------- #
# 轮间 + 轮内联合对比实验（每种方法用自己完整的两层组合）
# --------------------------------------------------------------------------- #
def run_joint_experiment(accuracy: np.ndarray, sim: SimConfig,
                         target_accuracy: float,
                         rng: np.random.Generator,
                         budgets_out: dict | None = None) -> Dict[str, List[float]]:
    """五种方法各用自己完整的「轮间 + 轮内」组合，比较逐轮时延。

    对应中期答辩 PPT「对比算法」整体对比：
        ours (GBSFL) : 轮间 IRBAI + 轮内 IRBSB（解析）
        euro_afl     : 轮间 InterBMA(DQN) + 轮内 IntraBMA(策略网)
        random       : 轮间随机 + 轮内随机
        average(FedAvg): 轮间均匀 + 轮内均匀
        fair         : 轮间均匀 + 轮内 Fair（Fair 仅定义轮内，轮间取均匀）

    Returns:
        {method: [逐轮时延]}，method ∈ {random, average, fair, euro_afl, ours}
    """
    sim.irbai_Q = target_accuracy
    # 各方法的轮间预算序列。
    budgets_ours = alg.inter_ours(accuracy, sim.total_budget, sim.num_rounds,
                                  sim.irbai_psi, target_accuracy)
    budgets_avg = alg.inter_average(sim.total_budget, sim.num_rounds)
    budgets_rand = alg.inter_random(sim.total_budget, sim.num_rounds, rng)

    # 轮内 RL 智能体（EURO-AFL 第二阶段）先预训练。
    intra_agent = IntraBMAAgent(sim.num_clients,
                                seed=derive_seed(sim.seed, "joint", "intra_agent"))
    _warmup_intra_agent(intra_agent, budgets_ours, sim,
                        derive_seed(sim.seed, "joint", "warmup"))
    # 轮间 RL 智能体（EURO-AFL 第一阶段）。
    inter_agent = InterBMAAgent(num_clients=sim.num_clients,
                                seed=derive_seed(sim.seed, "joint", "inter_agent"))
    _pretrain_inter_agent(inter_agent, accuracy, sim, target_accuracy,
                          derive_seed(sim.seed, "joint", "inter_pretrain"),
                          _policy_time_fn(intra_agent))
    remaining = sim.total_budget

    times = {k: [] for k in ("random", "average", "fair", "euro_afl", "ours")}
    if budgets_out is not None:   # 仅用于分析展示，不影响实验结果
        budgets_out.update({"ours": [float(x) for x in budgets_ours],
                            "average": [float(x) for x in budgets_avg],
                            "fair": [float(x) for x in budgets_avg],
                            "random": [float(x) for x in budgets_rand],
                            "euro_afl": []})

    for r in range(sim.num_rounds):
        cp = sample_clients(sim, rng)        # 本轮共享客户端参数
        acc = float(accuracy[r])

        # ours: IRBAI + IRBSB
        t_ours, _, _ = alg.intra_ours(float(budgets_ours[r]), cp.d_i, cp.c_i,
                                      cp.a_i, cp.sigma, cp.T_up)
        # random: 随机轮间 + 随机轮内
        t_rand, _, _ = alg.intra_random(float(budgets_rand[r]), cp.d_i, cp.c_i,
                                        cp.a_i, cp.sigma, cp.T_up, rng)
        # average(FedAvg): 均匀轮间 + 均匀轮内
        t_avg, _, _ = alg.intra_average(float(budgets_avg[r]), cp.d_i, cp.c_i,
                                        cp.a_i, cp.sigma, cp.T_up)
        # fair: 均匀轮间 + 轮内 Fair
        t_fair, _, _ = alg.intra_fair(float(budgets_avg[r]), cp.d_i, cp.c_i,
                                      cp.a_i, cp.sigma, cp.T_up)

        # euro_afl: 轮间 DQN 决定预算 -> 轮内策略网分配
        state = inter_agent.build_state(r, remaining, sim.num_rounds, sim.total_budget)
        action = inter_agent.select_action(state)
        rl_budget = inter_agent.action_to_budget(action, remaining, r,
                                                 sim.num_rounds, acc, target_accuracy)
        if budgets_out is not None:
            budgets_out["euro_afl"].append(float(rl_budget))
        intra_state = intra_agent.build_state(rl_budget, cp)
        _, rl_payments = intra_agent.act(intra_state, rl_budget)
        rl_tlist = alg.completion_times(rl_payments, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)
        t_rl = float(np.max(rl_tlist))

        remaining -= float(np.sum(rl_payments))
        inter_agent.update_history(rl_payments, rl_tlist, acc)   # 评估阶段只更新历史，不再学习

        times["random"].append(t_rand)
        times["average"].append(t_avg)
        times["fair"].append(t_fair)
        times["euro_afl"].append(t_rl)
        times["ours"].append(t_ours)

    return times


# --------------------------------------------------------------------------- #
# 总预算敏感性 / 参数敏感性实验
# --------------------------------------------------------------------------- #
def run_budget_sensitivity(accuracy: np.ndarray, sim: SimConfig,
                           target_accuracy: float, budgets: List[float],
                           rng: np.random.Generator) -> Dict[float, List[float]]:
    """不同总预算 B 下本文方法（IRBAI + IRBSB）的逐轮时延。

    每个 B 都从同一个客户端采样序列出发（而非顺序共享一条流），因此：
        - 不同 B 之间的差异完全来自预算本身，对比更公平；
        - 结果不再依赖 ``budgets`` 列表的顺序与长度，可复现性更强。
    """
    base = int(rng.integers(0, 2 ** 31 - 1))
    result = {}
    for B in budgets:
        sub = SimConfig(**{**sim.__dict__})
        sub.total_budget = B
        sub.irbai_Q = target_accuracy
        sub_rng = np.random.default_rng(base)
        round_budgets = alg.inter_ours(accuracy, B, sub.num_rounds, sub.irbai_psi, target_accuracy)
        times = []
        for r in range(sub.num_rounds):
            cp = sample_clients(sub, sub_rng)
            t, _, _ = alg.intra_ours(float(round_budgets[r]), cp.d_i, cp.c_i,
                                     cp.a_i, cp.sigma, cp.T_up)
            times.append(t)
        result[B] = times
    return result


def run_psi_sensitivity(accuracy: np.ndarray, sim: SimConfig,
                        target_accuracy: float, psis: List[float],
                        rng: np.random.Generator) -> Dict[float, List[float]]:
    """不同 ψ 下本文方法的逐轮时延（IRBAI 参数敏感性）。

    与预算敏感性同理：每个 ψ 共享同一客户端采样序列，差异只来自 ψ 本身。
    """
    base = int(rng.integers(0, 2 ** 31 - 1))
    result = {}
    for psi in psis:
        sub_rng = np.random.default_rng(base)
        round_budgets = alg.inter_ours(accuracy, sim.total_budget, sim.num_rounds,
                                       psi, target_accuracy)
        times = []
        for r in range(sim.num_rounds):
            cp = sample_clients(sim, sub_rng)
            t, _, _ = alg.intra_ours(float(round_budgets[r]), cp.d_i, cp.c_i,
                                     cp.a_i, cp.sigma, cp.T_up)
            times.append(t)
        result[psi] = times
    return result
