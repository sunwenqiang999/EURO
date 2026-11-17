"""联邦学习轮次时延仿真器。

核心原则（修复原代码参数错位 bug）：
    - 同一轮内，所有对比方法共享同一组客户端异构参数，保证横向比较公平。
    - 通过显式的 numpy Generator 控制随机性，保证可复现。

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
from .config import SimConfig, sample_clients


# --------------------------------------------------------------------------- #
# 轮内对比实验
# --------------------------------------------------------------------------- #
def _warmup_intra_agent(agent: IntraBMAAgent, round_budgets, sim: SimConfig,
                        rng: np.random.Generator, epochs: int = 30) -> None:
    """对轮内 DQN 做若干 epoch 预训练，使其策略优于随机/均匀基线。

    使用独立的随机流，避免污染评估阶段的客户端采样序列。
    """
    warm_rng = np.random.default_rng(sim.seed + 100)
    for _ in range(epochs):
        for r in range(sim.num_rounds):
            budget = float(round_budgets[r])
            cp = sample_clients(sim, warm_rng)
            state = agent.build_state(budget, cp)
            agent.remember(state, cp, budget)
            agent.replay()


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

    agent = IntraBMAAgent(sim.num_clients)
    # 预训练：让轮内策略网络在多个 epoch 上学习，收敛到优于朴素基线。
    _warmup_intra_agent(agent, round_budgets, sim, rng, epochs=30)
    agent.epsilon = 0.0  # 评估阶段使用已学习的确定性策略
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
        agent.remember(state, cp, budget)
        for _ in range(4):
            agent.replay()

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
                         rng: np.random.Generator) -> Dict[str, List[float]]:
    """固定轮内 IRBSB 解析分配，比较不同轮间策略的逐轮时延。

    Returns:
        {method: [逐轮时延]}，method ∈
        {random, average, euro_afl, ours}
        （对应 PPT 轮间消融：IRBAI / InterBMA / Random / FedAvg）
    """
    sim.irbai_Q = target_accuracy
    budgets_ours = alg.inter_ours(accuracy, sim.total_budget, sim.num_rounds,
                                  sim.irbai_psi, target_accuracy)
    budgets_avg = alg.inter_average(sim.total_budget, sim.num_rounds)
    budgets_rand = alg.inter_random(sim.total_budget, sim.num_rounds, rng)

    inter_agent = InterBMAAgent(num_clients=sim.num_clients)
    remaining = sim.total_budget

    times = {k: [] for k in ("random", "average", "euro_afl", "ours")}

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

        prev_acc = inter_agent.history["accuracies"][-1] if inter_agent.history["accuracies"] else 0.0
        reward = inter_agent.calculate_reward(acc, prev_acc, t_rl)
        remaining -= float(np.sum(rl_payments))
        inter_agent.update_history(rl_payments, rl_tlist, acc)
        next_state = inter_agent.build_state(r + 1, remaining, sim.num_rounds, sim.total_budget)
        done = (r >= sim.num_rounds - 1) or (remaining <= 0)
        inter_agent.remember(state, action, reward, next_state, done)
        for _ in range(4):
            inter_agent.replay()

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
                         rng: np.random.Generator) -> Dict[str, List[float]]:
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
    intra_agent = IntraBMAAgent(sim.num_clients)
    _warmup_intra_agent(intra_agent, budgets_ours, sim, rng, epochs=30)
    intra_agent.epsilon = 0.0
    # 轮间 RL 智能体（EURO-AFL 第一阶段）。
    inter_agent = InterBMAAgent(num_clients=sim.num_clients)
    remaining = sim.total_budget

    times = {k: [] for k in ("random", "average", "fair", "euro_afl", "ours")}

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
        intra_state = intra_agent.build_state(rl_budget, cp)
        _, rl_payments = intra_agent.act(intra_state, rl_budget)
        rl_tlist = alg.completion_times(rl_payments, cp.d_i, cp.c_i, cp.a_i, cp.sigma, cp.T_up)
        t_rl = float(np.max(rl_tlist))

        prev_acc = inter_agent.history["accuracies"][-1] if inter_agent.history["accuracies"] else 0.0
        reward = inter_agent.calculate_reward(acc, prev_acc, t_rl)
        remaining -= float(np.sum(rl_payments))
        inter_agent.update_history(rl_payments, rl_tlist, acc)
        next_state = inter_agent.build_state(r + 1, remaining, sim.num_rounds, sim.total_budget)
        done = (r >= sim.num_rounds - 1) or (remaining <= 0)
        inter_agent.remember(state, action, reward, next_state, done)
        for _ in range(4):
            inter_agent.replay()

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
    """不同总预算 B 下本文方法（IRBAI + IRBSB）的逐轮时延。"""
    result = {}
    for B in budgets:
        sub = SimConfig(**{**sim.__dict__})
        sub.total_budget = B
        sub.irbai_Q = target_accuracy
        round_budgets = alg.inter_ours(accuracy, B, sub.num_rounds, sub.irbai_psi, target_accuracy)
        times = []
        for r in range(sub.num_rounds):
            cp = sample_clients(sub, rng)
            t, _, _ = alg.intra_ours(float(round_budgets[r]), cp.d_i, cp.c_i,
                                     cp.a_i, cp.sigma, cp.T_up)
            times.append(t)
        result[B] = times
    return result


def run_psi_sensitivity(accuracy: np.ndarray, sim: SimConfig,
                        target_accuracy: float, psis: List[float],
                        rng: np.random.Generator) -> Dict[float, List[float]]:
    """不同 ψ 下本文方法的逐轮时延（IRBAI 参数敏感性）。"""
    result = {}
    for psi in psis:
        round_budgets = alg.inter_ours(accuracy, sim.total_budget, sim.num_rounds,
                                       psi, target_accuracy)
        times = []
        for r in range(sim.num_rounds):
            cp = sample_clients(sim, rng)
            t, _, _ = alg.intra_ours(float(round_budgets[r]), cp.d_i, cp.c_i,
                                     cp.a_i, cp.sigma, cp.T_up)
            times.append(t)
        result[psi] = times
    return result
