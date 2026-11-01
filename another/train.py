# budget_allocator/train.py

import torch
import numpy as np
import matplotlib.pyplot as plt

from another.dqn_agent import IntraBMA_Agent
from another.environment import FLSessionsEnv
from another.config import (
    TOTAL_TRAINING_ROUNDS, NUM_CLIENTS, CLIENT_PARAMS, BUDGET_MIN, BUDGET_MAX,
    EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE_FREQ
)


def main():
    # 初始化环境和智能体
    env = FLSessionsEnv(CLIENT_PARAMS)
    agent = IntraBMA_Agent(num_clients=NUM_CLIENTS)

    rewards_history = []
    losses_history = []

    print("--- Starting Training for Intra-Session Budget Allocation ---")

    for i_round in range(TOTAL_TRAINING_ROUNDS):
        # 1. 模拟从 InterBMA 获得本轮的总预算 p_c
        # 这里我们随机生成一个预算，您可以替换为您已知的每轮预算
        current_round_budget = np.random.uniform(BUDGET_MIN, BUDGET_MAX)
        state = current_round_budget

        # 2. 根据 Epsilon-Greedy 策略选择动作 (预算分配比例)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * agent.steps_done / EPS_DECAY)
        agent.steps_done += 1

        action_omega = agent.select_action(state, eps_threshold)

        # 3. 环境执行动作并返回奖励
        # 对应 Algorithm 1, lines 6-9
        reward = env.step(current_round_budget, action_omega)
        rewards_history.append(reward)

        # 4. 将经验存入回放缓冲区
        # 对于 IntraBMA, state 和 next_state 概念不适用，我们将 (budget, action, reward) 存入
        agent.memory.push(state, action_omega, reward, None)

        # 5. 优化模型
        loss = agent.optimize_model()
        if loss is not None:
            losses_history.append(loss)

        # 6. 定期更新目标网络
        # 对应 Algorithm 1, line 16
        if i_round % TARGET_UPDATE_FREQ == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if (i_round + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Round [{i_round + 1}/{TOTAL_TRAINING_ROUNDS}], Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {eps_threshold:.4f}")

    print("--- Training Finished ---")

    # 绘制结果图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Reward per Round (Negative Waiting Time)")
    plt.xlabel("Round")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(losses_history)
    plt.title("DQN Loss per Optimization Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()

    # 保存训练好的模型
    torch.save(agent.policy_net.state_dict(), "intra_bma_dqn_model.pth")
    print("Model saved to intra_bma_dqn_model.pth")


if __name__ == "__main__":
    main()