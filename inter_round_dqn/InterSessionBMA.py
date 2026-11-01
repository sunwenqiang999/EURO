import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from inter_round_dqn.dqn import DQN
from inter_round_dqn.replay import ReplayBuffer


class InterSessionBMA:
    """Inter-session Budget Management Agent using DQN"""

    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.1,
                 gamma=1.0,  # 论文中设置为1
                 epsilon=0.5,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 tau=1,  # target network update frequency
                 memory_capacity=10000,
                 batch_size=4,
                 C_prime=3):

        self.C_prime = C_prime
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size

        # 神经网络
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(memory_capacity)

        # 计数器
        self.step_count = 0

        # 初始化目标网络
        self.update_target_network()

        # 历史记录
        self.history = {
            'payments': [],  # {p_k,c}
            'times': [],  # {T_k,c}
            'accuracies': [],  # F(w_c)
            'sessions': [],

        }

    def build_state(self, current_session, remaining_budget):
        """构建轮间状态 s_inter_c"""
        state = []

        # 当前会话索引
        state.append(current_session)

        # 剩余预算
        state.append(remaining_budget)

        # 历史准确率（最近C'个会话）
        recent_accuracies = self.history['accuracies'][-self.C_prime:]
        while len(recent_accuracies) < self.C_prime:
            recent_accuracies = [0.0] + recent_accuracies  # 用0填充
        state.extend(recent_accuracies)

        # 历史支付（最近C'个会话，假设K个客户端）
        recent_payments = self.history['payments'][-self.C_prime:]
        while len(recent_payments) < self.C_prime:
            recent_payments = [[0.0] * 20] + recent_payments  # 假设20个客户端，用0填充
        for payments in recent_payments:
            state.extend(payments[:20])  # 限制为20个客户端

        # 历史时间（最近C'个会话）
        recent_times = self.history['times'][-self.C_prime:]
        while len(recent_times) < self.C_prime:
            recent_times = [[0.0] * 20] + recent_times  # 假设10个客户端，用0填充
        for times in recent_times:
            state.extend(times[:20])  # 限制为20个客户端

        return np.array(state, dtype=np.float32)

    def select_action(self, state):
        """使用ε-贪心策略选择动作"""
        if np.random.random() < self.epsilon:
            # 随机选择动作
            return np.random.randint(0, self.action_dim)
        else:
            # 贪心选择动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def action_to_budget(self, action, total_budget, current_session, max_sessions, accuracy, target_accuracy):
        """将离散动作转换为预算分配"""
        # 考虑剩余会话数的预算分配策略
        remaining_sessions = max_sessions - current_session
        allocated_budget = 0.0
        if accuracy < target_accuracy:
            # 简单的线性映射策略
            # 动作空间可以表示为预算分配的百分比
            budget_ratio = (action + 1) / self.action_dim
            if remaining_sessions > 0:
                # 保守分配，为后续会话保留预算
                max_allocation = total_budget / remaining_sessions * 3  # 允许一定程度的超额分配
            else:
                max_allocation = total_budget

            allocated_budget = min(budget_ratio * max_allocation, total_budget)
            allocated_budget = max(allocated_budget, 150)
        else:
            # 目标准确率已达，分配全部预算
            allocated_budget = total_budget/remaining_sessions
        return  allocated_budget  # 确保最小预算

    def calculate_reward(self, current_accuracy, previous_accuracy, waiting_time, lambda_param=10):
        """计算轮间奖励 r_inter_c"""
        # 公式 (12): r_inter_c = λ[F(w_c) - F(w_c-1)] - T_c
        accuracy_improvement = current_accuracy - previous_accuracy
        reward = lambda_param * accuracy_improvement - waiting_time
        return reward

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)

    def update_history(self, payments, times, accuracy, session):
        """更新历史记录"""
        self.history['payments'].append(payments)
        self.history['times'].append(times)
        self.history['accuracies'].append(accuracy)
        self.history['sessions'].append(session)

    def train(self):
        """训练DQN网络"""
        if len(self.memory) < self.batch_size:
            return

        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 下一状态的最大Q值
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.tau == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.step_count
        }, filepath)
