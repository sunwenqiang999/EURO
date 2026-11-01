import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json


class IntraRoundBudgetAgent:
    """
    轮内预算分配代理（IntraBMA）
    负责在每个联邦学习通信轮次内将预算分配给不同的客户端
    """

    def __init__(self, num_clients, state_dim=None, action_dim=None,
                 learning_rate=0.1, epsilon=0.5, epsilon_decay=0.995,
                 epsilon_min=0.01, memory_size=10000, batch_size=8):
        """
        初始化轮内预算分配代理

        Args:
            num_clients: 客户端数量
            state_dim: 状态维度（如果为None，会自动计算）
            action_dim: 动作维度（等于客户端数量）
            learning_rate: 学习率
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            memory_size: 经验回放缓冲区大小
            batch_size: 批处理大小
        """
        self.num_clients = num_clients
        self.state_dim = state_dim or 2  # 默认状态维度
        self.action_dim = num_clients  # 动作维度等于客户端数量

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)

        # 构建神经网络
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 更新目标网络
        self.update_target_network()

        # 训练参数
        self.gamma = 1.0  # 折扣因子，论文中设为1
        self.update_target_freq = 1 # 目标网络更新频率
        self.training_step = 0

    def _build_network(self):
        """构建深度Q网络"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_state(self, round_budget, client_data_sizes, client_cpu_cycles,client_a,
                  client_t_up, historical_payments=None, historical_times=None
                  ):
        """
        构建当前状态

        Args:
            round_budget: 当前轮次的预算
            client_data_sizes: 客户端数据大小列表
            client_cpu_cycles: 客户端CPU周期列表
            historical_payments: 历史支付记录
            historical_times: 历史时间记录

        Returns:
            状态向量
        """
        # state = []
        #
        # # 当前轮次预算
        # state.append(round_budget)
        #
        # # 客户端数量
        # state.append(self.num_clients)
        #
        # # 客户端特征
        #
        # for i in range(self.num_clients):
        #     state.extend([
        #         client_data_sizes[i] if i < len(client_data_sizes) else 0,
        #         client_cpu_cycles[i] if i < len(client_cpu_cycles) else 0,
        #         client_t_up[i] if i < len(client_t_up) else 0
        #         ,client_a[i] if i < len(client_a) else 0
        #     ])
        #
        # 如果状态维度不匹配，进行填充或截断
        state = [round_budget, self.num_clients]
        # for i in range(self.num_clients):
        #     state.extend([
        #             client_data_sizes[i] / np.max(client_data_sizes) if len(client_data_sizes) else 0,
        #             client_cpu_cycles[i] / np.max(client_cpu_cycles) if len(client_cpu_cycles) else 0,
        #             client_t_up[i] / np.max(client_t_up) if len(client_t_up) else 0
        #             ,client_a[i] / np.max(client_a) if len(client_a) else 0
        #     ])
        return np.array(state, dtype=np.float32)

    def act(self, state, round_budget):
        """
        选择动作（预算分配比例）

        Args:
            state: 当前状态
            round_budget: 当前轮次预算

        Returns:
            动作（预算分配比例）
        """
        if np.random.random() <= self.epsilon:
            # 探索：随机选择动作
            weights = np.random.dirichlet(np.ones(self.num_clients))
        else:
            # 利用：使用神经网络预测
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            weights = torch.softmax(q_values, dim=1).squeeze().detach().numpy()

        # 确保权重和为1
        weights = (weights+0.5) / np.sum(weights+0.5)

        # 计算每个客户端的实际支付
        payments = weights * round_budget

        return weights, payments

    def calculate_optimal_cpu_cycles(self, payments, client_params,round_budget):
        """
        根据支付计算客户端最优CPU周期

        Args:
            payments: 每个客户端的支付
            client_params: 客户端参数字典

        Returns:
            最优CPU周期列表
        """
        optimal_cycles = []
        sigma = client_params.get('sigma', 3)# 本地训练轮数
        d = client_params['d_i']
        c = client_params['c_i']
        T = client_params['T_up']
        a = client_params['a_i']
        payments = payments * (c * d * sigma*a+ T)
        payments = (payments) / np.sum(payments)
        payments = round_budget * payments
        for i in range(self.num_clients):
            if i < len(payments):
                p_i = payments[i]
                d_i = client_params['d_i'][i] if i < len(client_params['d_i']) else 1
                c_i = client_params['c_i'][i] if i < len(client_params['c_i']) else 1
                a_i = client_params['a_i'][i] if i < len(client_params['a_i']) else 1
                # 根据论文公式(17)计算最优CPU周期
                denominator = 2 * sigma * d_i * c_i * a_i
                optimal_cycle = np.sqrt(p_i / denominator)
                optimal_cycles.append(optimal_cycle)  # 添加最优CPU周期
            else:
                print(f"Warning: client {i} has no payment")
        return optimal_cycles

    def calculate_training_time(self, cpu_cycles, client_params):
        """
        计算训练时间

        Args:
            cpu_cycles: CPU周期列表
            client_params: 客户端参数

        Returns:
            每个客户端的训练时间，最大训练时间
        """
        training_times = []
        sigma = client_params.get('sigma', 3)


        for i in range(self.num_clients):
            if i < len(cpu_cycles):
                # 计算时间组件
                d_i = client_params['d_i'][i] if i < len(client_params['d_i']) else 1
                c_i = client_params['c_i'][i] if i < len(client_params['c_i']) else 1
                T_up = client_params['T_up'][i] if i < len(client_params['T_up']) else 5

                # 计算时间（论文公式4-6）
                T_cmp = (sigma * d_i * c_i) / cpu_cycles[i]
                T_total = T_cmp + T_up

                training_times.append(T_total)
            else:
                print(f"Warning: client {i} has no CPU cycles")

        max_time = max(training_times) if training_times else 0
        return training_times, max_time

    def calculate_reward(self, training_time,payments,round_budget):
        """
        计算奖励（负的等待时间）

        Args:
            training_time: 训练时间

        Returns:
            奖励值
        """
        # 更合理的奖励设计
        reward = -training_time  # 主要奖励
        reward -= 0.1 * np.std(payments)  # 鼓励均匀分配
        reward -= 0.05 * np.abs(np.sum(payments) - round_budget)
        return reward

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)

        target_q_values = rewards + (self.gamma * torch.max(next_q_values, dim=1)[0] * ~dones)

        # 计算动作对应的Q值
        action_indices = torch.argmax(actions, dim=1)
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.training_step += 1
        if self.training_step % self.update_target_freq == 0:
            self.update_target_network()

    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)

    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']


