# budget_allocator/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

from config import DEVICE, LEARNING_RATE, GAMMA, BATCH_SIZE, REPLAY_BUFFER_SIZE, NUM_CLIENTS

# 定义经验回放的 Transition 结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer:
    """固定大小的缓冲区来存储经验元组"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一个 transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """从 memory 中随机采样一个 batch 的 transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    DQN 网络模型 (IntraBMA Policy Network)
    输入: state (该轮次的总预算 p_c)
    输出: 每个客户端的预算分配比例的 Q-values
    """

    def __init__(self, num_outputs):
        super(DQN, self).__init__()
        # 状态是单个值：总预算 p_c
        self.layer1 = nn.Linear(1, 128)
        self.layer2 = nn.Linear(128, 128)
        # 输出是每个客户端的分配权重，所以输出维度是客户端数量
        self.output_layer = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # 输出原始 Q-values
        return self.output_layer(x)


class IntraBMA_Agent:
    """Intra-session Budget Management Agent"""

    def __init__(self, num_clients):
        self.num_clients = num_clients

        # 建立策略网络和目标网络
        self.policy_net = DQN(num_clients).to(DEVICE)
        self.target_net = DQN(num_clients).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不进行训练

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.steps_done = 0

    def select_action(self, state, eps_threshold):
        """
        根据 Epsilon-Greedy 策略选择一个动作。
        state: 当前轮次的总预算 p_c
        """
        sample = random.random()
        if sample > eps_threshold:
            # 利用 (Exploitation)
            with torch.no_grad():
                # state 是一个标量，需要转换为 [1, 1] 的 tensor
                state_tensor = torch.tensor([[state]], device=DEVICE, dtype=torch.float32)
                # 从 policy_net 获取 Q-values 并应用 softmax 得到分配比例
                action_q_values = self.policy_net(state_tensor)
                action_probs = F.softmax(action_q_values, dim=1).squeeze()
                return action_probs.cpu().numpy()
        else:
            # 探索 (Exploration): 生成一个随机的、和为1的分配比例向量
            random_weights = np.random.rand(self.num_clients)
            return random_weights / np.sum(random_weights)

    def optimize_model(self):
        """
        从回放缓冲区中采样一个 batch 进行训练，更新 policy_net。
        对应论文中的 Algorithm 1, lines 10-15。
        """
        if len(self.memory) < BATCH_SIZE:
            return None  # 如果缓冲区数据不够，则不训练

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 将 batch 数据转换为 PyTorch Tensors
        state_batch = torch.tensor(batch.state, device=DEVICE, dtype=torch.float32).view(-1, 1)
        action_batch = torch.tensor(np.array(batch.action), device=DEVICE, dtype=torch.float32)
        reward_batch = torch.tensor(batch.reward, device=DEVICE, dtype=torch.float32).view(-1, 1)

        # 1. 计算 Q(s, a)
        # policy_net 输出所有动作的Q值，然后我们根据实际采取的动作 action_batch 选出对应的Q值
        # action_batch 是一个 [batch_size, num_clients] 的概率分布，不能直接作为索引
        # 我们需要计算 Q(s_t, a_t) = sum(policy_net(s_t) * a_t)
        state_action_values = self.policy_net(state_batch)
        state_action_values = torch.sum(state_action_values * action_batch, dim=1).unsqueeze(1)

        # 2. 计算 V(s_{t+1}) for all next states.
        # 对于 IntraBMA，它的任务在轮内就结束了，没有 "next_state"。
        # 奖励是在轮次结束时一次性获得的。因此，TD-target y 就是 r。
        # 公式 (21) y = r + γ * max_a' Q(s', a'; θ_hat) 简化为 y = r
        # 对应论文 [cite: 252] IntraBMA 的 discount factor 为 1，但其生命周期仅为一轮。
        expected_state_action_values = reward_batch

        # 3. 计算损失
        # 公式 (20): L(θ) = 1/2 * E[(y - Q(s,a;θ))^2]
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # 4. 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 对梯度进行裁剪，防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()