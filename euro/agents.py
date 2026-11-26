"""强化学习智能体：轮内 IntraBMA 与轮间 InterBMA (EURO-AFL)。

对应中期答辩 PPT 中的「强化学习分配 (EURO-AFL)」对比方法：
    第一阶段 (InterBMA)：DQN 根据当前状态预测本轮总预算。
    第二阶段 (IntraBMA)：另一 DQN 在给定预算下确定各客户端预算比例。

相比原始实现的关键修复：
    - 学习率从 1.0（对 Adam 极不稳定）降为 1e-3。
    - 轮内状态纳入客户端异构特征，使智能体能针对客户端条件差异化决策。
    - 奖励量纲统一；折扣因子、目标网络更新频率合理化。
    - **可复现性**：智能体内部不再使用全局 ``random`` / ``np.random`` / torch
      全局 RNG。构造时传入 ``seed`` 后，经验回放采样、ε-探索、网络权重初始化
      全部走各自独立的随机流，因此「单独跑某个实验」与「批量跑多个实验」
      得到的结果完全一致。
"""

from __future__ import annotations

import random
from collections import deque
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --------------------------------------------------------------------------- #
# 随机性工具：保证智能体拥有彼此独立且可复现的随机流
# --------------------------------------------------------------------------- #
@contextmanager
def torch_seed(seed: int | None):
    """临时将 torch 全局 RNG 置为 ``seed``，退出时恢复原状态。

    用于确定性地初始化网络权重（``nn.Linear`` 的默认初始化只能走 torch 全局
    RNG，无法直接注入 Generator），同时不污染调用方的全局随机状态——
    这正是原实现中「多创建一个智能体就会改变后续所有随机数」的根因。
    """
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _make_py_rng(seed: int | None):
    """返回 ``random.Random`` 实例；seed 为 None 时退化为全局 random 模块。"""
    return random.Random(seed) if seed is not None else random


def _make_np_rng(seed: int | None):
    """返回 ``np.random.Generator``；seed 为 None 时退化为全局 np.random。"""
    return np.random.default_rng(seed) if seed is not None else np.random


def _randint(rng, high: int) -> int:
    """兼容 Generator(``integers``) 与 legacy ``np.random``(``randint``)。"""
    if hasattr(rng, "integers"):
        return int(rng.integers(0, high))
    return int(rng.randint(0, high))


class _MLP(nn.Module):
    """通用多层感知机 Q 网络。"""

    def __init__(self, in_dim: int, out_dim: int, hidden=(128, 128, 64)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区。

    Args:
        capacity: 缓冲区容量。
        seed: 采样随机种子。给定后使用独立的 ``random.Random``，
              不再消耗全局 ``random`` 状态。
    """

    def __init__(self, capacity: int = 10000, seed: int | None = None):
        self.buffer: deque = deque(maxlen=capacity)
        self._rng = _make_py_rng(seed)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = self._rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# 轮内策略：IntraBMA
# --------------------------------------------------------------------------- #
class IntraBMAAgent:
    """轮内预算分配智能体（EURO-AFL 第二阶段），纯强化学习版本。

    与旧版的区别：
        - **不再使用时延公式的可微损失**（旧实现见本类末尾的注释块）。智能体
          只能通过环境返回的「本轮完成时延」这一奖励信号学习，不知道
          ``T = h / sqrt(b / 2κ) + T_up`` 这条公式。
        - 采用 REINFORCE 策略梯度：策略网络给每个客户端打分（logit），训练时
          在 logit 上加高斯噪声探索，softmax 后得到预算权重。
        - 打分网络对每个客户端**共享参数**（逐客户端特征 -> 标量分数），天然对
          客户端顺序不敏感，样本效率远高于旧版「81 维拼接向量 -> 20 维输出」。

    状态（逐客户端一行，形状 (K, 5)）：
        [d_i/mean(d), c_i/mean(c), a_i/mean(a), T_up_i/mean(T_up), 预算归一化]
    动作：K 维 logit -> softmax -> 预算权重（加极小地板避免 0 支付）。
    奖励：``1 - T_max / T_uniform``，T_uniform 为均分预算的时延，仅用于量纲
          归一化，使奖励不随数据集/预算量级漂移。

    Args:
        seed: 智能体专属随机种子（网络初始化、探索噪声各走独立随机流）。
    """

    FEAT_DIM = 5

    def __init__(self, num_clients: int, learning_rate: float = 3e-3,
                 explore_std: float = 0.5, batch_size: int = 32,
                 floor: float = 0.02, seed: int | None = None):
        self.num_clients = num_clients
        self.explore_std = explore_std
        self.batch_size = batch_size
        self.floor = floor
        self.seed = seed
        self._np_rng = _make_np_rng(None if seed is None else seed + 1)  # 探索噪声

        with torch_seed(None if seed is None else seed + 2):
            self.policy = _MLP(self.FEAT_DIM, 1, hidden=(64, 64))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.memory: list = []            # on-policy 小批量缓冲
        self.last_logits: np.ndarray | None = None
        self.training_step = 0

    # ------------------------------------------------------------------ #
    def build_state(self, round_budget: float, cp) -> np.ndarray:
        """逐客户端特征矩阵 (K, 5)。"""
        feats = [cp.d_i / (np.mean(cp.d_i) + 1e-9),
                 cp.c_i / (np.mean(cp.c_i) + 1e-9),
                 cp.a_i / (np.mean(cp.a_i) + 1e-9),
                 cp.T_up / (np.mean(cp.T_up) + 1e-9),
                 np.full(len(cp.d_i), round_budget / 1000.0)]
        return np.stack(feats, axis=1).astype(np.float32)

    def _logits(self, state) -> torch.Tensor:
        x = torch.as_tensor(state, dtype=torch.float32)
        return self.policy(x).squeeze(-1)

    def _weights_from_logits(self, logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        w = np.exp(z)
        w = w / w.sum()
        w = w + self.floor / self.num_clients
        return w / w.sum()

    def set_eval(self) -> None:
        """评估模式：关闭探索噪声，使用确定性策略。"""
        self.explore_std = 0.0

    def act(self, state: np.ndarray, round_budget: float):
        """返回 (weights, payments)。训练期在 logit 上加高斯噪声探索。"""
        with torch.no_grad():
            mu = self._logits(state).numpy()
        if self.explore_std > 0.0:
            z = mu + self.explore_std * self._np_rng.standard_normal(len(mu))
        else:
            z = mu
        self.last_logits = z.astype(np.float32)
        weights = self._weights_from_logits(z)
        return weights, weights * round_budget

    # ------------------------------------------------------------------ #
    def learn(self, state, logits, reward: float) -> float | None:
        """记录一条 (状态, 实际执行的带噪 logit, 奖励)，攒满一批后做一次策略梯度更新。"""
        self.memory.append((np.asarray(state, dtype=np.float32),
                            np.asarray(logits, dtype=np.float32),
                            float(reward), float(self.explore_std)))
        if len(self.memory) < self.batch_size:
            return None
        batch, self.memory = self.memory, []
        states = torch.as_tensor(np.stack([b[0] for b in batch]))     # (B, K, F)
        z = torch.as_tensor(np.stack([b[1] for b in batch]))          # (B, K)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        std = torch.as_tensor([b[3] for b in batch], dtype=torch.float32).clamp_min(1e-3)

        adv = (r - r.mean()) / (r.std() + 1e-6)                       # 标准化优势
        mu = self.policy(states).squeeze(-1)                          # (B, K)
        # 高斯策略的对数概率（省去与 mu 无关的常数项）。
        logp = -((z - mu) ** 2).sum(dim=1) / (2.0 * std ** 2)
        loss = -(adv * logp / self.num_clients).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.training_step += 1
        return loss.item()

    # ------------------------------------------------------------------ #
    # 旧实现：直接优化时延公式的可微损失（已停用，仅保留备查）。
    # 该做法让网络「看到」了 T = h / sqrt(b / 2κ) + T_up 这条公式并对其求导，
    # 不属于标准的强化学习（只依赖环境奖励），因此注释掉不再使用。
    # 若要恢复：取消注释本段，并在 remember() 中同时存入 d/c/a/sigma/T_up。
    #
    #     def replay(self):
    #         """可微重放：最小化批次的软化最大完成时间。"""
    #         if len(self.memory) < self.batch_size:
    #             return None
    #         batch = self._py_rng.sample(self.memory, self.batch_size)
    #         states = torch.FloatTensor(np.stack([b[0] for b in batch]))
    #         d = torch.FloatTensor(np.stack([b[1] for b in batch]))
    #         c = torch.FloatTensor(np.stack([b[2] for b in batch]))
    #         a = torch.FloatTensor(np.stack([b[3] for b in batch]))
    #         sigma = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1)
    #         tup = torch.FloatTensor(np.stack([b[5] for b in batch]))
    #         budget = torch.FloatTensor([b[6] for b in batch]).unsqueeze(1)
    #
    #         weights = self._weights_from_logits(self.policy(states))
    #         payments = weights * budget                     # (B, K)
    #         h = sigma * c * d                                # 工作量
    #         kappa = a * h
    #         freq = torch.sqrt(torch.clamp(payments / (2.0 * kappa + 1e-9), min=1e-9))
    #         times = h / (freq + 1e-9) + tup                  # (B, K) 完成时间
    #         # 软化 max：softmax 加权平均逼近最大值，可微。
    #         beta = self.softmin_beta
    #         soft_max = (times * torch.softmax(beta * times, dim=1)).sum(dim=1)
    #         loss = soft_max.mean()
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon *= self.epsilon_decay
    #         self.training_step += 1
    #         return loss.item()
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path))


# --------------------------------------------------------------------------- #
# 轮间 DQN：InterBMA
# --------------------------------------------------------------------------- #
class InterBMAAgent:
    """轮间预算分配智能体：决定每一会话（轮）的预算比例。

    状态：[会话索引归一化, 剩余预算归一化, 最近 C' 轮的 准确率 / 总支付 / 轮时延]
          （只用标量统计量：客户端参数每轮重抽，逐客户端历史没有预测意义）
    动作：20 个离散预算比例档位。
    奖励：λ·准确率提升/100 - 轮时延/time_scale（time_scale 由预训练标定）。

    训练方式：需先经 ``simulator`` 中的多 episode 预训练（每个 episode 重新采样
    客户端），评估阶段调用 ``set_eval()`` 关闭探索并不再更新网络。

    Args:
        seed: 智能体专属随机种子，控制 Q 网络初始化、ε-greedy 与回放采样。
    """

    def __init__(self, num_clients: int = 20, action_dim: int = 20,
                 c_prime: int = 3, learning_rate: float = 1e-3,
                 gamma: float = 0.99, epsilon: float = 0.9,
                 epsilon_decay: float = 0.9995, epsilon_min: float = 0.05,
                 batch_size: int = 64, memory_size: int = 20000,
                 target_update_freq: int = 100, seed: int | None = None):
        self.num_clients = num_clients
        self.action_dim = action_dim
        self.c_prime = c_prime
        # 状态维度：会话索引 + 剩余预算 + C'×(准确率 + 总支付 + 轮时延)
        self.state_dim = 2 + c_prime * 3
        self.time_scale = 1.0     # 时延归一化常数，由预训练标定

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.seed = seed
        self._np_rng = _make_np_rng(None if seed is None else seed + 1)  # ε-greedy

        # 两个网络的初始化走同一条独立 torch 随机流，不污染全局状态。
        with torch_seed(None if seed is None else seed + 2):
            self.q_network = _MLP(self.state_dim, action_dim, hidden=(64, 64))
            self.target_network = _MLP(self.state_dim, action_dim, hidden=(64, 64))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.memory = ReplayBuffer(memory_size, seed=seed)
        self.training_step = 0
        self.history = {"payments": [], "times": [], "accuracies": []}

    def build_state(self, session: int, remaining_budget: float,
                    max_sessions: int, total_budget: float) -> np.ndarray:
        state = [session / max(max_sessions, 1), remaining_budget / max(total_budget, 1e-9)]

        accs = self.history["accuracies"][-self.c_prime:]
        accs = [0.0] * (self.c_prime - len(accs)) + list(accs)
        state.extend([a / 100.0 for a in accs])

        pays = self.history["payments"][-self.c_prime:]
        pays = [0.0] * (self.c_prime - len(pays)) + list(pays)
        pay_ref = total_budget / max(max_sessions, 1)
        state.extend([p / max(pay_ref, 1e-9) for p in pays])

        times = self.history["times"][-self.c_prime:]
        times = [0.0] * (self.c_prime - len(times)) + list(times)
        state.extend([t / self.time_scale for t in times])

        return np.asarray(state, dtype=np.float32)

    def select_action(self, state: np.ndarray) -> int:
        if self.epsilon > 0.0 and self._np_rng.random() < self.epsilon:
            return _randint(self._np_rng, self.action_dim)
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0)
            return int(self.q_network(st).argmax().item())

    def action_to_budget(self, action: int, remaining_budget: float,
                         session: int, max_sessions: int,
                         accuracy: float, target_accuracy: float) -> float:
        """将离散动作映射为本轮预算（含剩余预算与最小预算约束）。"""
        remaining_sessions = max(max_sessions - session, 1)
        avg_remaining = remaining_budget / remaining_sessions
        if accuracy < target_accuracy:
            ratio = (action + 1) / self.action_dim
            # 允许早期适度超额分配（上限为均值的 3 倍）。
            allocation = ratio * avg_remaining * 3.0
            allocation = max(min(allocation, remaining_budget), 150.0)
        else:
            allocation = avg_remaining
        return min(allocation, remaining_budget)

    def calculate_reward(self, cur_acc: float, prev_acc: float,
                         waiting_time: float, lam: float = 1.0) -> float:
        return lam * (cur_acc - prev_acc) / 100.0 - waiting_time / self.time_scale

    def update_history(self, payments, times, accuracy):
        """记录标量统计量：本轮总支付、本轮时延（最慢客户端）、准确率。"""
        self.history["payments"].append(float(np.sum(payments)))
        self.history["times"].append(float(np.max(times)))
        self.history["accuracies"].append(float(accuracy))

    def reset_history(self) -> None:
        self.history = {"payments": [], "times": [], "accuracies": []}

    def set_eval(self) -> None:
        """评估模式：关闭 ε 探索（评估阶段不再更新网络）。"""
        self.epsilon = 0.0

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return loss.item()

    def save(self, path: str):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
