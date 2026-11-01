# budget_allocator/config.py
import torch

# --- 训练和智能体超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_TRAINING_ROUNDS = 5000 # 总训练轮次（Episodes）
LEARNING_RATE = 1e-4           # DQN学习率
GAMMA = 0.99                   # 奖励折扣因子 (论文中IntraBMA设为1，但DQN通常使用折扣) [cite: 252]
BATCH_SIZE = 128               # 从回放缓冲区中采样的批量大小
REPLAY_BUFFER_SIZE = 10000     # 回放缓冲区的最大容量
TARGET_UPDATE_FREQ = 100       # 目标网络更新频率

# Epsilon-Greedy 策略参数
EPS_START = 0.9         # Epsilon 初始值
EPS_END = 0.05          # Epsilon 最终值
EPS_DECAY = 1000        # Epsilon 衰减率

# --- 联邦学习环境参数 ---
NUM_CLIENTS = 10              # 参与方（Clients/DOs）的数量 K

# 每轮的总预算范围 (模拟 InterBMA 的输出 p_c)
# 在您的应用中，您可以直接输入每轮的已知预算
BUDGET_MIN = 50.0
BUDGET_MAX = 100.0

# --- 物理和模型参数 (根据论文公式) ---
# 这些值需要根据您的实际场景或预先测量来设定 [cite: 181]

# N: 每轮训练中的总通信回合数
# o: 每个通信回合中的本地训练 epoch 数
# ζ: 本地模型大小（bits）
N_COMM_ROUNDS = 10
O_LOCAL_EPOCHS = 5
ZETA_MODEL_SIZE = 1.2e6  # e.g., 1.2 Mbits

# 客户端特定参数 (为简化，此处为每个客户端随机生成)
# 在实际应用中，这些应该是每个客户端的已知或估计值
CLIENT_PARAMS = {
    f"client_{k}": {
        # m_k: 处理每 bit 数据所需的 CPU 周期数
        "m_k": torch.randint(500, 1000, (1,)).item(),
        # d_k: 执行单个本地 epoch 所需的总数据位数
        "d_k": torch.randint(int(5e5), int(1e6), (1,)).item(),
        # alpha_k: 计算芯片的有效电容系数
        "alpha_k": torch.normal(mean=1e-27, std=1e-28).item(),
        # B_k,c: 客户端 k 的可用网络带宽 (bits/sec)
        "B_k": torch.randint(int(1e6), int(5e6), (1,)).item(),
        # ξ_k^min, ξ_k^max: 最小/最大 CPU 频率
        "xi_min": 1e9, # 1 GHz
        "xi_max": 3e9, # 3 GHz
    }
    for k in range(NUM_CLIENTS)
}

print(f"Running on device: {DEVICE}")
print(f"Number of clients: {NUM_CLIENTS}")