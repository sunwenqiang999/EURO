# budget_allocator/environment.py

import torch
import numpy as np
from config import NUM_CLIENTS, N_COMM_ROUNDS, O_LOCAL_EPOCHS, ZETA_MODEL_SIZE, CLIENT_PARAMS


class FLSessionsEnv:
    """
    模拟联邦学习(FL)轮内环境。
    这个环境接收一个预算分配动作，并计算出该轮次的总等待时间，
    然后返回负的等待时间作为奖励。
    """

    def __init__(self, client_params):
        self.num_clients = len(client_params)
        self.client_params = client_params
        self.client_ids = list(client_params.keys())

    def calculate_waiting_time(self, budget_total_p_c, allocation_weights_omega):
        """
        根据给定的总预算和分配权重，计算并返回总等待时间 T_c。
        """
        if len(allocation_weights_omega) != self.num_clients:
            raise ValueError("Allocation weights must match the number of clients.")

        client_total_times_T_k_c = []

        for i, client_id in enumerate(self.client_ids):
            params = self.client_params[client_id]
            omega_k_c = allocation_weights_omega[i]

            # 1. 计算客户端 k 在轮次 c 的报酬 p_k,c
            # 公式 (15): p_k,c = p_c * ω_k,c
            p_k_c = budget_total_p_c * omega_k_c

            # 2. 计算客户端 k 的最优 CPU 频率 ξ*_k,c
            # 公式 (17): ξ*_k,c = p_k,c / (2 * N * σ * m_k * d_k * α_k)
            # 注意: 论文中的 σ (o) 和 N 在此处用 config 中的常量代替
            numerator = p_k_c
            denominator = 2 * N_COMM_ROUNDS * O_LOCAL_EPOCHS * params["m_k"] * params["d_k"] * params["alpha_k"]

            if denominator == 0:
                # 避免除以零，如果没有计算需求，则 CPU 频率为最小值
                xi_k_c_star = params["xi_min"]
            else:
                xi_k_c_star = numerator / denominator

            # 限制 CPU 频率在硬件范围内
            xi_k_c = np.clip(xi_k_c_star, params["xi_min"], params["xi_max"])

            # 3. 计算客户端 k 的计算时间 T_k,c^cmp
            # 公式 (5): T_k,c^cmp = (N * σ * m_k * d_k) / ξ_k,c
            T_k_c_cmp = (N_COMM_ROUNDS * O_LOCAL_EPOCHS * params["m_k"] * params["d_k"]) / xi_k_c

            # 4. 计算客户端 k 的模型上传时间 T_k,c^com
            # 公式 (6): T_k,c^com = (N * ζ) / B_k,c
            # 注意: ζ (zeta) 是模型大小，B_k,c 是带宽
            T_k_c_com = (N_COMM_ROUNDS * ZETA_MODEL_SIZE) / params["B_k"]

            # 5. 计算客户端 k 的总时间 T_k,c
            # 公式 (4): T_k,c = T_k,c^cmp + T_k,c^com
            T_k_c = T_k_c_cmp + T_k_c_com
            client_total_times_T_k_c.append(T_k_c)

        # 6. 计算轮次 c 的总等待时间 T_c
        # 公式 (3): T_c = max(T_k,c) for k in K
        T_c = np.max(client_total_times_T_k_c)

        return T_c

    def step(self, budget_total_p_c, action):
        """
        执行一个步骤：接收预算和动作，返回奖励。
        - action: 一个包含 K 个预算分配比例的 numpy 数组，和为 1。
        - returns: 奖励 (reward)
        """
        # 计算等待时间
        waiting_time_T_c = self.calculate_waiting_time(budget_total_p_c, action)

        # 7. 计算轮内奖励 r_c^intra
        # 公式 (19): r_c^intra = -T_c
        reward = -waiting_time_T_c

        return reward