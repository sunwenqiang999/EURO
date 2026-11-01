from eu import *
from inter_budget_compute import round_inter_compute, round_intra_average, round_intra_random, round_intra_reforcement, \
    round_intra_compute, fair, round_inter_compute_cifar


class FederatedLearning:
    """联邦学习模拟器"""

    def __init__(self, num_clients, num_rounds):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.budget_agent = IntraRoundBudgetAgent(num_clients)

        # 模拟客户端参数
        self.client_params = {
            'sigma': 3,  # 本地训练轮数
            'd_i': np.random.rand(num_clients) * 2 + 0.5,
            'c_i': np.random.rand(num_clients) * 2 + 0.5,
            'a_i': np.random.rand(num_clients) * 2 + 0.5,
            'T_up': np.random.rand(num_clients) * 5 + 5,
        }

    def simulate_round(self, round_budget, d_i=None, c_i=None, a_i=None, sigma=None, T_up= None):
        """
        模拟一个联邦学习轮次

        Args:
            round_budget: 当前轮次预算
            round_idx: 轮次索引

        Returns:
            轮次结果字典
        """
        #每轮更新客户端参数
        self.client_params = {
            'sigma': sigma if sigma is not None else 3,
            'd_i': d_i if d_i is not None else np.random.rand(self.num_clients) * 1.5 + 1,
            'c_i': c_i if c_i is not None else np.random.rand(self.num_clients) * 1.5 + 1,
            'a_i': a_i if a_i is not None else np.random.rand(self.num_clients) * 1.5 + 1,
            'T_up':T_up if T_up is not None else np.random.rand(self.num_clients) * 5 + 3,
        }
        # self.client_params = {
        #     'sigma': 3,  # 本地训练轮数
        #     'd_i': np.random.rand(self.num_clients) * 2.5 + 1,
        #     'c_i': np.random.rand(self.num_clients) * 2.5 + 1,
        #     'a_i': np.random.rand(self.num_clients) * 2.5 + 1,
        #     'T_up': np.random.rand(self.num_clients) * 10 + 5,
        # }
        # self.client_params = {
        #     'sigma': 3,  # 本地训练轮数
        #     'd_i': np.random.rand(self.num_clients) * 3 + 1.25,
        #     'c_i': np.random.rand(self.num_clients) * 3 + 1.25,
        #     'a_i': np.random.rand(self.num_clients) * 3 + 1.25,
        #     'T_up': np.random.rand(self.num_clients) * 10 + 5,
        # }
        result_average, payment_average,_ = round_intra_average(round_budget, self.client_params.get('d_i'),
                                                self.client_params.get('c_i'), self.client_params.get('a_i'),
                                                self.client_params.get('sigma'), self.client_params.get('T_up'))
        result_random, payment_random,_ = round_intra_random(round_budget, self.client_params.get('d_i'),
                                               self.client_params.get('c_i'), self.client_params.get('a_i'),
                                               self.client_params.get('sigma'), self.client_params.get('T_up'))
        result_ours, payment_ours,_ = round_intra_compute(round_budget, self.client_params.get('d_i'),
                                                self.client_params.get('c_i'), self.client_params.get('a_i'),
                                                self.client_params.get('sigma'), self.client_params.get('T_up'))
        result_fair, payment_fair,_ = fair(round_budget, self.client_params.get('d_i'),
                                                self.client_params.get('c_i'), self.client_params.get('a_i'),
                                                self.client_params.get('sigma'), self.client_params.get('T_up'))
        # 获取当前状态
        state = self.budget_agent.get_state(
            round_budget=round_budget,
            client_data_sizes=self.client_params.get('d_i'),
            client_cpu_cycles=self.client_params.get('c_i'),
            client_a=self.client_params.get('a_i'),
            client_t_up=self.client_params.get('T_up')
        )
        # 选择动作（预算分配）
        weights, payments = self.budget_agent.act(state, round_budget)
        # result_reforcement = round_intra_reforcement(payments, self.client_params.get('d_i'),
        #                                   self.client_params.get('c_i'), self.client_params.get('a_i'),
        #                                   self.client_params.get('sigma'), self.client_params.get('T_up'))
        # 计算最优CPU周期
        optimal_cycles = self.budget_agent.calculate_optimal_cpu_cycles(
            payments, self.client_params,round_budget
        )

        # 计算训练时间
        training_times, result_reforcement = self.budget_agent.calculate_training_time(
            optimal_cycles, self.client_params
        )

        # 计算奖励
        reward = self.budget_agent.calculate_reward(result_reforcement, payments,round_budget)

        # 模拟下一状态（简化）
        next_state = state.copy()

        # 存储经验
        self.budget_agent.remember(state, weights, reward, next_state, False)

        # 训练
        # 训练：每轮训练多次，提高经验利用率
        for _ in range(4):  # 每轮训练4次
            self.budget_agent.replay()

        return {
            'budget': round_budget,
            'weights': weights,
            'payments': payments,
            'payment_average': payment_average,
            'payment_random': payment_random,
            'payment_ours': payment_ours,
            'payment_fair': payment_fair,
            'optimal_cycles': optimal_cycles,
            'training_times': training_times,
            #'max_time': max_time,
            'average_time': result_average,
            'random_time': result_random,
            'our_time': result_ours,
            'reforcement_time': result_reforcement,
            'fair_time': result_fair,
            'reward': reward
        }

    def run_simulation(self, round_budgets):
        """
        运行完整的联邦学习模拟

        Args:
            round_budgets: 每轮的预算列表

        Returns:
            模拟结果列表
        """
        results = []

        for round_idx, budget in enumerate(round_budgets):
            if round_idx >= self.num_rounds:
                break

            result = self.simulate_round(budget, round_idx)
            results.append(result)

            print(f"Round {round_idx + 1}:")
            print(f"  Budget: {budget:.2f}")
            #print(f"  Max Training Time: {result['max_time']:.2f}")
            print(f"  Random Training Time: {result['random_time']:.2f}")
            print(f"  Average Training Time: {result['average_time']:.2f}")
            print(f"  Fair Training Time: {result['fair_time']:.2f}")
            print(f"  Reforcement Training Time: {result['reforcement_time']:.2f}")
            print(f"  Our Training Time: {result['our_time']:.2f}")
            print(f"  Reward: {result['reward']:.2f}")
            # print(f"  weights: {result['weights']}")
            print(f"payment_random:{result['payment_random']}")
            print(sum(result['payment_random']))
            print(f"payment_average:{result['payment_average']}")
            print(sum(result['payment_average']))
            print(f"payment:{result['payments']}")
            print(sum(result['payments']))
            print(f"payment_fair:{result['payment_fair']}")
            print(sum(result['payment_fair']))
            print(f"payment_ours:{result['payment_ours']}")
            print(sum(result['payment_ours']))

        return results


