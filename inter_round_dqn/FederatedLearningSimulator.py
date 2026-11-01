import numpy as np

from eyy import FederatedLearning
from inter_budget_compute import round_intra_compute, round_intra_average, round_inter_average, round_inter_compute, \
    round_inter_random, round_inter_compute_cifar, round_intra_random, fair
from inter_round_dqn.InterSessionBMA import InterSessionBMA


class FederatedLearningSimulator:
    """联邦学习模拟器，用于测试InterBMA"""

    def __init__(self, num_clients=20, max_sessions=50, total_budget=10000):
        self.num_clients = num_clients
        self.max_sessions = max_sessions
        self.total_budget = total_budget
        self.remaining_budget = total_budget

        # 初始化InterBMA
        # 状态维度：会话索引(1) + 剩余预算(1) + 历史准确率(3) + 历史支付(3*10) + 历史时间(3*10)
        state_dim = 1 + 1 + 3 + 3 * 20 + 3 * 20  # = 65
        action_dim = 20  # 20种不同的预算分配策略

        self.inter_bma = InterSessionBMA(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=1,
            gamma=1.0,
            epsilon=0.9,
            epsilon_decay=0.995,
            tau=1
        )

        self.current_session = 0
        self.session_history = []
        self.client_params = {
            'sigma': 3,
            'd_i': np.random.rand(self.num_clients) * 1.5 + 1,
            'c_i': np.random.rand(self.num_clients) * 1.5 + 1,
            'a_i': np.random.rand(self.num_clients) * 1.5 + 1,
            'T_up': np.random.rand(self.num_clients) * 5 + 3,
        }

    def simulate_fl_training(self, round_budget, average, ours, random,simulator):
        """模拟单次联邦学习训练会话"""
        # 每轮更新客户端参数
        self.client_params = {
            'sigma': 3,
            'd_i': np.random.rand(self.num_clients) * 1.5 + 1,
            'c_i': np.random.rand(self.num_clients) * 1.5 + 1,
            'a_i': np.random.rand(self.num_clients) * 1.5 + 1,
            'T_up': np.random.rand(self.num_clients) * 5 + 3,
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
        # result_average, payment_average, times_average = round_intra_compute(average, self.client_params.get('d_i'),
        #                                                              self.client_params.get('c_i'),
        #                                                              self.client_params.get('a_i'),
        #                                                              self.client_params.get('sigma'),
        #                                                              self.client_params.get('T_up'))
        # result_ours, payment_ours, times_ours = round_intra_compute(ours, self.client_params.get('d_i'),
        #                                                              self.client_params.get('c_i'),
        #                                                              self.client_params.get('a_i'),
        #                                                              self.client_params.get('sigma'),
        #                                                              self.client_params.get('T_up'))
        # result_random, payment_random, times_random = round_intra_compute(random, self.client_params.get('d_i'),
        #                                                              self.client_params.get('c_i'),
        #                                                              self.client_params.get('a_i'),
        #                                                              self.client_params.get('sigma'),
        #                                                              self.client_params.get('T_up'))
        # result, payment, times = round_intra_compute(round_budget, self.client_params.get('d_i'),
        #                                                              self.client_params.get('c_i'),
        #                                                              self.client_params.get('a_i'),
        #                                                              self.client_params.get('sigma'),
        #                                                              self.client_params.get('T_up'))
        result_average, payment_average, times_average = round_intra_average(average, self.client_params.get('d_i'),
                                                                             self.client_params.get('c_i'),
                                                                             self.client_params.get('a_i'),
                                                                             self.client_params.get('sigma'),
                                                                             self.client_params.get('T_up'))
        result_ours, payment_ours, times_ours = round_intra_compute(ours, self.client_params.get('d_i'),
                                                                    self.client_params.get('c_i'),
                                                                    self.client_params.get('a_i'),
                                                                    self.client_params.get('sigma'),
                                                                    self.client_params.get('T_up'))
        result_random, payment_random, times_random = round_intra_random(random, self.client_params.get('d_i'),
                                                                         self.client_params.get('c_i'),
                                                                         self.client_params.get('a_i'),
                                                                         self.client_params.get('sigma'),
                                                                         self.client_params.get('T_up'))
        result_fair, payment_fair, times_fair = fair(average, self.client_params.get('d_i'),
                                                     self.client_params.get('c_i'),
                                                     self.client_params.get('a_i'),
                                                     self.client_params.get('sigma'),
                                                     self.client_params.get('T_up'))
        results=simulator.simulate_round(round_budget, self.client_params.get('d_i'), self.client_params.get('c_i'),
                                 self.client_params.get('a_i'),self.client_params.get('sigma'),
                                 self.client_params.get('T_up'))
        result=results['reforcement_time']
        payment=results['payments']
        times=results['training_times']
        return payment, times, result, result_average, result_ours, result_random, result_fair

    def run_training_session(self, lambda_param, accuracy, target_accuracy, payment_average, payment_ours,
                             payment_random, simulator):
        """运行单次训练会话"""
        if self.current_session >= self.max_sessions or self.remaining_budget <= 0:
            return None

        # 构建当前状态
        current_state = self.inter_bma.build_state(self.current_session, self.remaining_budget)

        # 选择动作（预算分配策略）
        action = self.inter_bma.select_action(current_state)

        # 将动作转换为具体的预算分配
        session_budget = self.inter_bma.action_to_budget(
            action, self.remaining_budget, self.current_session, self.max_sessions, accuracy, target_accuracy
        )

        # 确保不超出剩余预算
        session_budget = min(session_budget, self.remaining_budget)
        # 模拟联邦学习训练
        payments, times, waiting_time, result_average, result_ours, result_random, result_fair = self.simulate_fl_training(
            session_budget, payment_average, payment_ours, payment_random,simulator)

        # 计算奖励
        previous_accuracy = self.inter_bma.history['accuracies'][-1] if self.inter_bma.history['accuracies'] else 0.7
        reward = self.inter_bma.calculate_reward(accuracy, previous_accuracy, waiting_time, lambda_param)

        # 更新预算
        actual_cost = sum(payments)
        self.remaining_budget -= actual_cost

        # 更新历史
        self.inter_bma.update_history(payments, times, accuracy, self.current_session)

        # 构建下一状态
        next_state = self.inter_bma.build_state(self.current_session + 1, self.remaining_budget)

        # 存储经验
        done = (self.current_session >= self.max_sessions - 1) or (self.remaining_budget <= 0)
        self.inter_bma.store_experience(current_state, action, reward, next_state, done)

        loss = []

        for i in range(4):
            # 训练网络
            loss = self.inter_bma.train()

        # 记录会话结果
        session_result = {
            'session': self.current_session,
            'action': action,
            'session_budget': session_budget,
            'actual_cost': actual_cost,
            'accuracy': accuracy,
            'waiting_time': waiting_time,
            'reward': reward,
            'remaining_budget': self.remaining_budget,
            'loss': loss,
            'result_average': result_average,
            'result_ours': result_ours,
            'result_random': result_random,
            'result_fair': result_fair
        }

        self.session_history.append(session_result)
        self.current_session += 1

        return session_result

    def run_complete_training(self, accuracy, target_accuracy, lambda_param=1):
        """运行完整的多轮训练"""
        print("开始多会话联邦学习训练...")
        print(f"总预算: {self.total_budget}, 最大会话数: {self.max_sessions}")
        print("-" * 80)
        payment_average = round_inter_average(10000, 50)
        print("平均预算:", payment_average)
        payment_ours = round_inter_compute_cifar(accuracy, 10000, 50, 10, target_accuracy)
        print("ours预算:", payment_ours)
        payment_random = round_inter_random(10000, 50)
        print("random预算:", payment_random)
        simulator = FederatedLearning(20, 50)
        i = 0
        while self.current_session < self.max_sessions and self.remaining_budget > 0:
            result = self.run_training_session(lambda_param, accuracy[i], target_accuracy,
                                               payment_average[i], payment_ours[i], payment_random[i], simulator)
            i += 1
            if result is None:
                break

            print(f"会话 {result['session']:2d}: "
                  f"预算={result['session_budget']:6.1f}, "
                  f"实际花费={result['actual_cost']:6.1f}, "
                  f"准确率={result['accuracy']:.3f}, "
                  f"等待时间={result['waiting_time']:5.1f}, "
                  f"平均时间={result['result_average']:.3f}, "
                  f"ours时间={result['result_ours']:.3f}, "
                  f"random时间={result['result_random']:.3f}, "
                  f"剩余预算={result['remaining_budget']:6.1f}")

        print("-" * 80)
        print(f"训练完成！总会话数: {len(self.session_history)}")

        return self.session_history
