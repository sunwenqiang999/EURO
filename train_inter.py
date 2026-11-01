# 运行示例
from generate_picture.generate_picture_inter import generate_picture_inter_fmnist, generate_picture_inter_bar_fmnist, \
    generate_picture_inter_mnist, generate_picture_inter_bar_mnist, generate_picture_inter_cifar10, \
    generate_picture_inter_bar_cifar10, generate_picture_inter_cifar100, generate_picture_inter_bar_cifar100
from inter_round_dqn.FederatedLearningSimulator import FederatedLearningSimulator
import numpy as np
import json
if __name__ == "__main__":
    # 创建模拟器并运行训练
    simulator = FederatedLearningSimulator(
        num_clients=20,
        max_sessions=50,
        total_budget=10000
    )
    #accuracy = np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
    #accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
    #accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
    accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')
    accuracy = np.array(accuracy)
    # 运行完整的训练过程

    results = simulator.run_complete_training(accuracy,target_accuracy=97, lambda_param=1)
    # 可以分析结果
    if results:
        accuracies = [r['accuracy'] for r in results]
        waiting_times = [r['waiting_time'] for r in results]
        session_budgets = [r['session_budget'] for r in results]
        actions = [r['action'] for r in results]
        result_averages = [r['result_average'] for r in results]
        result_ours = [r['result_ours'] for r in results]
        result_random = [r['result_random'] for r in results]
        print("准确率:", accuracies)
        print("等待时间:", waiting_times)
        print("平均时间:", result_averages)
        print("ours时间:", result_ours)
        print("random时间:", result_random)
        print("会话预算:", session_budgets)
        print("动作:", actions)
        generate_picture_inter_cifar100(accuracy, result_random, result_averages, waiting_times, result_ours)
        generate_picture_inter_bar_cifar100(accuracy, result_random, result_averages, waiting_times, result_ours)
        np.save('E:\pycharm\EURO\\time\\timelist_inter\cifar100\\random_time_list.npy', result_random)
        np.save('E:\pycharm\EURO\\time\\timelist_inter\cifar100\\average_time_list.npy', result_averages)
        np.save('E:\pycharm\EURO\\time\\timelist_inter\cifar100\\reforcement_time_list.npy', waiting_times)
        np.save('E:\pycharm\EURO\\time\\timelist_inter\cifar100\\our_time_list.npy', result_ours)
        with open('model/simulation_results_inter_cifar100.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # 保存模型
        simulator.inter_bma.save_model('model/intra_budget_agent_inter_cifar100.pth')