from eu import *
from eyy import FederatedLearning
from generate_picture.generate_picture import generate_picture_fminst
from generate_picture.generate_picture_acc import generate_picture_acc_fmnist

from inter_budget_compute import round_inter_compute, round_intra_average, round_intra_random, round_intra_reforcement, \
    round_intra_compute, fair, round_inter_compute_cifar
if __name__ == "__main__":
    # 创建模拟器
    num_clients = 20
    num_rounds = 50
    simulator = FederatedLearning(num_clients, num_rounds)

    # 假设您已经有了每轮的预算分配
    #accuracy = np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
    accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
    #accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
    #accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')
    print(accuracy)
    accuracy = np.array(accuracy)
    B = 10000
    round_budgets = round_inter_compute(accuracy, B, 50,10)
    print(round_budgets)
    # 运行模拟
    results = simulator.run_simulation(round_budgets)
    random_time_list = [result['random_time'] for result in results]
    average_time_list = [result['average_time'] for result in results]
    fair_time_list = [result['fair_time'] for result in results]
    reforcement_time_list = [result['reforcement_time'] for result in results]
    our_time_list = [result['our_time'] for result in results]
    generate_picture_fminst(accuracy, random_time_list,
                            average_time_list, fair_time_list,
                            reforcement_time_list, our_time_list)
    generate_picture_acc_fmnist(accuracy, random_time_list,
                            average_time_list, fair_time_list,
                            reforcement_time_list, our_time_list)
    # 保存结果
    # 保存为 .npy 文件
    np.save('E:\pycharm\EURO\\time\\fmnist\\random_time_list.npy', random_time_list)
    np.save('E:\pycharm\EURO\\time\\fmnist\\average_time_list.npy', average_time_list)
    np.save('E:\pycharm\EURO\\time\\fmnist\\fair_time_list.npy', fair_time_list)
    np.save('E:\pycharm\EURO\\time\\fmnist\\reforcement_time_list.npy', reforcement_time_list)
    np.save('E:\pycharm\EURO\\time\\fmnist\\our_time_list.npy', our_time_list)
    with open('simulation_results_fmnist.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # 保存模型
    simulator.budget_agent.save_model('intra_budget_agent_fmnist.pth')

    print("模拟完成！结果已保存到 simulation_results_fmnist.json")
    print("模型已保存到 intra_budget_agent.pth")
