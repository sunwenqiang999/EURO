import numpy as np

from generate_picture.generate_picture_budget import generate_picture_budget_fmnist
from inter_budget_compute import round_inter_compute, round_intra_compute

if __name__ == "__main__":
    num_clients = 20
    num_rounds = 50
    # 假设您已经有了每轮的预算分配
    #accuracy = np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
    accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
    #accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
    #accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')

    print(accuracy)
    accuracy = np.array(accuracy)

    target_accuracy = 92
    round_budgets = round_inter_compute(accuracy, 5000, 50, 10,target_accuracy)
    round_budgets2 = round_inter_compute(accuracy, 10000, 50, 10,target_accuracy)
    round_budgets3 = round_inter_compute(accuracy, 20000, 50, 10,target_accuracy)
    round_budgets4 = round_inter_compute(accuracy, 30000, 50, 10,target_accuracy)
    print(round_budgets)
    print(round_budgets2)
    print(round_budgets3)
    print(round_budgets4)
    result_ours = []
    result_ours2 = []
    result_ours3 = []
    result_ours4 = []
    result_ours5 = []
    for round_idx in range(50):
        sigma = 3  # 本地训练轮数
        d_i = np.random.rand(num_clients) * 1.5 + 1
        c_i = np.random.rand(num_clients) * 1.5 + 1
        a_i = np.random.rand(num_clients) * 1.5 + 1
        T_up = np.random.rand(num_clients) * 5 + 3
        ours, payment_ours = round_intra_compute(round_budgets[round_idx], d_i, c_i, a_i, sigma, T_up)
        ours2, payment_ours2 = round_intra_compute(round_budgets2[round_idx], d_i, c_i, a_i, sigma, T_up)
        ours3, payment_ours3 = round_intra_compute(round_budgets3[round_idx], d_i, c_i, a_i, sigma, T_up)
        ours4, payment_ours4 = round_intra_compute(round_budgets4[round_idx], d_i, c_i, a_i, sigma, T_up)
        result_ours.append(ours)
        result_ours2.append(ours2)
        result_ours3.append(ours3)
        result_ours4.append(ours4)
    print(result_ours)
    print(result_ours2)
    print(result_ours3)
    print(result_ours4)
    generate_picture_budget_fmnist(accuracy, result_ours, result_ours2, result_ours3, result_ours4)
    # 保存结果
    # 保存为 .npy 文件
    np.save('E:\pycharm\EURO\\time\\timelist_Budget\\fmnist\our_time_list.npy', result_ours)
    np.save('E:\pycharm\EURO\\time\\timelist_Budget\\fmnist\our_time_list2.npy', result_ours2)
    np.save('E:\pycharm\EURO\\time\\timelist_Budget\\fmnist\our_time_list3.npy', result_ours3)
    np.save('E:\pycharm\EURO\\time\\timelist_Budget\\fmnist\our_time_list4.npy', result_ours4)