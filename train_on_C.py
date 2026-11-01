from eu import *
from eyy import FederatedLearningSimulator
from generate_picture_acc import generate_picture_acc_cifar100
from generate_picture.gnerate_picture_C import generate_picture_C_cifar10, generate_picture_C_cifar100, generate_picture_C_mnist, \
    generate_picture_C_fmnist
from inter_budget_compute import round_inter_compute, round_intra_average, round_intra_random, round_intra_reforcement, \
    round_intra_compute, fair

if __name__ == "__main__":
    # 创建模拟器
    num_clients = 20
    num_rounds = 50
    # 假设您已经有了每轮的预算分配
    #accuracy = np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
    accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
    #accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
   #accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')
    print(accuracy)
    accuracy = np.array(accuracy)
    B = 10000
    round_budgets = round_inter_compute(accuracy, B, 50, 1)
    round_budgets2 = round_inter_compute(accuracy, B, 50, 10)
    round_budgets3 = round_inter_compute(accuracy, B, 50, 100)
    round_budgets4 = round_inter_compute(accuracy, B, 50, 500)
    round_budgets5 = round_inter_compute(accuracy, B, 50, 1000)
    print(round_budgets)
    print(round_budgets2)
    print(round_budgets3)
    print(round_budgets4)
    print(round_budgets5)
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
        ours5, payment_ours5 = round_intra_compute(round_budgets5[round_idx], d_i, c_i, a_i, sigma, T_up)
        result_ours.append(ours)
        result_ours2.append(ours2)
        result_ours3.append(ours3)
        result_ours4.append(ours4)
        result_ours5.append(ours5)
    print(result_ours)
    print(result_ours2)
    print(result_ours3)
    print(result_ours4)
    print(result_ours5)
    generate_picture_C_fmnist(accuracy, result_ours, result_ours2, result_ours3, result_ours4, result_ours5)
    # 保存结果
    # 保存为 .npy 文件
    np.save('E:\pycharm\EURO\\time\\timelist_C\\fmnist\\our_time_list.npy', result_ours)
    np.save('E:\pycharm\EURO\\time\\timelist_C\\fmnist\\our_time_list2.npy', result_ours2)
    np.save('E:\pycharm\EURO\\time\\timelist_C\\fmnist\\our_time_list3.npy', result_ours3)
    np.save('E:\pycharm\EURO\\time\\timelist_C\\fmnist\\our_time_list4.npy', result_ours4)
    np.save('E:\pycharm\EURO\\time\\timelist_C\\fmnist\\our_time_list5.npy', result_ours5)
