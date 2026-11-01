import numpy as np
import json

from generate_picture.generate_picture_pretest import generate_picture_pre_fmnist, generate_picture_pre_bar_fmnist, \
    generate_picture_pre_cifar10, generate_picture_pre_bar_cifar10
from inter_budget_compute import round_intra_random, round_intra_average, round_intra_compute

if __name__ == "__main__":
    num_clients = 20
    #accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
    accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
    B_increase = []
    B_average = []
    B_decrease = []
    B = 10000
    for i in range(len(accuracy)):
        if i < 10:
            p1 = B / len(accuracy) * 1.5
            p2 = B / len(accuracy) * 1
            p3 = B / len(accuracy) * 0.5
            B_decrease.append(p1)
            B_average.append(p2)
            B_increase.append(p3)
        elif 10 <= i < 20:
            p1 = B / len(accuracy) * 1.25
            p2 = B / len(accuracy) * 1
            p3 = B / len(accuracy) * 0.75
            B_decrease.append(p1)
            B_average.append(p2)
            B_increase.append(p3)
        elif 20 <= i < 30:
            p1 = B / len(accuracy) * 1
            p2 = B / len(accuracy) * 1
            p3 = B / len(accuracy) * 1
            B_decrease.append(p1)
            B_average.append(p2)
            B_increase.append(p3)
        elif 30 <= i < 40:
            p1 = B / len(accuracy) * 0.75
            p2 = B / len(accuracy) * 1
            p3 = B / len(accuracy) * 1.25
            B_decrease.append(p1)
            B_average.append(p2)
            B_increase.append(p3)
        else:
            p1 = B / len(accuracy) * 0.5
            p2 = B / len(accuracy) * 1
            p3 = B / len(accuracy) * 1.5
            B_decrease.append(p1)
            B_average.append(p2)
            B_increase.append(p3)
    print(B_decrease)
    print(B_average)
    print(B_increase)
    result_decreases = []
    result_averages = []
    result_increases = []
    for i in range(len(accuracy)):
        # sigma = 3  # 本地训练轮数
        # d_i = np.random.rand(num_clients) * 1.5 + 1
        # c_i = np.random.rand(num_clients) * 1.5 + 1
        # a_i = np.random.rand(num_clients) * 1.5 + 1
        # T_up = np.random.rand(num_clients) * 5 + 3
        sigma = 3  # 本地训练轮数
        d_i = np.random.rand(num_clients) * 2.5 + 1
        c_i = np.random.rand(num_clients) * 2.5 + 1
        a_i = np.random.rand(num_clients) * 2.5 + 1
        T_up = np.random.rand(num_clients) * 10 + 5
        result_decrease, _, _ = round_intra_average(B_decrease[i], d_i, c_i, a_i, sigma, T_up)
        result_average, _, _ = round_intra_average(B_average[i], d_i, c_i, a_i, sigma, T_up)
        result_increase, _, _ = round_intra_average(B_increase[i], d_i, c_i, a_i, sigma, T_up)
        result_decreases.append(result_decrease)
        result_averages.append(result_average)
        result_increases.append(result_increase)
    print(result_decreases)
    print(result_averages)
    print(result_increases)
    np.save('E:\pycharm\EURO\\time\\timelist_pre\\cifar10\\decreases_time_list.npy', result_decreases)
    np.save('E:\pycharm\EURO\\time\\timelist_pre\\cifar10\\average_time_list.npy', result_averages)
    np.save('E:\pycharm\EURO\\time\\timelist_pre\\cifar10\\increases_time_list.npy', result_increases)
    generate_picture_pre_cifar10(accuracy, result_decreases, result_averages, result_increases)
    generate_picture_pre_bar_cifar10(accuracy, result_decreases, result_averages, result_increases)
