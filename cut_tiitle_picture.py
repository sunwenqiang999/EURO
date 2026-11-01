import numpy as np
import matplotlib.pyplot as plt

from generate_picture.generate_picture import generate_picture_fminst, generate_picture_minst, generate_picture_cifar10, \
    generate_picture_cifar100
from generate_picture.generate_picture_acc import generate_picture_acc_fmnist, generate_picture_acc_mnist, \
    generate_picture_acc_cifar10, generate_picture_acc_cifar100
from generate_picture.generate_picture_all import generate_picture_all_cifar100
from generate_picture.generate_picture_budget import generate_picture_budget_cifar100, generate_picture_budget_cifar10, \
    generate_picture_budget_mnist, generate_picture_budget_fmnist
from generate_picture.generate_picture_inter import generate_picture_inter_fmnist, generate_picture_inter_bar_fmnist, \
    generate_picture_inter_mnist, generate_picture_inter_bar_mnist, generate_picture_inter_cifar10, \
    generate_picture_inter_bar_cifar10, generate_picture_inter_cifar100, generate_picture_inter_bar_cifar100
from generate_picture.gnerate_picture_C import generate_picture_C_fmnist, generate_picture_C_mnist, \
    generate_picture_C_cifar10, generate_picture_C_cifar100

def find_first_exceed_index(accuracy_list, target_accuracy):
    """找到第一个超过目标精度的索引"""
    for i, accuracy in enumerate(accuracy_list):
        if accuracy >= target_accuracy:
            return i
    return None


def calculate_cumulative_time(time_list, index):
    """计算索引及其之前所有时间的总和"""
    return sum(time_list[:index + 1])

accuracy = np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')
print(accuracy)
# 读取.npy文件
random_timelist = np.load('E:\pycharm\EURO\\time\\mnist\\random_time_list.npy')

average_timelist = np.load('E:\pycharm\EURO\\time\\mnist\\average_time_list.npy')

fair_timelist = np.load('E:\pycharm\EURO\\time\\cifar100\\fair_time_list.npy')

reforcement_timelist = np.load('E:\pycharm\EURO\\time\\mnist\\reforcement_time_list.npy')

our_timelist = np.load('E:\pycharm\EURO\\time\\mnist\\our_time_list.npy')
target_accuracies = [96, 97, 98, 99]
alltime_lists = [
    our_timelist,  # The proportional share allocation algorithm (假设这是我们的方法)
    reforcement_timelist,  # Bid-price-first algorithm (假设这是强化学习)
    random_timelist,  # Random algorithm (假设这是随机算法)
    average_timelist,  # The proposed auction mechanism (假设这是平均机制)
    fair_timelist,  # The fairness mechanism (假设这是公平机制)
]
time_lists = [[] for i in range(5)]
for target_accuracy in target_accuracies:
    for idx, discrete_time_list in enumerate(alltime_lists):
        index = find_first_exceed_index(accuracy, target_accuracy)
        if index is not None:
            cumulative_time = calculate_cumulative_time(discrete_time_list, index)
            time_lists[idx].append(cumulative_time)
print('our',time_lists[0])
print('reforcement',time_lists[1])
print('random',time_lists[2])
print('average',time_lists[3])
print('fair',time_lists[4])
#generate_picture_acc_cifar100(accuracy, random_timelist, average_timelist,fair_timelist, reforcement_timelist, our_timelist)

