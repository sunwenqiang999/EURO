import numpy as np

from generate_picture.generate_picture_all import generate_picture_all_mnist, generate_picture_all_bar_mnist, \
    generate_picture_all_fmnist, generate_picture_all_bar_fmnist, generate_picture_all_cifar10, \
    generate_picture_all_bar_cifar10, generate_picture_all_cifar100, generate_picture_all_bar_cifar100
from generate_picture.generate_picture_budget import generate_picture_budget_mnist, generate_picture_budget_fmnist, \
    generate_picture_budget_cifar10, generate_picture_budget_cifar100
from generate_picture.gnerate_picture_C import generate_picture_C_cifar100, generate_picture_C_cifar10, \
    generate_picture_C_fmnist, generate_picture_C_mnist

#accuracy = np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')

our_timelist = np.load('E:\pycharm\EURO\\time\\timelist_Budget\\cifar100\\our_time_list.npy')

our_timelist2 = np.load('E:\pycharm\EURO\\time\\timelist_Budget\\cifar100\\our_time_list2.npy')

our_timelist3 = np.load('E:\pycharm\EURO\\time\\timelist_Budget\\cifar100\\our_time_list3.npy')

our_timelist4 = np.load('E:\pycharm\EURO\\time\\timelist_Budget\\cifar100\\our_time_list4.npy')

generate_picture_budget_cifar100(accuracy, our_timelist, our_timelist2, our_timelist3, our_timelist4)
