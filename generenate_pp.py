import numpy as np

from generate_picture.generate_picture_pretest import generate_picture_pre_cifar10, generate_picture_pre_bar_cifar10, \
    generate_picture_pre_fmnist, generate_picture_pre_bar_fmnist

#accuracy = np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')

result_decreases=np.load('E:\pycharm\EURO\\time\\timelist_pre\\fmnist\\decreases_time_list.npy')
result_averages=np.load('E:\pycharm\EURO\\time\\timelist_pre\\fmnist\\average_time_list.npy')
result_increases=np.load('E:\pycharm\EURO\\time\\timelist_pre\\fmnist\\increases_time_list.npy')

generate_picture_pre_fmnist(accuracy, result_decreases, result_averages, result_increases)
generate_picture_pre_bar_fmnist(accuracy, result_decreases, result_averages, result_increases)