import numpy as np

from generate_picture.generate_picture_all import generate_picture_all_cifar100, generate_picture_all_bar_cifar100

#accuracy = np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
#accuracy = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')

random_timelist = np.load('E:\pycharm\EURO\\time\\timelist_all\\cifar100\\random_time_list.npy')

average_timelist = np.load('E:\pycharm\EURO\\time\\timelist_all\\cifar100\\average_time_list.npy')

fair_timelist = np.load('E:\pycharm\EURO\\time\\timelist_all\\cifar100\\fair_time_list.npy')

reforcement_timelist = np.load('E:\pycharm\EURO\\time\\timelist_all\\cifar100\\reforcement_time_list.npy')

our_timelist = np.load('E:\pycharm\EURO\\time\\timelist_all\\cifar100\\our_time_list.npy')

generate_picture_all_cifar100(accuracy, random_timelist, average_timelist, reforcement_timelist, our_timelist,fair_timelist)
generate_picture_all_bar_cifar100(accuracy, random_timelist, average_timelist,  reforcement_timelist, our_timelist,fair_timelist)