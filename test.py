import numpy as np

if __name__ == '__main__':
    accuracy = np.load('E:\pycharm\EURO\\timelist\cifar100_trainacc.npy')
    accuracy2 = np.load('E:\pycharm\EURO\\timelist\cifar10_trainacc.npy')
    accuracy3= np.load('E:\pycharm\EURO\\timelist\minstcnn_trainacc.npy')
    accuracy4 = np.load('E:\pycharm\EURO\\timelist\\fminstcnn_trainacc.npy')
    print(accuracy)
    print(accuracy2)
    print(accuracy3)
    print(accuracy4)
