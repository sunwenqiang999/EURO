import matplotlib.pyplot as plt
import numpy as np
def generate_picture_acc_cifar100(accuracy_list, random_timelist, average_timelist,
                            fair_timelist, reforcement_timelist, ours_timelist):
    """
        绘制不同方法的精度随时间变化的曲线。

        :param accuracy_list: 精度列表（纵坐标）
        :param random_timelist: 随机方法每轮时间列表（横坐标）
        :param average_timelist: 平均机制每轮时间列表
        :param fair_timelist: 公平算法每轮时间列表
        :param reforcement_timelist: 强化学习方法每轮时间列表
        :param ours_timelist: 我们的方法每轮时间列表
        """
    threshold = 99.2
    index_99 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_99)
    accuracy_list = accuracy_list[:index_99+1]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    random_time = np.cumsum(random_timelist[:index_99+1])
    average_time = np.cumsum(average_timelist[:index_99+1])
    fair_time = np.cumsum(fair_timelist[:index_99+1])
    reforcement_time = np.cumsum(reforcement_timelist[:index_99+1])
    ours_time = np.cumsum(ours_timelist[:index_99+1])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(ours_time, accuracy_list, label='IRBAI+IRBSB', color='blue')
    plt.plot(reforcement_time, accuracy_list, label='IRBAI+IntraBMA', color='orange')
    plt.plot(random_time, accuracy_list, label='IRBAI+Random', color='green')
    plt.plot(average_time, accuracy_list, label='IRBAI+FedAvg', color='red')
    plt.plot(fair_time, accuracy_list, label='IRBAI+Fair', color='purple')


    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')


    plt.legend(loc='lower right')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\intra_picture\\accVstime_cifar100.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\intra_picture\\accVstime_cifar100.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
def generate_picture_acc_cifar10(accuracy_list, random_timelist, average_timelist,
                            fair_timelist, reforcement_timelist, ours_timelist):
    """
        绘制不同方法的精度随时间变化的曲线。

        :param accuracy_list: 精度列表（纵坐标）
        :param random_timelist: 随机方法每轮时间列表（横坐标）
        :param average_timelist: 平均机制每轮时间列表
        :param fair_timelist: 公平算法每轮时间列表
        :param reforcement_timelist: 强化学习方法每轮时间列表
        :param ours_timelist: 我们的方法每轮时间列表
        """
    threshold = 97
    index_99 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_99)
    accuracy_list = accuracy_list[:index_99+1]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    random_time = np.cumsum(random_timelist[:index_99+1])
    average_time = np.cumsum(average_timelist[:index_99+1])
    fair_time = np.cumsum(fair_timelist[:index_99+1])
    reforcement_time = np.cumsum(reforcement_timelist[:index_99+1])
    ours_time = np.cumsum(ours_timelist[:index_99+1])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(ours_time, accuracy_list, label='IRBAI+IRBSB', color='blue')
    plt.plot(reforcement_time, accuracy_list, label='IRBAI+IntraBMA', color='orange')
    plt.plot(random_time, accuracy_list, label='IRBAI+Random', color='green')
    plt.plot(average_time, accuracy_list, label='IRBAI+FedAvg', color='red')
    plt.plot(fair_time, accuracy_list, label='IRBAI+Fair', color='purple')

    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\intra_picture\\accVstime_cifar10.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\intra_picture\\accVstime_cifar10.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
def generate_picture_acc_mnist(accuracy_list, random_timelist, average_timelist,
                            fair_timelist, reforcement_timelist, ours_timelist):
    """
        绘制不同方法的精度随时间变化的曲线。

        :param accuracy_list: 精度列表（纵坐标）
        :param random_timelist: 随机方法每轮时间列表（横坐标）
        :param average_timelist: 平均机制每轮时间列表
        :param fair_timelist: 公平算法每轮时间列表
        :param reforcement_timelist: 强化学习方法每轮时间列表
        :param ours_timelist: 我们的方法每轮时间列表
        """
    threshold = 99
    index_99 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_99)
    accuracy_list = accuracy_list[:index_99+1]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    random_time = np.cumsum(random_timelist[:index_99+1])
    average_time = np.cumsum(average_timelist[:index_99+1])
    fair_time = np.cumsum(fair_timelist[:index_99+1])
    reforcement_time = np.cumsum(reforcement_timelist[:index_99+1])
    ours_time = np.cumsum(ours_timelist[:index_99+1])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(ours_time, accuracy_list, label='IRBAI+IRBSB', color='blue')
    plt.plot(reforcement_time, accuracy_list, label='IRBAI+IntraBMA', color='orange')
    plt.plot(random_time, accuracy_list, label='IRBAI+Random', color='green')
    plt.plot(average_time, accuracy_list, label='IRBAI+FedAvg', color='red')
    plt.plot(fair_time, accuracy_list, label='IRBAI+Fair', color='purple')

    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\intra_picture\\accVstime_mnist.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\intra_picture\\accVstime_mnist.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
def generate_picture_acc_fmnist(accuracy_list, random_timelist, average_timelist,
                            fair_timelist, reforcement_timelist, ours_timelist):
    """
        绘制不同方法的精度随时间变化的曲线。

        :param accuracy_list: 精度列表（纵坐标）
        :param random_timelist: 随机方法每轮时间列表（横坐标）
        :param average_timelist: 平均机制每轮时间列表
        :param fair_timelist: 公平算法每轮时间列表
        :param reforcement_timelist: 强化学习方法每轮时间列表
        :param ours_timelist: 我们的方法每轮时间列表
        """
    threshold = 91.5
    index_99 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_99)
    accuracy_list = accuracy_list[:index_99+1]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    random_time = np.cumsum(random_timelist[:index_99+1])
    average_time = np.cumsum(average_timelist[:index_99+1])
    fair_time = np.cumsum(fair_timelist[:index_99+1])
    reforcement_time = np.cumsum(reforcement_timelist[:index_99+1])
    ours_time = np.cumsum(ours_timelist[:index_99+1])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(ours_time, accuracy_list, label='IRBAI+IRBSB', color='blue')
    plt.plot(reforcement_time, accuracy_list, label='IRBAI+IntraBMA', color='orange')
    plt.plot(random_time, accuracy_list, label='IRBAI+Random', color='green')
    plt.plot(average_time, accuracy_list, label='IRBAI+FedAvg', color='red')
    plt.plot(fair_time, accuracy_list, label='IRBAI+Fair', color='purple')

    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\intra_picture\\accVstime_fmnist.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\intra_picture\\accVstime_fmnist.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()