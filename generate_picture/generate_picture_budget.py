import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def generate_picture_budget_fmnist(accuracy_list, B1_timelist, B2_timelist,
                            B3_timelist, B4_timelist):

    threshold = 92
    index_99 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_99)
    accuracy_list = accuracy_list[:index_99+1]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    B1_time = np.cumsum(B1_timelist[:index_99+1])
    B2_time = np.cumsum(B2_timelist[:index_99+1])
    B3_time = np.cumsum(B3_timelist[:index_99+1])
    B4_time = np.cumsum(B4_timelist[:index_99+1])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(B1_time, accuracy_list, label='B=5', color='orange')
    plt.plot(B2_time, accuracy_list, label='B=10', color='green')
    plt.plot(B3_time, accuracy_list, label='B=20', color='red')
    plt.plot(B4_time, accuracy_list, label='B=30', color='purple')

    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\budget_picture\\budget_fmnist.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\budget_picture\\budget_fmnist.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
def generate_picture_budget_mnist(accuracy_list, B1_timelist, B2_timelist,
                            B3_timelist, B4_timelist):

    threshold = 99
    index_99 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_99)
    accuracy_list = accuracy_list[:index_99+1]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    B1_time = np.cumsum(B1_timelist[:index_99+1])
    B2_time = np.cumsum(B2_timelist[:index_99+1])
    B3_time = np.cumsum(B3_timelist[:index_99+1])
    B4_time = np.cumsum(B4_timelist[:index_99+1])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(B1_time, accuracy_list, label='B=5', color='orange')
    plt.plot(B2_time, accuracy_list, label='B=10', color='green')
    plt.plot(B3_time, accuracy_list, label='B=20', color='red')
    plt.plot(B4_time, accuracy_list, label='B=30', color='purple')

    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\budget_picture\\budget_mnist.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\budget_picture\\budget_mnist.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()

def generate_picture_budget_cifar10(accuracy_list, B1_timelist, B2_timelist,
                            B3_timelist, B4_timelist):

    threshold = 97
    index_99 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_99)
    accuracy_list = accuracy_list[:index_99+1]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    B1_time = np.cumsum(B1_timelist[:index_99+1])
    B2_time = np.cumsum(B2_timelist[:index_99+1])
    B3_time = np.cumsum(B3_timelist[:index_99+1])
    B4_time = np.cumsum(B4_timelist[:index_99+1])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(B1_time, accuracy_list, label='B=5', color='orange')
    plt.plot(B2_time, accuracy_list, label='B=10', color='green')
    plt.plot(B3_time, accuracy_list, label='B=20', color='red')
    plt.plot(B4_time, accuracy_list, label='B=30', color='purple')

    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\budget_picture\\budget_cifar10.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\budget_picture\\budget_cifar10.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
def generate_picture_budget_cifar100(accuracy_list, B1_timelist, B2_timelist,
                            B3_timelist, B4_timelist):

    threshold = 99.5
    index_99 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_99)
    accuracy_list = accuracy_list[:index_99+1]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    B1_time = np.cumsum(B1_timelist[:index_99+1])
    B2_time = np.cumsum(B2_timelist[:index_99+1])
    B3_time = np.cumsum(B3_timelist[:index_99+1])
    B4_time = np.cumsum(B4_timelist[:index_99+1])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(B1_time, accuracy_list, label='B=5', color='orange')
    plt.plot(B2_time, accuracy_list, label='B=10', color='green')
    plt.plot(B3_time, accuracy_list, label='B=20', color='red')
    plt.plot(B4_time, accuracy_list, label='B=30', color='purple')

    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\budget_picture\\budget_cifar100.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\budget_picture\\budget_cifar100.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()