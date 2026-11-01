import matplotlib.pyplot as plt


def find_first_exceed_index(accuracy_list, target_accuracy):
    """找到第一个超过目标精度的索引"""
    for i, accuracy in enumerate(accuracy_list):
        if accuracy >= target_accuracy:
            return i
    return None


def calculate_cumulative_time(time_list, index):
    """计算索引及其之前所有时间的总和"""
    return sum(time_list[:index + 1])


def generate_picture_fminst(accuracy_list, random_timelist, average_timelist,
                            fair_timelist, reforcement_timelist, ours_timelist):
    """
    生成训练时间与精准度的条形图。

    :param accuracy_list: 精准度列表
    :param average_timelist: 平均机制时间列表
    :param random_timelist: 随机算法时间列表
    :param reforcement_timelist: 强化学习时间列表
    :param ours_timelist: 我们的方法时间列表
    :param fair_timelist: 公平算法时间列表
    """
    # 指定精度列表
    target_accuracies = [80, 85, 88, 90]
    # 组织时间数据
    alltime_lists = [
        ours_timelist,  # The proportional share allocation algorithm (假设这是我们的方法)
        reforcement_timelist,  # Bid-price-first algorithm (假设这是强化学习)
        random_timelist,  # Random algorithm (假设这是随机算法)
        average_timelist,  # The proposed auction mechanism (假设这是平均机制)
        fair_timelist,  # Quality-aware algorithm (假设这是公平算法)

    ]

    # 方法名称列表
    methods = [
        "IRBAI+IRBSB",
        "IRBAI+IntraBMA",
        "IRBAI+Random",
        "IRBAI+FedAvg",
        "IRBAI+Fair",
    ]
    # 初始化时间列表
    time_lists = [[] for _ in methods]

    # 对于每个指定精度，找到第一个超过该精度的点并计算时间总和
    for target_accuracy in target_accuracies:
        for idx, discrete_time_list in enumerate(alltime_lists):
            index = find_first_exceed_index(accuracy_list, target_accuracy)
            if index is not None:
                cumulative_time = calculate_cumulative_time(discrete_time_list, index)
                time_lists[idx].append(cumulative_time)
            else:
                time_lists[idx].append(0)  # 如果没有找到，时间设为 0

    # 设置条形图的位置
    bar_width = 0.15
    r = [i for i in range(len(target_accuracies))]  # 只有 4 个目标精度
    colors = ['b', 'orange', 'g', 'r', 'purple']

    # 绘制每个方法的条形图
    for idx, times in enumerate(time_lists):
        plt.bar([x + idx * bar_width for x in r], times, color=colors[idx], width=bar_width, label=methods[idx])

    # 添加标题和标签
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Time (s)')

    # 设置 x 轴刻度
    plt.xticks([r + bar_width * 2 for r in range(len(target_accuracies))], target_accuracies)

    # 添加图例
    plt.legend()
    plt.savefig(
        r'E:\pycharm\EURO\figures\intra_picture\training_time_fminst.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        r'E:\pycharm\EURO\figures\intra_picture\training_time_fmnist.png',
        format='png',
        dpi=300,
        bbox_inches='tight'
    )
    # 显示图形
    plt.show()


def generate_picture_minst(accuracy_list, random_timelist, average_timelist,
                           fair_timelist, reforcement_timelist, ours_timelist):
    # 指定精度列表
    target_accuracies = [96, 97, 98, 99]
    # 组织时间数据
    alltime_lists = [
        ours_timelist,  # The proportional share allocation algorithm (假设这是我们的方法)
        reforcement_timelist,  # Bid-price-first algorithm (假设这是强化学习)
        random_timelist,  # Random algorithm (假设这是随机算法)
        average_timelist,  # The proposed auction mechanism (假设这是平均机制)
        fair_timelist,  # Quality-aware algorithm (假设这是公平算法)

    ]

    # 方法名称列表
    methods = [
        "IRBAI+IRBSB",
        "IRBAI+IntraBMA",
        "IRBAI+Random",
        "IRBAI+FedAvg",
        "IRBAI+Fair",

    ]
    # 初始化时间列表
    time_lists = [[] for _ in methods]

    # 对于每个指定精度，找到第一个超过该精度的点并计算时间总和
    for target_accuracy in target_accuracies:
        for idx, discrete_time_list in enumerate(alltime_lists):
            index = find_first_exceed_index(accuracy_list, target_accuracy)
            if index is not None:
                cumulative_time = calculate_cumulative_time(discrete_time_list, index)
                time_lists[idx].append(cumulative_time)
            else:
                time_lists[idx].append(0)  # 如果没有找到，时间设为 0

    # 设置条形图的位置
    bar_width = 0.15
    r = [i for i in range(len(target_accuracies))]  # 只有 4 个目标精度
    colors = ['b', 'orange', 'g', 'r', 'purple']

    # 绘制每个方法的条形图
    for idx, times in enumerate(time_lists):
        plt.bar([x + idx * bar_width for x in r], times, color=colors[idx], width=bar_width, label=methods[idx])

    # 添加标题和标签
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Time (s)')


    # 设置 x 轴刻度
    plt.xticks([r + bar_width * 2 for r in range(len(target_accuracies))], target_accuracies)

    # 添加图例
    plt.legend()

    plt.savefig(
        r'E:\pycharm\EURO\figures\intra_picture\training_time_mnist.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    plt.savefig(
        r'E:\pycharm\EURO\figures\intra_picture\training_time_mnist.png',
        format='png',
        dpi=300,
        bbox_inches='tight'
    )
    # 显示图形
    plt.show()


def generate_picture_cifar10(accuracy_list, random_timelist, average_timelist,
                             fair_timelist, reforcement_timelist, ours_timelist):
    # 指定精度列表
    target_accuracies = [80, 85, 90, 95]
    # 组织时间数据
    alltime_lists = [
        ours_timelist,  # The proportional share allocation algorithm (假设这是我们的方法)
        reforcement_timelist,  # Bid-price-first algorithm (假设这是强化学习)
        random_timelist,  # Random algorithm (假设这是随机算法)
        average_timelist,  # The proposed auction mechanism (假设这是平均机制)
        fair_timelist,  # Quality-aware algorithm (假设这是公平算法)

    ]

    # 方法名称列表
    methods = [
        "IRBAI+IRBSB",
        "IRBAI+IntraBMA",
        "IRBAI+Random",
        "IRBAI+FedAvg",
        "IRBAI+Fair",
    ]
    # 初始化时间列表
    time_lists = [[] for _ in methods]

    # 对于每个指定精度，找到第一个超过该精度的点并计算时间总和
    for target_accuracy in target_accuracies:
        for idx, discrete_time_list in enumerate(alltime_lists):
            index = find_first_exceed_index(accuracy_list, target_accuracy)
            if index is not None:
                cumulative_time = calculate_cumulative_time(discrete_time_list, index)
                time_lists[idx].append(cumulative_time)
            else:
                time_lists[idx].append(0)  # 如果没有找到，时间设为 0

    # 设置条形图的位置
    bar_width = 0.15
    r = [i for i in range(len(target_accuracies))]  # 只有 4 个目标精度
    colors = ['b', 'orange', 'g', 'r', 'purple']

    # 绘制每个方法的条形图
    for idx, times in enumerate(time_lists):
        plt.bar([x + idx * bar_width for x in r], times, color=colors[idx], width=bar_width, label=methods[idx])

    # 添加标题和标签
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Time (s)')


    # 设置 x 轴刻度
    plt.xticks([r + bar_width * 2 for r in range(len(target_accuracies))], target_accuracies)

    # 添加图例
    plt.legend()

    plt.savefig(
        r'E:\pycharm\EURO\figures\intra_picture\raining_time_cifar10.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    plt.savefig(
        r'E:\pycharm\EURO\figures\intra_picture\training_time_cifar10.png',
        format='png',
        dpi=300,
        bbox_inches='tight'
    )
    # 显示图形
    plt.show()
def generate_picture_cifar100(accuracy_list, random_timelist, average_timelist,
                             fair_timelist, reforcement_timelist, ours_timelist):
    # 指定精度列表
    target_accuracies = [60, 70, 80, 90]
    # 组织时间数据
    alltime_lists = [
        ours_timelist,  # The proportional share allocation algorithm (假设这是我们的方法)
        reforcement_timelist,  # Bid-price-first algorithm (假设这是强化学习)
        random_timelist,  # Random algorithm (假设这是随机算法)
        average_timelist,  # The proposed auction mechanism (假设这是平均机制)
        fair_timelist,  # Quality-aware algorithm (假设这是公平算法)

    ]

    # 方法名称列表
    methods = [
        "IRBAI+IRBSB",
        "IRBAI+IntraBMA",
        "IRBAI+Random",
        "IRBAI+FedAvg",
        "IRBAI+Fair",
    ]
    # 初始化时间列表
    time_lists = [[] for _ in methods]

    # 对于每个指定精度，找到第一个超过该精度的点并计算时间总和
    for target_accuracy in target_accuracies:
        for idx, discrete_time_list in enumerate(alltime_lists):
            index = find_first_exceed_index(accuracy_list, target_accuracy)
            if index is not None:
                cumulative_time = calculate_cumulative_time(discrete_time_list, index)
                time_lists[idx].append(cumulative_time)
            else:
                time_lists[idx].append(0)  # 如果没有找到，时间设为 0

    # 设置条形图的位置
    bar_width = 0.15
    r = [i for i in range(len(target_accuracies))]  # 只有 4 个目标精度
    colors = ['b', 'orange', 'g', 'r', 'purple']

    # 绘制每个方法的条形图
    for idx, times in enumerate(time_lists):
        plt.bar([x + idx * bar_width for x in r], times, color=colors[idx], width=bar_width, label=methods[idx])

    # 添加标题和标签
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Time (s)')

    # 设置 x 轴刻度
    plt.xticks([r + bar_width * 2 for r in range(len(target_accuracies))], target_accuracies)

    # 添加图例
    plt.legend()

    plt.savefig(
        r'E:\pycharm\EURO\figures\intra_picture\training_time_cifar100.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    plt.savefig(
        r'E:\pycharm\EURO\figures\intra_picture\training_time_cifar100.png',
        format='png',
        dpi=300,
        bbox_inches='tight'
    )
    # 显示图形
    plt.show()