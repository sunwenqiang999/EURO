import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
def find_first_exceed_index(accuracy_list, target_accuracy):
    """找到第一个超过目标精度的索引"""
    for i, accuracy in enumerate(accuracy_list):
        if accuracy >= target_accuracy:
            return i
    return None


def calculate_cumulative_time(time_list, index):
    """计算索引及其之前所有时间的总和"""
    return sum(time_list[:index + 1])
def generate_picture_pre_fmnist(accuracy_list, decrease_timelist, average_timelist,
                            increase_timelist):

    threshold = 90
    index_90 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_90)
    accuracy_list = accuracy_list[:index_90+2]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    decrease_time = np.cumsum(decrease_timelist[:index_90+2])
    average_time = np.cumsum(average_timelist[:index_90+2])
    increase_time = np.cumsum(increase_timelist[:index_90+2])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(decrease_time, accuracy_list, label='Decreasing budget allocation', color='blue')
    plt.plot(average_time, accuracy_list, label='Uniform budget allocation', color='orange')
    plt.plot(increase_time, accuracy_list, label='Increasing budget allocation', color='green')

    plt.axvline(x=decrease_time[index_90 - 1], color='b', linestyle='--')
    # 添加注释显示x轴的具体值
    plt.annotate(f'Decrease\nTime(s): {decrease_time[index_90 - 1]:.1f}',
                 xy=(decrease_time[index_90 - 1], accuracy_list[index_90 - 1]),
                 xytext=(decrease_time[index_90 - 1], accuracy_list[index_90 - 1] + 1.5),  # 调整文字位置
                 arrowprops=dict(facecolor='black', shrink=0.1, linewidth=1, headwidth=5, headlength=5),
                 fontsize=10)

    plt.axvline(x=average_time[index_90 - 1], color='orange', linestyle='--')
    # 添加注释显示x轴的具体值
    plt.annotate(f'Average\nTime(s): {average_time[index_90 - 1]:.1f}',
                 xy=(average_time[index_90 - 1], accuracy_list[index_90 - 1]),
                 xytext=(average_time[index_90 - 1] + 50, accuracy_list[index_90 - 1] + 1.5),  # 调整文字位置
                 arrowprops=dict(facecolor='black', shrink=0.1, linewidth=1, headwidth=5, headlength=5),
                 fontsize=10)

    plt.axvline(x=increase_time[index_90 - 1], color='g', linestyle='--')
    # 添加注释显示x轴的具体值
    plt.annotate(f'Increase\nTime(s): {increase_time[index_90 - 1]:.1f}',
                 xy=(increase_time[index_90 - 1], accuracy_list[index_90 - 1]),
                 xytext=(increase_time[index_90 - 1] + 40, accuracy_list[index_90 - 1] + 1.5),  # 调整文字位置
                 arrowprops=dict(facecolor='black', shrink=0.1, linewidth=1, headwidth=5, headlength=5),
                 fontsize=10)

    # 标记准确度刚好达到90%的点
    plt.scatter(decrease_time[index_90 - 1], accuracy_list[index_90 - 1], color='b', zorder=5)
    plt.scatter(average_time[index_90 - 1], accuracy_list[index_90 - 1], color='orange', zorder=5)
    plt.scatter(increase_time[index_90 - 1], accuracy_list[index_90 - 1], color='g', zorder=5)
    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\pre_picture\\accVstime_fmnist.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\pre_picture\\accVstime_fmnist.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()

def generate_picture_pre_bar_fmnist(accuracy_list, decrease_timelist, average_timelist,
                            increase_timelist):
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
        decrease_timelist,  # The proportional share allocation algorithm (假设这是我们的方法)
        average_timelist,  # The proposed auction mechanism (假设这是平均机制)
        increase_timelist,  # Quality-aware algorithm (假设这是公平算法)

    ]

    # 方法名称列表
    methods = [
        "Decreasing budget allocation",
        "Uniform budget allocation",
        "Increasing budget allocation",
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
    bar_width = 0.25
    r = [i for i in range(len(target_accuracies))]  # 只有 4 个目标精度
    colors = ['b', 'orange', 'g']

    # 绘制每个方法的条形图
    for idx, times in enumerate(time_lists):
        plt.bar([x + idx * bar_width for x in r], times, color=colors[idx], width=bar_width, label=methods[idx])
    # 设置 x 轴刻度（调整为居中位置）
    plt.xticks([x + bar_width for x in r], target_accuracies)
    # 添加标题和标签
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Time (s)')


    # 添加图例
    plt.legend()
    plt.savefig(
        r'E:\pycharm\EURO\figures\pre_picture\training_time_fminst.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        r'E:\pycharm\EURO\figures\pre_picture\training_time_fmnist.png',
        format='png',
        dpi=300,
        bbox_inches='tight'
    )
    # 显示图形
    plt.show()
def generate_picture_pre_cifar10(accuracy_list, decrease_timelist, average_timelist,
                            increase_timelist):

    threshold = 96
    index_90 = np.argmax(accuracy_list >= threshold)
    print('index_threshold:', index_90)
    accuracy_list = accuracy_list[:index_90+3]
    #如果时间列表是每轮的时间，需要先做累加得到累计时间
    decrease_time = np.cumsum(decrease_timelist[:index_90+3])
    average_time = np.cumsum(average_timelist[:index_90+3])
    increase_time = np.cumsum(increase_timelist[:index_90+3])

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(decrease_time, accuracy_list, label='Decreasing budget allocation', color='blue')
    plt.plot(average_time, accuracy_list, label='Uniform budget allocation', color='orange')
    plt.plot(increase_time, accuracy_list, label='Increasing budget allocation', color='green')

    plt.axvline(x=decrease_time[index_90 - 1], color='b', linestyle='--')
    # 添加注释显示x轴的具体值
    plt.annotate(f'Decrease\nTime(s): {decrease_time[index_90 - 1]:.1f}',
                 xy=(decrease_time[index_90 - 1], accuracy_list[index_90 - 1]),
                 xytext=(decrease_time[index_90 - 1], accuracy_list[index_90 - 1] + 7),  # 调整文字位置
                 arrowprops=dict(facecolor='black', shrink=0.1, linewidth=1, headwidth=5, headlength=5),
                 fontsize=10)

    plt.axvline(x=average_time[index_90 - 1], color='orange', linestyle='--')
    # 添加注释显示x轴的具体值
    plt.annotate(f'Average\nTime(s): {average_time[index_90 - 1]:.1f}',
                 xy=(average_time[index_90 - 1], accuracy_list[index_90 - 1]),
                 xytext=(average_time[index_90 - 1] + 230, accuracy_list[index_90 - 1] + 7),  # 调整文字位置
                 arrowprops=dict(facecolor='black', shrink=0.1, linewidth=1, headwidth=5, headlength=5),
                 fontsize=10)

    plt.axvline(x=increase_time[index_90 - 1], color='g', linestyle='--')
    # 添加注释显示x轴的具体值
    plt.annotate(f'Increase\nTime(s): {increase_time[index_90 - 1]:.1f}',
                 xy=(increase_time[index_90 - 1], accuracy_list[index_90 - 1]),
                 xytext=(increase_time[index_90 - 1] + 130, accuracy_list[index_90 - 1] + 7),  # 调整文字位置
                 arrowprops=dict(facecolor='black', shrink=0.1, linewidth=1, headwidth=5, headlength=5),
                 fontsize=10)

    # 标记准确度刚好达到90%的点
    plt.scatter(decrease_time[index_90 - 1], accuracy_list[index_90 - 1], color='b', zorder=5)
    plt.scatter(average_time[index_90 - 1], accuracy_list[index_90 - 1], color='orange', zorder=5)
    plt.scatter(increase_time[index_90 - 1], accuracy_list[index_90 - 1], color='g', zorder=5)
    # 设置标签和标题
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        r'E:\pycharm\EURO\\figures\\pre_picture\\accVstime_cifar10.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )
    # 保存图片（可选）
    plt.savefig('E:\pycharm\EURO\\figures\\pre_picture\\accVstime_cifar10.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()

def generate_picture_pre_bar_cifar10(accuracy_list, decrease_timelist, average_timelist,
                            increase_timelist):

    # 指定精度列表
    target_accuracies = [80, 85, 90, 95]
    # 组织时间数据
    alltime_lists = [
        decrease_timelist,  # The proportional share allocation algorithm (假设这是我们的方法)
        average_timelist,  # The proposed auction mechanism (假设这是平均机制)
        increase_timelist,  # Quality-aware algorithm (假设这是公平算法)

    ]

    # 方法名称列表
    methods = [
        "Decreasing budget allocation",
        "Uniform budget allocation",
        "Increasing budget allocation",
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
    bar_width = 0.25
    r = [i for i in range(len(target_accuracies))]  # 只有 4 个目标精度
    colors = ['b', 'orange', 'g']

    # 绘制每个方法的条形图
    for idx, times in enumerate(time_lists):
        plt.bar([x + idx * bar_width for x in r], times, color=colors[idx], width=bar_width, label=methods[idx])
    # 设置 x 轴刻度（调整为居中位置）
    plt.xticks([x + bar_width for x in r], target_accuracies)
    # 添加标题和标签
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Time (s)')


    # 添加图例
    plt.legend()
    plt.savefig(
        r'E:\pycharm\EURO\figures\pre_picture\training_time_cifar10.eps',
        format='eps',
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        r'E:\pycharm\EURO\figures\pre_picture\training_time_cifar10.png',
        format='png',
        dpi=300,
        bbox_inches='tight'
    )
    # 显示图形
    plt.show()
