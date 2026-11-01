import numpy as np
def round_inter_compute(accuracy, B, T, C,Q):
    round_inter_budget = []
    pre_mean = 0
    total_sum = 0  # 用于存储前i个元素的总和
    round_inter_budget.append(B / T * 2)
    b = B / T * 2
    for i in range(len(accuracy) - 1):
        if accuracy[i] < Q:
            s = (accuracy[i] - pre_mean) * 0.01
            if s < 0:
                s = 0
            if i == len(accuracy) - 2:
                p = ((B - b) / (T - i - 1))
            else:
                p = ((B - b) / (T - i - 1)) *C ** min(s,0.5)
        else:
            p = ((B - b) / (T - i - 1))
        total_sum += accuracy[i]  # 更新总和
        pre_mean = total_sum / (i + 2)  # 计算前i个元素的平均值
        b += p
        round_inter_budget.append(np.around(p, 4))
    return round_inter_budget
def round_inter_compute_cifar(accuracy, B, T,C,Q):
    round_inter_budget = []
    pre_mean = 0
    total_sum = 0  # 用于存储前i个元素的总和
    round_inter_budget.append(B / T * 2)
    b = B / T * 2
    for i in range(len(accuracy) - 1):
        if accuracy[i] < Q:
            s = (accuracy[i] - pre_mean) * 0.01
            if s < 0:
                s = 0
            if i == len(accuracy) - 2:
                p = ((B - b) / (T - i - 1))
            else:
                p = ((B - b) / (T - i - 1)) * C ** s
        else:
            p = ((B - b) / (T - i - 1))
        total_sum += accuracy[i]  # 更新总和
        pre_mean = total_sum / (i + 2)  # 计算前i个元素的平均值
        b += p
        round_inter_budget.append(np.around(p, 4))
    return round_inter_budget

def round_inter_average(B, T):
    round_inter_budget = []
    for i in range(T):
        round_inter_budget.append(np.around(B / T, 4))
    return round_inter_budget

def round_inter_random(B, T):
    random_numbers = np.random.rand(T) * 10 + 0.5
    normalized_numbers = random_numbers / np.sum(random_numbers)
    scaled_numbers = normalized_numbers * B
    round_inter_budget = np.around(scaled_numbers, 4)
    return round_inter_budget
def round_intra_average(P, d_i ,c_i,a_i,sigma,T_up):
    p_i = P / len(d_i)
    p_average = np.sqrt(p_i*2*sigma*c_i*d_i*a_i)
    ck=2*np.square(sigma*c_i*d_i)*a_i
    dk=T_up
    t = ck / p_average + dk
    p_assigned = p_average*p_average/(2*sigma*c_i*d_i*a_i)
    return max(t),p_assigned,t

def round_intra_random(P, d_i ,c_i,a_i,sigma,T_up):
    random_numbers = np.random.rand(len(d_i)) * 1+0.25
    normalized_numbers = random_numbers / np.sum(random_numbers)
    p_i = normalized_numbers * P
    p_random = np.sqrt(p_i*2*sigma*c_i*d_i*a_i)
    ck = 2 * np.square(sigma * c_i * d_i) * a_i
    dk = T_up
    t = ck / p_random + dk
    p_assigned = p_random * p_random / (2 * sigma * c_i * d_i * a_i)
    return max(t),p_assigned,t
def round_intra_reforcement(p_i, d_i ,c_i,a_i,sigma,T_up):
    p_edu = np.sqrt(p_i*2*sigma*c_i*d_i*a_i)
    ck=2*np.square(sigma*c_i*d_i)*a_i
    dk=T_up
    t = ck / p_edu + dk
    return max(t)

def round_intra_compute(P, d_i ,c_i,a_i,sigma,T_up):
    Lk = min(2 * np.square(sigma * c_i * d_i) * a_i)
    Lm=max(2*sigma * c_i * d_i* a_i)

    Rk = max(2 * np.square(sigma * c_i * d_i) * a_i)
    Rm=min(2*sigma * c_i * d_i* a_i)

    T_up_max=max(T_up)
    T_up_min=min(T_up)

    L=max(Lk/np.sqrt(P/len(d_i)*Lm)+T_up_min,T_up_max)
    R=Rk/np.sqrt(P/len(d_i)*Rm)+T_up_max

    result=0
    budget=[]
    while L < R:
        M = (L+ R) / 2
        a=np.square((2 * np.square(sigma * c_i * d_i) * a_i)/(M-T_up))
        b=2*sigma * c_i * d_i*a_i
        budget=a/b
        B_ts=sum(budget)
        if  P - B_ts > 1e-4:
            R=M
        elif P - B_ts <0:
            L=M
        else:
            result=M
            break
    p_edu = np.sqrt(budget*2*sigma*c_i*d_i*a_i)
    ck=2*np.square(sigma*c_i*d_i)*a_i
    dk=T_up
    t = ck / p_edu + dk
    return result, budget,t

def fair (P, d_i ,c_i,a_i,sigma,T_up):
    bid=sigma*c_i*d_i+T_up
    p_i=bid/np.sum(bid)*P
    p_fair = np.sqrt(p_i*2*sigma*c_i*d_i*a_i)
    ck=2*np.square(sigma*c_i*d_i)*a_i
    dk = T_up
    t = ck / p_fair + dk
    p_assigned = p_fair * p_fair / (2 * sigma * c_i * d_i * a_i)
    return max(t),p_assigned,t