import math

import numpy as np


def entropy(data, positive, negative):
    """
    熵权分析法

    :param data: 数据，二维数组，每行为一个样本，每列为一个特征
    :param positive: 正向特征，列的下标数组
    :param negative: 负向特征，列的下标数组
    :return: 信息熵值，信息效用
    """

    data[:, positive] = (data[:, positive] - np.min(data[:, positive])) / (
        np.max(data[:, positive] - np.min(data[:, positive])))
    data[:, negative] = (np.max(data[:, negative]) - data[:, negative]) / (
        np.max(data[:, negative] - np.min(data[:, negative])))

    for i in data.shape[1]:
        sum_col = np.sum(data[:, i])
        data[:, i] = data[:, i] / sum_col
    data[data == 0] = 1e-6

    k = 1 / np.log(data.shape[0])
    h = -k * np.sum(data, axis=1)
    g = 1 - h
    return h, g


def topsis(data, bests):
    """
    Topsis 数据评价

    :param data: 数据，一行一个样本，一列一个特征
    :param bests: 特征的最优值：越大越好->math.inf，越小越好-> -math.inf，中值最好->中值，区间最好->(下界，上界)
    :return: 评价完的数据，一个样本有一个评分
    """

    for i in range(len(bests)):
        best = bests[i]

        # 越大越好
        if best == math.inf:
            continue

        # 越小越好
        elif best == -math.inf:
            data[:, i] = np.max(data[:, i]) - data[:, i]

        elif type(best) == tuple:
            col = data[:, i]
            m = np.max([best[0] - np.min(col), np.max(col) - best[1]])

            col[col < best[0]] = 1 - np.abs(best[0] - col[col < best[0]]) / m
            col[(col < best[1]) & (col > best[0])] = 1
            col[col > best[1]] = 1 - np.abs(col[col > best[1]] - best[1]) / m

            data[:, i] = col

        else:
            col = data[:, i]
            m = np.max(np.abs(col - best))
            col = 1 - np.abs(col - best) / m
            data[:, i] = col

    # 标准化
    data = data / np.sqrt(np.sum(data, axis=0))

    # Topsis
    max_col = np.max(data, axis=0)
    min_col = np.min(data, axis=0)
    max_dis = np.sqrt(np.sum((max_col - data) ** 2, axis=1))
    min_dis = np.sqrt(np.sum((min_col - data) ** 2, axis=1))
    result = min_dis / (max_dis + min_dis)
    return result.reshape(-1, 1)


data = np.asarray([[1, 3],
                   [5, 2],
                   [8, 1]])
# print((1, 2) == tuple)
print(topsis(data, [(3, 7), -math.inf]))
