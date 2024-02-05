import numpy as np
import pandas as pd
import scipy.io as scio
from pandas import DataFrame


def read_mat(path):
    """
    读取 .mat 文件
    读取到的是字典，使用 dict['变量名'] 访问变量
    :param path:
    :return:
    """
    return scio.loadmat(path)


def write_mat(path, data):
    """
    保存数据进 .mat 文件
    数据格式需要为 {变量名: 内容}
    :param data: 数据
    :param path: 路径
    """
    scio.savemat(path, data)


def read_csv(path, dtype=None, delimiter=',', skiprows=0, fillna=0):
    """
    读取 .csv 文件，自动去除 NaN 并替换为 0, 并转化为 nparray
    :param path: 文件路径
    :param dtype: 读取数据类型
    :param delimiter: 分隔符，默认为 ','
    :param skiprows: 跳过开头几行
    :return: 数据，nparray
    """
    return (pd.read_csv(path, dtype=dtype, delimiter=delimiter, skiprows=skiprows, header=None)
            .fillna(fillna).to_numpy())


def write_csv(path, data):
    """
    保存 .csv 文件
    :param path: 路径
    :param data: 文件
    :return:
    """
    np.savetxt(path, data)


def read_excel(path):
    """
    读取 excel
    :param path: 路径
    :return: 数据, ndarray
    """
    return pd.read_excel(path, header=None).fillna(0).to_numpy()


def write_excel(path, data):
    """
    写入 excel
    :param path: 路径
    :param data: 数据
    :return:
    """
    DataFrame(data).to_excel(path, header=None, index=False)
