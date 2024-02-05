import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from statsmodels.api import Logit

import hio
from feature import extract_features_althletes

# 读取原始数据
data = hio.read_csv('data_ori.csv', skiprows=1)

# 划定需要分析的运动员范围(实际上是全部)
althletes = np.unique(data[:, 1:3].reshape(-1))

# 特征提取，按列分别为一盘比赛中的：最大连胜次数，发球得分总数，接发球得分总数，发出好球频率，非强制误差频率
_, features_sheet = extract_features_althletes(data, althletes)

# 标准化
X, y = features_sheet[:, :-1], features_sheet[:, -1]
X = MinMaxScaler().fit_transform(X)

# 逻辑回归
model = Logit(y, X)
result = model.fit(disp=0)

# 输出回归结果
print(f"Params: {result.params}")
print(f"P-values: {result.pvalues}")
print(result.prsquared)
print(result.summary())

"""
Output:
================
Optimization terminated successfully.
         Current function value: 0.625050
         Iterations 5
Params: [-2.88136255  1.69837211  4.33355369 -1.85164661 -1.66897033]
P-values: [2.56526596e-03 4.57137183e-02 4.83453226e-06 4.52804307e-03
 2.06757156e-02]

"""
# # 取最相关的3个特征与标签的关系，绘制3D散点图
#
# # Pick col 0, 2, 3 for plotting
# xs = X[:, 0]
# ys = X[:, 2]
# zs = X[:, 3]
#
# # pred = model.predict(exog=)
#
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# ax.scatter(xs[y == 1], ys[y == 1], zs[y == 1], c='#FF9997')
# ax.scatter(xs[y == 0], ys[y == 0], zs[y == 0], c='#A08CEF')
#
# ax.set_xlabel('Maximum Combo Points')
# ax.set_ylabel("Points won at opponent's serving")
# ax.set_zlabel('ACE density peaks')
# ax.legend(['Set Won', 'Set Lost'])
# plt.show()
#
# # 保存模型
# with open("model_logistic.pkl", 'wb') as file:
#     pickle.dump(model, file)
