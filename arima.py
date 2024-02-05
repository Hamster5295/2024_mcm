import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.arima.model as arima
import statsmodels.tsa.stattools as sta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

"""
时间序列模型使用 statsmodels 库
变种相当的多，详细情况参考 https://zl;huanlan.zhihu.com/p/410812961
?>概括而言之：

ARIMA   -   有趋势，无季节，时间->单变量
SARIMA  -   有趋势，可以有季节，时间->单变量
SARIMAX -   有趋势，可以有季节，时间->单变量，可以输入 exog=外生变量（一维时间）
VARMA   -   无趋势，无季节，时间->多变量
VARMAX  -   无趋势，无季节，时间->多变量，可以输入 exog=外生变量（一维时间）
"""


def stationary_test(data, max_diff=5):
    """
    时间序列稳定性测试
    :param data: 时间序列(样本数, 时间采样数)
    :param max_diff: 最大尝试差分次数
    :return: (adf结果, 对应差分次数)
    """
    result = None
    level = 0
    diff = 0
    print("稳定性测试: \n" + "=" * 20)
    for i in range(max_diff):
        data_diffed = np.diff(data, i)
        adf = sta.adfuller(data_diffed)
        print(f"{i} 阶差分结果: \n{'T值':<10}{adf[0]:>10f}\n{'p-value':<10}{adf[1]:>10f}")
        print(adf[1])
        if adf[1] < 0.05:
            if adf[0] < adf[4]['1%']:
                print("致信概率 > 99%, 已采纳! ")
                result = adf
                diff = i
                break
            elif adf[0] < adf[4]['5%']:
                print("致信概率 > 95%")
                if level < 0.95:
                    level = 0.95
                    result = adf
                    diff = i
            elif adf[0] < adf[4]['10%']:
                print("致信概率 > 90%")
                if level < 0.9:
                    level = 0.9
                    result = adf
                    diff = i
            else:
                print("致信概率 < 90%, 未采纳")
        else:
            print("p-value > 0.05, 显著, 未采纳")
        print()
    print("=" * 20)
    return result, diff


def plot_acf_pacf(data):
    fig, axs = plt.subplots(2, 1)
    plot_acf(data, axs[0])
    plot_pacf(data, axs[1])
    plt.show()


def fit(data, pq=None):
    """
    训练 ARIMA 模型
    拿到模型后，使用 model.forcast(次数) 进行未知预测，越多越趋近均值；model.predict(start, end) 进行任意预测

    :param data: 训练数据，一维序列
    :param pq: 提供的 p 和 q 值，默认为 None 时将绘制 自相关&偏自相关图，并使用 AIC, BIC 自动分析 pq、训练相应模型; 提供值时将使用提供的值训练
    :return: 不提供pq时为 (model_aic, model_bic), 提供pq时为 model_pq
    """
    _, diff = stationary_test(data)
    data = np.diff(data, diff)

    if pq is None:
        plot_acf_pacf(data)
        print("\n自动分析p, q 取值")
        evaluation = sta.arma_order_select_ic(data, ic=['aic', 'bic'], trend='n')
        order_aic = evaluation['aic_min_order']
        order_bic = evaluation['bic_min_order']
        print(f"AIC: {order_aic}\nBIC: {order_bic}")

        model_aic = arima.ARIMA(data, order=(order_aic[0], diff, order_aic[1])).fit()
        model_bic = arima.ARIMA(data, order=(order_bic[0], diff, order_bic[1])).fit()
        print("模型训练完毕!\n" + "=" * 20)

        return model_aic, model_bic
    else:
        print("使用提供的 p, q")
        return arima.ARIMA(data, order=(pq[0], diff, pq[1])).fit()

# data = np.asarray([1, 2, 2, 3, 4, -1, 2, 3, 1, 4, -1, -4, -3, 1, -1, 0])
# model = fit(data, (1, 1))
#
# print(model.forecast(5))
#
# l = len(model.predict())
# print(model.predict(l, l + 4))
