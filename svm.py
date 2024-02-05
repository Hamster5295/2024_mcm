from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.calibration import CalibratedClassifierCV


def fit_svc(X, y, kernel="rbf", probability=True, plot=True, legends=None, **kwargs):
    """
    SVM 支持向量机分类器（非线性版本）
    线性请使用 fit_svc_linear()
    :param X: 特征集 (样本数, 特征数)
    :param y: 标签集 (样本数) 或多类别分类的 (样本数, 类型数)
    :param kernel: 核函数[rbf, poly, sigmoid, linear(不推荐)]
    详情见 https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
    :param probability: 使用概率进行分类
    :param plot: 绘制分类图像
    :return: 训练好的模型
    """
    model = SVC(kernel=kernel, probability=probability, **kwargs)
    model.fit(X, y)

    print(f"{kernel}内核 SVM 分类器训练完毕! ")
    print(f"得分: {model.score(X, y)} / 1")

    if plot:
        plot_svc(model, X, y, legend_labels=legends)

    return model


def fit_svc_linear(X, y, **kwargs):
    """
    SVM 支持向量机分类器（线性版本），对线性数据具有更强分类能力
    非线性请使用 fit_svc()
    :param X: 特征集 (样本数, 特征数)
    :param y: 标签集 (样本数) 或多类别分类的 (样本数, 类型数)
    :return: 训练好的模型
    """

    model = LinearSVC(dual='auto', **kwargs).fit(X, y)

    print(f"线性 SVM 分类器训练完毕! ")
    print(f"得分: {model.score(X, y)} / 1")

    return model


def fit_svr(X, y, kernel='rbf', degree=3, **kwargs):
    """
    SVM 支持向量机拟合器（非线性版本），对非线性数据具有更强拟合能力
    线性请使用 fit_svr_linear()
    如果希望回归，不妨尝试带核岭回归 https://scikit-learn.org/stable/modules/kernel_ridge.html
    :param X: 特征集 (样本数, 特征数)
    :param y: 标签集 (样本数) 或多类别分类的 (样本数, 类型数)
    :param kernel: 核函数
    :param degree: poly 核函数的阶数
    :return: 训练好的模型
    """

    model = SVR(kernel=kernel, degree=degree, **kwargs).fit(X, y)

    print(f"{kernel}内核 SVM 拟合器训练完毕! ")
    print(f"得分: {model.score(X, y)} / 1")

    return model


def fit_svr_linear(X, y, **kwargs):
    """
    SVM 支持向量机分类器（线性版本），对线性数据具有更强分类能力
    非线性请使用 fit_svr()
    :param X: 特征集 (样本数, 特征数)
    :param y: 标签集 (样本数) 或多类别分类的 (样本数, 类型数)
    :return: 训练好的模型
    """

    model = LinearSVR(dual='auto', **kwargs).fit(X, y)

    print(f"线性 SVM 拟合器训练完毕! ")
    print(f"得分: {model.score(X, y)} / 1")

    return model


def plot_svc(model, X, y, legend_labels=None):
    fig, ax = plt.subplots()

    common_params = {"estimator": model, "X": X, "ax": ax}

    # 绘制填充
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict_proba",
        plot_method="contourf",
        alpha=0.3,
    )

    # 绘制包络线
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # 绘制支持向量高亮
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=250,
        facecolors="none",
        edgecolors="k",
    )

    # 绘制样本点
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")

    legend_style = {'loc': "upper right", 'title': "Classes"}

    if legend_labels is not None:
        handles, _ = scatter.legend_elements()
        ax.legend(handles=handles, labels=legend_labels, **legend_style)
    else:
        ax.legend(*scatter.legend_elements(), **legend_style)

    plt.show()
    return ax
