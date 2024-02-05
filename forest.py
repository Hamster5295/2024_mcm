from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score


def scorer(func):
    def score(model, X, y):
        return func(y, model.predict(X))

    return score


def fit_rfc(X, y, n_estimators=100, criterion='gini', **kwargs):
    """
    训练随机森林分类器
    文档: https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
    :param X: 特征集 (样本数, 特征数)
    :param y: 标签集 (样本数) 或多类别分类的 (样本数, 类型数)
    :param n_estimators: “树”的数量
    :param criterion: 评价标准，['gini', 'entropy', 'log_loss']
    :return: 训练好的模型
    """

    X = X.astype(float)
    y = y.astype(float)
    model = (RandomForestClassifier(n_estimators, criterion=criterion, **kwargs))
    model.fit(X, y)

    print("Random Forest Classifier trained! ")
    print(f"Score: {cross_val_score(model, X, y)}")
    # print(f"MSE Loss: {cross_val_score(model, X, y, scoring=scorer(mean_squared_error))}")
    # print(f"RMSE Loss: {cross_val_score(model, X, y, scoring=scorer(root_mean_squared_error))}")
    # print(f"MAE Loss: {cross_val_score(model, X, y, scoring=scorer(mean_absolute_error))}")
    return model


def fit_rfr(X, y, n_estimators=100, criterion='squared_error', **kwargs):
    """
    训练随机森林拟合器
    文档: https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
    :param X: 特征集 (样本数, 特征数)
    :param y: 标签集 (样本数) 或多类别分类的 (样本数, 类型数)
    :param n_estimators: “树”的数量
    :param criterion: 评价标准，['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
    :return: 训练好的模型
    """

    X = X.astype(float)
    y = y.astype(float)

    model = RandomForestRegressor(n_estimators, criterion=criterion, **kwargs).fit(X, y)

    print("随机森林拟合器训练完成! ")
    print(f"Score: {cross_val_score(model, X, y)}")
    # print(f"MSE Loss: {cross_val_score(model, X, y, scoring=scorer(mean_squared_error))}")
    # print(f"RMSE Loss: {cross_val_score(model, X, y, scoring=scorer(root_mean_squared_error))}")
    # print(f"MAE Loss: {cross_val_score(model, X, y, scoring=scorer(mean_absolute_error))}")
    return model


def fit_hgbtc(X, y, **kwargs):
    """
    训练随机森林分类器
    文档: https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
    :param X: 特征集 (样本数, 特征数)
    :param y: 标签集 (样本数) 或多类别分类的 (样本数, 类型数)
    :param n_estimators: “树”的数量
    :return: 训练好的模型
    """

    X = X.astype(float)
    y = y.astype(float)

    model = HistGradientBoostingClassifier(**kwargs).fit(X, y)

    print("HGBT Classifier trained! ")
    print(f"Score: {cross_val_score(model, X, y)}")
    # print(f"MSE Loss: {cross_val_score(model, X, y, scoring=scorer(mean_squared_error))}")
    # print(f"RMSE Loss: {cross_val_score(model, X, y, scoring=scorer(root_mean_squared_error))}")
    # print(f"MAE Loss: {cross_val_score(model, X, y, scoring=scorer(mean_absolute_error))}")
    return model
