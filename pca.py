from sklearn.decomposition import PCA


def fit(X, components):
    """
    PCA 主成分分析
    文档： https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
    :param X: 原始数据，形状为 (sample, feature)
    :param components: 分为几个成分
    :return: 新数据，PCA模型，新数据各列对数据区别的贡献
    """

    # 模型: PCA
    model = PCA(n_components=components).fit(X)
    result = model.transform(X)

    print("PCA 完成! ")

    return result, model, model.explained_variance_ratio_
