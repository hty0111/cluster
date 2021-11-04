import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score as dunn
from sklearn.manifold import TSNE
import difflib
import seaborn as sns
from scipy.spatial.distance import pdist, euclidean


def get_wrong_date(data_set):
    """使用difflib库处理错误的车名"""
    pending_data = []
    for i in data_set:
        for j in data_set:
            x = difflib.SequenceMatcher(a=i, b=j).ratio()   # 返回两个序列的匹配程度
            if 0.6 < x < 1:
                pending_data.append(i)
                pending_data.append(j)
    return pd.DataFrame(pending_data)


def car_name_dict():
    """建立字典以替换错误的车名，注意到vw采用了缩写，同样进行处理"""
    dic = {'maxda rx3': 'mazda rx3', 'maxda glc deluxe': 'mazda glc deluxe',
           'Nissan versa': 'nissan versa', 'porcshce panamera': 'porsche panamera',
           'toyouta tercel': 'toyota tercel', 'vokswagen rabbit': 'volkswagen rabbit',
           'vw dasher': 'volkswagen dasher', 'vw rabbit': 'volkswagen rabbit'}
    return dic


def draw_boxplot(picture_set, data_set, is_processed=False):
    """绘制箱线图"""
    col = 4
    row = math.ceil(len(picture_set) / col)
    fig = plt.figure('Original boxplot', figsize=(20, 30), dpi=100)
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    for i in range(len(picture_set)):
        ax = fig.add_subplot(row, col, i + 1, title=picture_set[i])
        ax.boxplot(data_set[picture_set[i]].values)
    if is_processed:
        fig.savefig('../pictures/Processed boxplot')
    else:
        fig.savefig('../pictures/Original boxplot')
    plt.close(fig)


def PauTa(list_set, data_set):
    """使用拉伊达准则处理异常数据"""
    for feature in list_set:
        data_values = data_set[feature].values
        mean = data_values.mean()
        std = data_values.std()
        for i, value in enumerate(data_values):
            if abs(value - mean) > 3 * std:
                data_values[i] = mean + 3 * std     # 将大于3sigma的值替换为该上限
        data_set[feature] = data_values
    return data_set


def quartile(list_set, data_set):
    """将超过四分位数限制的数据替换为此边界"""
    for feature in list_set:
        data_values = np.array(data_set[feature].values)
        q1, q3 = np.percentile(data_values, [25, 75])
        iqr = q3 - q1
        ceil_value = q3 + 1.5 * iqr
        floor_value = q1 - 1.5 * iqr
        for i, value in enumerate(data_values):
            if value > ceil_value:
                data_values[i] = ceil_value
            if value < floor_value:
                data_values[i] = floor_value
        data_set[feature] = data_values
    return data_set


def to_digit(data_set):
    """将有序属性转化为对应数字"""
    to_digit_list = ['doornumber', 'cylindernumber', ]
    dic = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
           'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
           'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15}
    for i in to_digit_list:
        data_set[i] = data_set[i].replace(dic)
    return data_set


def normalization(data_set):
    """使用min-max标准化，将数据线性映射到[0, 1]"""
    col = data_set.columns
    for feature in col:
        data_values = np.array(data_set[feature].values)
        max_value = np.max(data_values)
        min_value = np.min(data_values)
        for i, value in enumerate(data_values):
            data_values[i] = (value - min_value) / (max_value - min_value)
        data_set[feature] = data_values
    return pd.DataFrame(data_set)


def draw_heatmap(data_set_corr):
    """绘制协方差热力图"""
    plt.figure('Heat map', figsize=(20, 15), dpi=150)
    sns.heatmap(data_set_corr, linewidths=0.3, square=True, annot=True, cmap="rainbow")
    plt.savefig('../pictures/Heat Map')
    plt.close()


def draw_variance(pca, data_set):
    """绘制每个变量对方差的贡献率"""
    data_num = data_set.shape[1]
    plt.figure('Variance contribution', figsize=(15, 10), dpi=50)
    x = np.array([i for i in range(data_num)])
    y = np.array([np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(data_num)])
    plt.plot(x, y)
    plt.xlabel('variable number')
    plt.ylabel('variance ratio')
    plt.grid()
    plt.savefig('../pictures/Variance contribution')
    plt.close()


def draw_sse(data_set):
    """手肘法绘图确定最佳K值"""
    x = np.arange(1, 20)
    sse = []
    for k in x:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_set)
        sse.append(kmeans.inertia_)
    plt.figure('Best K by SSE', figsize=(15, 10), dpi=50)
    plt.plot(x, sse, 'bo-')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.grid()
    plt.savefig('../pictures/Best K by SSE')
    plt.close()


# def draw_calinski(data_set):
#     """使用calinski_harabaz分数确定最佳K值"""
#     x = np.arange(1, 20)
#     sse = []
#     for k in x:
#         kmeans = KMeans(n_clusters=k)
#         labels = kmeans.fit_predict(data_set)
#         model = kmeans.fit(data_set)
#         pre = model.labels_
#         sse.append(metrics.calinski_harabasz_score(data_set, labels))
#     plt.figure('Best K by CH', figsize=(20, 15), dpi=50)
#     plt.plot(x, sse, 'bo-')
#     plt.xlabel('K')
#     plt.ylabel('CH')
#     plt.grid()
#     plt.savefig('../pictures/Best K by CH')
#     plt.close()


def draw_comparison(data, carName):
    car_name = pd.Series(carName).str.split(expand=True)[0]
    data_set = pd.concat((pd.DataFrame({'carName': car_name}), data), axis=1)
    fig = plt.figure("Comparison", figsize=(30, 40), dpi=100)
    sub = ['price', 'horsepower', 'curbweight', 'compressionratio', 'citympg', 'highwaympg']
    cnt = 1
    for i in sub:
        fig.add_subplot(3, 2, cnt, title=i)
        table = data_set.loc[:, ['carName', i]].groupby('carName').mean()
        value = table[i].values
        value.sort()
        sns.barplot(x=value, y=data_set['carName'].value_counts().index)
        # fig.subplots_adjust(left=0, bottom=1, right=2, top=200, wspace=0.2, hspace=0.1)
        cnt += 1
    plt.savefig('../pictures/Comparison')
    plt.close(fig)


def draw_TSNE2d(data, labels, cluster_num):
    plt.figure("Graphic model", figsize=(15, 10), dpi=100)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    label = np.array(labels.values).reshape(-1)

    # samples = [] * 100
    # labels = [samples] * 2
    # cluster = [labels] * cluster_num
    # for i in range(cluster_num):
    #     cluster[i] = np.array(result[label == i]).reshape(2, -1)
    #     plt.scatter(cluster[i][0], cluster[i][1], label=f'class, {i+1}')
    # plt.show()

    color = []
    cluster_1 = result[label == 0]
    cluster_2 = result[label == 1]
    cluster_3 = result[label == 2]
    cluster_4 = result[label == 3]
    cluster_5 = result[label == 4]
    cluster_6 = result[label == 5]
    cluster_7 = result[label == 6]
    # cluster_8 = result[label == 7]
    plt.scatter(cluster_1[:, 0], cluster_1[:, 1], c='r', label='class1')
    plt.scatter(cluster_2[:, 0], cluster_2[:, 1], c='g', label='class2')
    plt.scatter(cluster_3[:, 0], cluster_3[:, 1], c='b', label='class3')
    plt.scatter(cluster_4[:, 0], cluster_4[:, 1], c='c', label='class4')
    plt.scatter(cluster_5[:, 0], cluster_5[:, 1], c='m', label='class5')
    plt.scatter(cluster_6[:, 0], cluster_6[:, 1], c='y', label='class6')
    plt.scatter(cluster_7[:, 0], cluster_7[:, 1], c='k', label='class7')
    # plt.scatter(cluster_8[:, 0], cluster_8[:, 1], c='w', label='class8')
    plt.legend()
    plt.savefig(f'../pictures/visualized_{labels.columns[0][7:]}')
    plt.close()


def DaviesBouldin(X, labels):
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis=0) for k in cluster_k]
    # 求S
    S = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    Ri = []
    for i in range(n_cluster):
        Rij = []
        # 计算Rij
        for j in range(n_cluster):
            if j != i:
                r = (S[i] + S[j]) / euclidean(centroids[i], centroids[j])
                Rij.append(r)
        # 求Ri
        Ri.append(max(Rij))
    # 求dbi
    dbi = np.mean(Ri)
    return dbi

