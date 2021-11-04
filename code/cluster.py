import pandas as pd

from my_function import *


if __name__ == '__main__':
    # 一、数据预处理

    # (1) 导入数据

    path = '../dataset/car_price.csv'
    org_data = pd.read_csv(path)    # 读取数据集
    # print(org_data.info())    查看信息
    # print(org_data.columns)   查看列标

    # (2) 处理车名

    # 获取车名和车名集合
    data = org_data.copy()
    car_name = data['CarName'].str.split(expand=True)[0]
    car_name_set = set(car_name)
    # print(car_name)
    pending_name = get_wrong_date(data_set=car_name_set)
    # print(pending_name)
    # 获取正确和错误的名字对，共五对
    pair_name = pending_name[pending_name.duplicated()].reset_index(drop=True)
    # print(pair_name)
    # 替换正确的车名
    data['CarName'] = data['CarName'].replace(car_name_dict())
    car_name = data['CarName'].values
    # print(car_name)
    # print(data['CarName'][50:60])     检查mazda替换

    # (3) 处理数据

    # 检查重复值，返回为0说明没有重复
    # print(data.duplicated().sum())

    # 检查缺失值，返回为0说明没有缺失
    # print(data.isnull().sum().sum())

    # 检查异常值并替换
    data = data.iloc[:, 3:]     # 丢弃前三列无关信息'car_ID','symboling','CarName'
    label = data.columns
    numeric_list = list(data[label].select_dtypes('number'))    # 取出值为数字的列
    # print(numeric_list)
    draw_boxplot(numeric_list, data)
    # PauTa(numeric_list, data)
    # 使用拉伊达准则效果不佳，采用四分位数的限制进行替换
    data = quartile(list_set=numeric_list, data_set=data)
    draw_boxplot(numeric_list, data, is_processed=True)

    draw_comparison(data, car_name)

    # 替换object类型对象
    # 有序属性转成对应数字，无序属性转独热编码
    data = to_digit(data)
    # print(data)
    non_numeric_list = list(data[label].select_dtypes('object'))
    # print(non_numeric_list)
    one_hot_data = pd.get_dummies(data[non_numeric_list])   # 转成独热编码
    # print(one_hot_data)
    # 丢弃 & 拼接
    for i in non_numeric_list:
        data = data.drop([i], axis=1)
    data = pd.concat([data, one_hot_data], axis=1)
    # print(data)

    # 归一化
    data = normalization(data)
    # print(data)

    # (4) PCA降维

    # 绘制相关性热力图
    data_corr = data.corr()
    draw_heatmap(data_corr)

    # 计算并绘制每个成分的方差贡献率
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)
    draw_variance(pca, data)

    threshold = 0.9        # 设置保留信息的阈值
    pca = PCA(threshold)
    data = pd.DataFrame(pca.fit_transform(data))
    # print(data)

    # 二、搭建模型

    # (1) 确定分类组数

    # 拐点图
    draw_sse(data)
    # CH分数
    # draw_calinski(data)
    cluster_num = 8       # 观察图形得最佳的分组数

    # (2) k-means

    k_means = KMeans(n_clusters=cluster_num, init='k-means++')
    labels_KMeans = k_means.fit_predict(data)
    # print(labels_KMeans)
    # print(labels_KMeans.shape)
    labels_KMeans = pd.DataFrame({'labels_KMeans': labels_KMeans})
    results_KMeans = pd.concat((org_data, labels_KMeans), axis=1)     # 将聚类结果添加到原始数据最后
    results_KMeans['CarName'] = car_name
    # print(results_KMeans)
    results_KMeans = results_KMeans.sort_values(by='labels_KMeans')
    results_KMeans.to_csv('../results/KMeans.csv', index=False)

    # (3) GMM

    gm = GaussianMixture(n_components=cluster_num)
    labels_GM = gm.fit_predict(data)
    labels_GM = pd.DataFrame({'labels_GM': labels_GM})
    results_GM = pd.concat((org_data, labels_GM), axis=1)
    results_GM['CarName'] = car_name
    results_GM = results_GM.sort_values(by='labels_GM')
    results_GM.to_csv('../results/GM.csv', index=False)

    # (4) DBSCAN
    db = DBSCAN(eps=0.6, min_samples=3)    # 设置半径和最小样本数
    labels_db = db.fit_predict(data)
    labels_db = pd.DataFrame({'labels_db': labels_db})
    results_db = pd.concat((org_data, labels_db), axis=1)
    results_db['CarName'] = car_name
    results_db = results_db.sort_values(by='labels_db')
    results_db.to_csv('../results/db.csv', index=False)

    draw_TSNE2d(data, labels_KMeans, cluster_num)
    draw_TSNE2d(data, labels_GM, cluster_num)
    draw_TSNE2d(data, labels_db, cluster_num)

    dunn_score_KMeans = dunn(data, labels_KMeans)    # 计算邓恩指数
    print('DI of KMeans: ', dunn_score_KMeans)
    dunn_score_GM = dunn(data, labels_GM)
    print('DI of GM: ', dunn_score_GM)
    dunn_score_db = dunn(data, labels_db)
    print('DI of db: ', dunn_score_db)

    # dbi = DaviesBouldin(data, labels_KMeans)
    # print('dbi:', dbi)



