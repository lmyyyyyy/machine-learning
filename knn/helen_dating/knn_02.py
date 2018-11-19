#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/18 下午4:09
# @Author  : liumingyu
# @Site    : 
# @File    : knn_02.py

import operator

import matplotlib.lines as m_lines
import matplotlib.pyplot as m_plt
import numpy as np
from matplotlib.font_manager import FontProperties

"""
函数说明:打开并解析文件,对数据进行分类:1-不喜欢,2-魅力一般,3-极具魅力

Parameters:
    filename - 文件名
Returns:
    return_mat - 特征矩阵
    class_label_vector - 分类label向量
"""


def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    array_lines = fr.readlines()
    # 得到文件行数
    number_lines = len(array_lines)
    # 返回的numpy矩阵,解析完成的数据:number_lines行,3列的被0填充的矩阵
    return_mat = np.zeros((number_lines, 3))
    # 定义返回的分类标签向量
    class_label_vector = []
    # 行的索引值
    index = 0
    for line in array_lines:
        # s.strip(rm) 当rm为空时,默认删除空白字符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片
        list_from_line = line.split('\t')
        # 将数据前三列提取出来,存放到return_mat的numpy矩阵中,也就是特征矩阵
        return_mat[index, :] = list_from_line[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if list_from_line[-1] == 'didntLike':
            class_label_vector.append(1)
        elif list_from_line[-1] == 'smallDoses':
            class_label_vector.append(2)
        elif list_from_line[-1] == 'largeDoses':
            class_label_vector.append(3)
        index += 1
    return return_mat, class_label_vector


"""
函数说明:数据可视化

Parameters:
    dating_data_mat - 特征矩阵
    dating_labels - 分类label
Returns:
    无
"""


def show_datas(dating_data_mat, dating_labels):
    # 设置汉字格式
    font = FontProperties(fname=r"/Library/Fonts/Songti.ttc", size=12)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrows=2,ncols=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = m_plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    labels_colors = []
    for i in dating_labels:
        if i == 1:
            labels_colors.append('black')
        if i == 2:
            labels_colors.append('orange')
        if i == 3:
            labels_colors.append('red')
    # 画出散点图,以dating_data_mat矩阵的第一列(飞行常客里程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 1], color=labels_colors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    m_plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    m_plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    m_plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以dating_data_mat矩阵的第一列(飞行常客里程)、第二列(冰淇淋)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 2], color=labels_colors, s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰淇淋公升数', FontProperties=font)
    m_plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    m_plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    m_plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以dating_data_mat矩阵的第二列(玩游戏)、第三列(冰淇淋)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=dating_data_mat[:, 1], y=dating_data_mat[:, 2], color=labels_colors, s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰淇淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰淇淋公升数', FontProperties=font)
    m_plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    m_plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    m_plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例
    didnt_like = m_lines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    small_doses = m_lines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    large_doses = m_lines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')

    # 添加图例
    axs[0][0].legend(handles=[didnt_like, small_doses, large_doses])
    axs[0][1].legend(handles=[didnt_like, small_doses, large_doses])
    axs[1][0].legend(handles=[didnt_like, small_doses, large_doses])

    # 显示图片
    m_plt.show()


"""
函数说明:对数据进行归一化

Parameters:
    data_set - 特征矩阵
Returns:
    data_set - 特征矩阵
    ranges - 数据范围
    min_vals - 数据最小值
"""


def auto_norm(data_set):
    # 获得数据的最小值
    min_vals = data_set.min(0)
    # 获得数据的最大值
    max_vals = data_set.max(0)
    # 最小值和最大的值间的范围
    ranges = max_vals - min_vals
    # shape(data_set)返回data_set的矩阵行列数,构建一个相同规格的矩阵
    norm_data_set = np.zeros(np.shape(data_set))
    # 返回data_set的行数
    m = data_set.shape[0]
    # 原始值减去最小值
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    # 除以最大和最小值的差,得到归一化数据
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return norm_data_set, ranges, min_vals


"""
函数说明:knn算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labels - 分类标签
    k - knn算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""


def classify0(inx, data_set, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    data_set_size = data_set.shape[0]
    # 在列向量防线重复inX一次(横向),行向量方向上重复inX共data_set_size次(纵向)
    diff_mat = np.tile(inx, (data_set_size, 1)) - data_set
    # 二维特征相减后平方
    sq_diff_mat = diff_mat ** 2
    # sum()所有元素相加,sum(0):列相加,sum(1):行相加
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开方,计算出距离
    distances = sq_distances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sorted_dist_indices = distances.argsort()
    # 定义一个记录类别次数的字典
    class_count = {}
    for i in range(k):
        # 取出前k个元素的类别
        vote_label = labels[sorted_dist_indices[i]]
        # dict.get(key, default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值
        # 计算类别次数
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1):根据字典的值进行排序
    # key=operator.itemgetter(0):根据字典的键进行排序
    # reverse=true:降序排序字典 false:升序排序字典
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所有分类的类别
    return sorted_class_count[0][0]


"""
函数说明:分类器测试函数

Parameters:
    无
Returns:
    norm_data_set - 归一化后的矩阵特征
    ranges - 数据范围
    min_vals - 数据最小值
"""


def dating_class_test():
    # 需要打开的文件名称
    file_name = 'datingTestSet.txt'
    # 打开并处理数据
    dating_data_mat, dating_labels = file2matrix(file_name)
    # 取所有数据的百分之十
    ho_ratio = 0.10
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # 获得归一化后矩阵的行数
    m = norm_mat.shape[0]
    # 百分之十的测试数据的个数
    num_test_vecs = int(m * ho_ratio)
    # 定义分类错误技术
    error_count = 0.0

    for i in range(num_test_vecs):
        # 前num_test_vecs个数据作为测试集,后m-num_test_vecs个数据作为训练集
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 4)
        print("分类结果: %d\t真实类别: %d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("错误率: %f%%" % (error_count / float(num_test_vecs) * 100))


def classify_person():
    # 所有可能结果集
    result_list = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征-用户输入
    percentage = float(input("玩视频游戏所消耗时间百分比: "))
    ff_miles = float(input("每年获得的飞行常客里程数: "))
    ice_cream = float(input("每周消费的冰淇淋公升数: "))
    # 打开的文件名
    file_name = "datingTestSet.txt"
    # 打开并处理数据
    dating_data_mat, dating_labels = file2matrix(file_name)
    # 训练集归一化
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # 生成numpy数组-测试集
    in_arr = np.array([percentage, ff_miles, ice_cream])
    # 测试集归一化
    norm_in_arr = (in_arr - min_vals) / ranges
    # 返回分类结果
    classifier_result = classify0(norm_in_arr, norm_mat, dating_labels, 3)
    # 输出结果
    print("你可能%s这个人" % (result_list[classifier_result - 1]))


if __name__ == '__main__':
    # filename = 'datingTestSet.txt'
    # dating_data_mat, dating_labels = file2matrix(filename)
    # show_datas(dating_data_mat, dating_labels)
    # dating_class_test()
    classify_person()
