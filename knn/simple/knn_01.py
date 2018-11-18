#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 下午10:05
# @Author  : liumingyu
# @Site    : 
# @File    : knn_01.py

import operator

import numpy as np

"""
函数说明:创建数据集

Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
"""


def create_data_set():
    # 四组二维特征
    groups = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return groups, labels


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


if __name__ == '__main__':
    # 创建数据集
    group, label = create_data_set()
    # 测试集
    test = [51, 47]
    # knn分类
    test_class = classify0(test, group, label, 3)
    # 打印分类结果
    print(test_class)
