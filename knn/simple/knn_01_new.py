#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/18 下午3:58
# @Author  : liumingyu
# @Site    : 
# @File    : knn_01_new.py

import collections

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
    label = ['爱情片', '爱情片', '动作片', '动作片']
    return groups, label


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


def classify0(inx, data_set, label, k):
    # 计算距离
    dist = np.sum((inx - data_set) ** 2, axis=1) ** 0.5
    # k个最近的标签
    k_labels = [label[index] for index in dist.argsort()[0:k]]
    # 出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label


if __name__ == '__main__':
    # 创建数据集
    group, labels = create_data_set()
    # 测试集
    test_set = [51, 47]
    # knn分类
    test_class = classify0(test_set, group, labels, 3)
    # 打印分类结果
    print(test_class)
