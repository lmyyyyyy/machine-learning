#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 下午4:19
# @Author  : liumingyu
# @Site    : 
# @File    : knn_03.py

import operator
import time
from os import listdir

import numpy as np

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
函数说明: 将32*32的二进制图像转换为1*1024向量

Parameters:
    filename - 文件名
Returns:
    return_vector - 返回的二进制图像的1*1024向量
"""


def img2vector(filename):
    # 创建1*1024的零向量
    return_vector = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        line = fr.readline()
        # 每一行的前32个元素依次添加到return_vector中
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line[j])
    # 返回转换后的1*1024向量
    return return_vector


"""
函数说明: 手写数字分类测试

Parameters:
    无
Returns:
    无
"""


def hand_writing_class_test():
    # 定义测试集的分类集合
    hand_write_labels = []
    # 返回train_digits目录下的文件名列表
    training_file_list = listdir('train_digits')
    # 返回训练集文件列表的数量
    train_file_count = len(training_file_list)
    # 初始化训练矩阵
    training_mat = np.zeros((train_file_count, 1024))
    for i in range(train_file_count):
        # 遍历返回训练集文件的名称
        file_name_str = training_file_list[i]
        # 获得分类的数字
        class_number = int(file_name_str.split('_')[0])
        # 存入分类集合中
        hand_write_labels.append(class_number)
        # 将每个1*1024数据存储到training_mat中
        training_mat[i, :] = img2vector('train_digits/%s' % file_name_str)
    # 返回test_digits目录下的文件名列表
    test_file_list = listdir('test_digits')
    # 定义错误计数
    error_count = 0.0
    # 返回测试集文件列表的数量
    test_file_count = len(test_file_list)
    for i in range(test_file_count):
        # 遍历返回测试测试集文件的名称
        file_name_str = test_file_list[i]
        # 获得正确分类数字
        class_number = int(file_name_str.split('_')[0])
        # 获得测试集的1*1024向量
        vector_under_test = img2vector('test_digits/%s' % file_name_str)
        # 获得识别结果
        classifier_result = classify0(vector_under_test, training_mat, hand_write_labels, 3)
        print("分类返回结果为%d\t真实结果为%d" % (classifier_result, class_number))
        if classifier_result != class_number:
            error_count += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (error_count, error_count / test_file_count))


if __name__ == '__main__':
    start_time = time.time()
    hand_writing_class_test()
    end_time = time.time()
    print("运行时间: %.2f秒" % (end_time - start_time))
