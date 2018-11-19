#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 下午5:34
# @Author  : liumingyu
# @Site    : 
# @File    : knn_04.py

import time
from os import listdir

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN

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
    # 构建kNN分类器
    neigh = kNN(n_neighbors=3, algorithm='auto')
    # 拟合模型,training_mat为训练矩阵,hand_write_labels为对应的分类标签
    neigh.fit(training_mat, hand_write_labels)
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
        classifier_result = neigh.predict(vector_under_test)
        print("分类返回结果为%d\t真实结果为%d" % (classifier_result, class_number))
        if classifier_result != class_number:
            error_count += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (error_count, error_count / test_file_count))


if __name__ == '__main__':
    start_time = time.time()
    hand_writing_class_test()
    end_time = time.time()
    print("运行时间: %.2f秒" % (end_time - start_time))
