#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 下午5:43
# @Author  : liumingyu
# @Site    : 
# @File    : decision_tree_01.py

from math import log

"""
函数说明:创建测试数据集

Parameters:
    无
Returns:
    data_set - 数据集
    labels - 分类属性
"""


def create_data_set():
    # 数据集
    data_set = [[0, 0, 0, 0, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    # 分类
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return data_set, labels


"""
函数说明:计算给定数据集的经验熵

Parameters:
    data_set - 数据集
Returns:
    shannon_ent - 经验熵
"""


def cal_shannon_ent(data_set):
    # 返回数据集的行数
    num_entries = len(data_set)
    # 定义每个分类出现的次数
    label_counts = {}
    # 对每组特征向量进行统计
    for feat_vec in data_set:
        # 提取分类信息
        current_label = feat_vec[-1]
        # 如果当前分类没有统计过,则加入字典中
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        # 计数
        label_counts[current_label] += 1
    # 经验熵(香农熵)
    shannon_ent = 0.0
    # 遍历分类计数字典
    for key in label_counts:
        # 计算该分类的概率
        prob = float(label_counts[key]) / num_entries
        # 利用公式计算
        shannon_ent -= prob * log(prob, 2)
    # 返回经验熵
    return shannon_ent


if __name__ == '__main__':
    data_set, labels = create_data_set()
    print(data_set)
    print(cal_shannon_ent(data_set))
