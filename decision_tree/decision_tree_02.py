#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 下午6:43
# @Author  : liumingyu
# @Site    : 
# @File    : decision_tree_02.py

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


"""
函数说明:按照给定特征划分数据集

Parameters:
    data_set - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    result - 划分后的数据集
"""


def split_data_set(data_set, axis, value):
    # 定义返回的数据集列表
    result = []
    # 遍历数据集
    for feat_vec in data_set:
        # 如果当前特征的值等于value
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            # 去掉axis特征
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            # 将符合条件的添加到结果集中
            result.append(reduced_feat_vec)
    return result


"""
函数说明:选择出最大增益的特征索引值

Parameters:
    data_set - 数据集
Returns:
    best_feature - 最大信息增益的特征索引值
"""


def choose_best_feature(data_set):
    # 特征数量
    num_features = len(data_set[0]) - 1
    # 计算数据集的经验熵
    base_entropy = cal_shannon_ent(data_set)
    # 信息增益
    best_info_gain = 0.0
    # 最优特征的索引值
    best_feature = -1
    # 遍历所有特征
    for i in range(num_features):
        # 获取数据集的第i个特征的所有值
        feat_list = [example[i] for example in data_set]
        # 去重
        unique_values = set(feat_list)
        # 经验条件熵
        new_entropy = 0.0
        # 遍历去重后的数据集
        for value in unique_values:
            # 划分后的子集
            sub_data_set = split_data_set(data_set, i, value)
            # 计算子集的概率
            prob = len(sub_data_set) / float(len(data_set))
            # 根据公式计算经验条件熵
            new_entropy += prob * cal_shannon_ent(sub_data_set)
        # 信息增益
        info_gain = base_entropy - new_entropy
        print("第%d个特征的增益为%.3f" % (i, info_gain))
        # 记录最大的信息增益和对应的特征索引值
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    # 返回最大信息增益的特征索引值
    return best_feature


if __name__ == '__main__':
    data_set, labels = create_data_set()
    print("最优特征索引值:" + str(choose_best_feature(data_set)))
