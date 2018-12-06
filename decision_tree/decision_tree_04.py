#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 下午5:32
# @Author  : liumingyu
# @Site    : 
# @File    : decision_tree_04.py

import operator
from math import log

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

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


"""
函数说明:统计集合中出现次数最多的元素(类标签)

Parameters:
    class_list - 类标签列表
Returns:
    sorted_class_count[0][0] - 出现次数最多的元素(类标签)
"""


def majority_cnt(class_list):
    # 定义统计每个元素出现的次数
    class_count = {}
    # 遍历集合
    for vote in class_list:
        # 如果key没出现过则初始化为0
        if vote not in class_count.keys():
            class_count[vote] = 0
        # 统计
        class_count[vote] += 1
    # 按字典的值降序排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的元素
    return sorted_class_count[0][0]


"""
函数说明:创建决策树

Parameters:
    data_set - 训练数据集
    labels - 分类属性标签
    feat_labels - 存储选择的最优特征标签
Returns:
    my_tree - 决策树
"""


def create_tree(data_set, labels, feat_labels):
    # 取分类标签(yes or no)
    class_list = [example[-1] for example in data_set]
    # 如果类别完全相同则停止继续划分直接返回
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有的特征时,返回出现次数最多的类标签
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 选择最优特征
    best_feat = choose_best_feature(data_set)
    # 最优特征标签
    best_feat_label = labels[best_feat]
    # 将最优特征标签放到一起
    feat_labels.append(best_feat_label)
    # 根据最优特征标签生成树
    result_tree = {best_feat_label: {}}
    # 删除已经使用的特征标签
    del (labels[best_feat])
    # 获得训练集中所有最优特征的属性值
    feat_values = [example[best_feat] for example in data_set]
    # 去重
    unique_values = set(feat_values)
    # 遍历特征,创建决策树
    for value in unique_values:
        result_tree[best_feat_label][value] \
            = create_tree(split_data_set(data_set, best_feat, value), labels, feat_labels)
    return result_tree


"""
函数说明:获取决策树叶子节点的数目

Parameters:
    my_tree - 决策树
Returns:
    num_leafs - 决策树的叶子节点的数目
"""


def get_num_leafs(my_tree):
    # 初始化叶子节点数目
    num_leafs = 0
    # py3中my_tree.keys()返回的是dict_keys,不再是list,所以不能使用my_tree.keys()[0]的方法获取节点属性,可以使用list(my_tree.keys())[0]
    first_str = next(iter(my_tree))
    # 获取下一组字典
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        # 判断该节点是否是字典,如果不是字典,代表此节点为叶子节点
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


"""
函数说明:获取决策树的层数

Parameters:
    my_tree - 决策树
Returns:
    max_depth - 决策树的层数
"""


def get_tree_depth(my_tree):
    # 初始化决策树层数
    max_depth = 0
    first_str = next(iter(my_tree))
    # 获取下一个字典
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        # 判断该节点是否是字典,如果不是字典,代表此节点为叶子节点
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        # 如果当前层数大于最大层数,则更新
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


"""
函数说明:绘制节点

Parameters:
    node_txt - 节点名称
    center_pt - 文本位置
    parent_pt - 标注的箭头位置
    node_type - 节点格式
Returns:
    无
"""


def plot_node(node_txt, center_pt, parent_pt, node_type):
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    # 设置中文字体
    font = FontProperties(fname=r"/Library/Fonts/Songti.ttc", size=14)
    # 绘制结点
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args, FontProperties=font)


"""
函数说明:标注有向边属性值

Parameters:
    cntr_pt - 用于计算标注位置
    parent_pt - 用于计算标注位置
    txt_string - 标注的内容
Returns:
    无
"""


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    # 计算标注位置
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


"""
函数说明:绘制决策树

Parameters:
    my_tree - 决策树(字典)
    parent_pt - 标注的内容
    node_txt - 节点名称
Returns:
    无
"""


def plot_tree(my_tree, parent_pt, node_txt):
    # 设置节点格式
    decision_node = dict(boxstyle="sawtooth", fc="0.8")
    # 设置叶节点格式
    leaf_node = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶节点数目,决定了树的宽度
    num_leafs = get_num_leafs(my_tree)
    # 获取决策树层数
    depth = get_tree_depth(my_tree)
    # 下个字典
    first_str = next(iter(my_tree))
    # 中心位置
    cntr_pt = (plot_tree.x0ff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.y0ff)
    # 标注有向边属性值
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    # 绘制节点
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    # 下一个字典,就是继续绘制子节点
    second_dict = my_tree[first_str]
    # y偏移
    plot_tree.y0ff = plot_tree.y0ff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # 如果不是叶节点,递归调用继续绘制
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            # 如果是叶节点,绘制叶节点,并标注有向边属性值
            plot_tree.x0ff = plot_tree.x0ff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.x0ff, plot_tree.y0ff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x0ff, plot_tree.y0ff), cntr_pt, str(key))
    plot_tree.y0ff = plot_tree.y0ff + 1.0 / plot_tree.totalD


"""
函数说明:创建绘制面板

Parameters:
    in_tree - 决策树(字典)
Returns:
    无
"""


def create_plot(in_tree):
    # 创建fig
    fig = plt.figure(1, facecolor='white')
    # 清空fig
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    create_plot.ax1 = plt.subplot(111, frameon=False, **ax_props)
    # 获取决策树叶节点数目
    plot_tree.totalW = float(get_num_leafs(in_tree))
    # 获取决策树层数
    plot_tree.totalD = float(get_tree_depth(in_tree))
    # x偏移
    plot_tree.x0ff = -0.5 / plot_tree.totalW;
    plot_tree.y0ff = 1.0;
    # 绘制决策树
    plot_tree(in_tree, (0.5, 1.0), '')
    # 显示绘制结果
    plt.show()


if __name__ == '__main__':
    data_set, labels = create_data_set()
    feat_labels = []
    my_tree = create_tree(data_set, labels, feat_labels)
    print(my_tree)
    create_plot(my_tree)
