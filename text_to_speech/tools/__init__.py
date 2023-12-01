""" 一些常用的、处理数据的脚本； """

from .copy_label_data import copy_label_files  # 读取一个label文件，复制里面的数据到新的路径下；

from .split_train_test import split_train_test  # 读取一个label文件，划分训练集、测试集；需要手动指定测试集；
