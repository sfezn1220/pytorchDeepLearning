"""关于字典处理的一些工具脚本；"""

import numpy as np


def dict2numpy(dic):
    """ 将字典型变量，转换为 np.array """
    dic_list = []
    for item in dic:
        dic_list.append([item, dic[item]])
    res_arr = np.array(dic_list, dtype=object)
    return res_arr


def numpy2dict(arr: np.array):
    """ 将字典型变量转换成的 np.array，再转换回字典 """
    res_dic = {}
    for pair in arr:
        item = pair[0]
        value = pair[1]
        res_dic[item] = value
    return res_dic


def dic_sort(dic, reverse=True):
    """
        将一个字典排序，并展示出来；
        排序规则：
            lambda item: dic[item]
    """

    # dic -> list
    list_1 = []
    for item in dic:
        list_1.append([item, dic[item]])

    # sort
    list_1.sort(key=lambda x: x[-1], reverse=reverse)

    # print
    print(f"\nPrint Sorted Dic:\n")
    for item, value in list_1:
        print(f"item: {item}   value: {value}")

    print(f"")

    return
