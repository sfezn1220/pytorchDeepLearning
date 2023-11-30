"""关于字典处理的一些工具脚本；"""


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
