""" 字符串 转换成 布尔值 """


def str2bool(value: str) -> bool:
    """ 字符串 转换成 布尔值 """
    value = value.lower()

    if value in ["true"]:
        return True
    elif value in ["false"]:
        return False
    else:
        raise ValueError(f"str2bool Error: \"{value}\"")
