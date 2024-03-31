""" 图像处理相关的脚本 """

from .read_file import read_json_lists    # 读取json格式的label文件，并读取为dict格式；

from .avif2png import avif2png    # 1、将 avif 图片，替换为 png 图片； 2、将一个文件夹里的所有 avif 图片，替换为 png 图片；
