""" 一些预处理的工具； """


from .read_file import read_json_lists    # 读取json格式的label文件，并读取为dict格式；

from .read_textgrid import read_all_textgrid    # 读取这个文件夹内的所有textgrid文件；返回字典：key=uttid, value=[[phoneme1, dur1], ...]；
