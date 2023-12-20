""" 一些预处理的工具； """

"""语音数据预处理"""
from .preprocess_main import main_make_label    # 读取一个文件夹内，每个说话人的所有数据，重命名、响度归一化、提取单声道、重采样至 22.05K、制作 label 文件；
from .preprocess_main import check_punctuation  # 读取 label 文件，统计标点符号有哪些；
from .preprocess_main import main_jieba_cut     # 读取 label 文件，进行 jieba 分词，生成包含 音素的 label 文件；

"""处理音频"""
from .audio_process import file2dur          # 读取音频、返回时长；
from .audio_process import file2sample_rate  # 读取音频、返回采样率；
from .audio_process import loudness_norm     # 响度归一化；
from .audio_process import channels_to_mono  # 双声道转单声道；
from .audio_process import resample          # 重采样；
from .audio_process import copy_and_rename   # 复制、转换成 wav 格式；

"""处理字典"""
from .dict_process import dic_sort  # 对字典排序、展示
from .dict_process import dict2numpy, numpy2dict  # 字典和np.array的转换

"""处理字符串"""
from .str2bool import str2bool  # 字符串 转 布尔值
