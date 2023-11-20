""" 定义一些数据集；"""

from .image_dataset import get_image_dataloader  # 图像分类 训练集/验证集的 DataLoader

from .tts_dataset import get_tts_dataloader  # 语音合成 训练集/验证集的 DataLoader
