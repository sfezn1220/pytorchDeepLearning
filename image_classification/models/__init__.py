""" 一些深度学习模型的定义； """

from .ResNet import ResNet18, ResNet152

from .vgg import VGG16

from .DarkNet import DarkNet53
from .DarkNet import CBLBlock, ResBlock  # Darknet、YOLOv3 的基本模块
