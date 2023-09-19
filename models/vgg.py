""" 定义 VGG 网络结构；"""

import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, n_classes, num_blocks, in_channels):
        """ 初始化 VGG 网络结构；"""
        super().__init__()

        self.n_classes = n_classes  # 分类的类别数量
        self.num_blocks = num_blocks  # 每个VGG模块由几个卷积层构成
        self.in_channels = in_channels  # VGG模型的输入尺寸

        self.VGGArchitecture = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))

        self.blocks = []  # TODO

    @staticmethod
    def vgg_block(num_blocks, in_channels, out_channels):
        """ 单个VGG模块；"""
        layers = nn.Sequential()

        for _ in range(num_blocks):

            layers.add(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )

            layers.add(
                nn.ReLU()
            )

            in_channels = out_channels

        layers.add(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
        )

        return layers

