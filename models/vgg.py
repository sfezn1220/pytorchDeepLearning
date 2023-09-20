""" 定义 VGG 网络结构；"""

import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, n_classes):
        """ 初始化 VGG 网络结构；"""
        super().__init__()

        self.n_classes = n_classes  # 分类的类别数量
        self.in_channels = 3  # 输入图像的通道处，默认为RGB三通道

        self.num_blocks_and_out_channels = [[2, 64], [2, 128], [3, 256], [3, 512], [3, 512]]
        self.final_out_channels = self.num_blocks_and_out_channels[-1][-1] * 7 * 7

        self.backbone = self.get_backbone()
        self.final_block = self.get_final_block()

    def forward(self, x):
        x = self.backbone(x)
        x = self.final_block(x)
        return x

    def get_backbone(self):
        """ 组合多个VGG模块；"""
        in_channels = self.in_channels
        blocks = nn.Sequential()
        for num_blocks, out_channels in self.num_blocks_and_out_channels:
            for layer in self.vgg_block(num_blocks, in_channels, out_channels):
                blocks.append(layer)
            in_channels = out_channels
        return blocks

    @staticmethod
    def vgg_block(num_blocks, in_channels, out_channels):
        """ 单个VGG模块；"""
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers

    def get_final_block(self):
        """ 最后的全连接层；"""
        layers = nn.Sequential()
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.final_out_channels, 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(4096, self.n_classes))
        return layers
