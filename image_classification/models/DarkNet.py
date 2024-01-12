""" 定义 YOLO v3 模型的核心 DarkNet 网络结构；"""

import torch.nn as nn


class BasicBlock(nn.Module):
    """ DarkNet 基础的模块：conv + BN + LeakyReLU """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv = nn.Conv2d(self.in_channel, out_channels=self.out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(self.out_channel)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """单个基础的 Residual 模块，包含2组 conv + BN + LeakyReLU 组合，以及残差连接； """
    def __init__(self, in_out_channel):
        super().__init__()
        self.in_out_channel = in_out_channel  # 输入、输出 通道数
        self.mid_channel = in_out_channel // 2  # 中间结果的通道数

        self.block = self.get_residual_block()
        self.final_act = nn.ReLU()

    def forward(self, x):
        residual = nn.Identity()(x)
        x = self.block(x)
        return x + residual

    def get_residual_block(self):
        """ 单个 residual 模块的主分支；"""
        layers = nn.Sequential()
        # 1*1 conv + BN + ReLU
        layers.append(
            BasicBlock(self.in_out_channel, self.mid_channel, kernel_size=1, stride=1)
        )
        # 3*3 conv + BN + ReLU
        layers.append(
            BasicBlock(self.mid_channel, self.in_out_channel, kernel_size=3, stride=1)
        )
        return layers


class DarkNet53(nn.Module):
    def __init__(self, conf: dict):
        """ 初始化 DarkNet53 网络结构；"""
        super().__init__()

        self.n_classes = conf["n_classes"]  # 分类的类别数量
        self.in_channel = 3  # 输入图像的通道处，默认为RGB三通道
        self.input_shape = conf["input_shape"]  # 默认 [416, 416]

        # TODO
        self.begin_block = self.get_begin_block()  # ResNet 模型的第一个模块
        self.block_1 = self.get_residual_block(64, 64, num_blocks=2, down_sample=False)
        self.block_2 = self.get_residual_block(64, 128, num_blocks=2, down_sample=True)
        self.block_3 = self.get_residual_block(128, 256, num_blocks=2, down_sample=True)
        self.block_4 = self.get_residual_block(256, 512, num_blocks=2, down_sample=True)
        self.final_block = self.get_final_block(in_features=512)  # ResNet 模型的最后一个模块

    def forward(self, x):
        x = self.begin_block(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.final_block(x)
        return x

    def get_residual_block(
            self,
            in_channel,
            out_channel,
            num_blocks=1,
            down_sample=True,
    ):
        """ ResNet的核心，包含多个残差模块；"""
        blocks = nn.Sequential()
        for block_i in range(num_blocks):
            if block_i == 0 and down_sample is True:
                strides = 2
            else:
                strides = 1
            if block_i != 0:
                in_channel = out_channel
            blocks.append(
                ResidualBlock(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    strides=strides
                )
            )
        return blocks

    def get_begin_block(self):
        """ ResNet的第一个模块；"""
        layers = nn.Sequential()
        layers.append(nn.Conv2d(self.in_channel, out_channels=64, kernel_size=7, stride=2, padding=3))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return layers

    def get_final_block(self, in_features=512):
        """ ResNet的最后一个；"""
        layers = nn.Sequential()
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # 无论输入尺寸是多少，输出尺寸都是[1, 1]
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=in_features, out_features=self.n_classes))
        return layers


if __name__ == "__main__":
    conf = {
        "n_classes": 170,
        "input_shape": [224, 224],
    }

    model = ResNet18(conf)

    print(model)