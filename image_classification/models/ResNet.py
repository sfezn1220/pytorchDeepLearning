""" 定义 ResNet 网络结构；"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """单个基础的 Residual 模块，包含2个 3*3 conv + BN + ReLU 组合，以及残差连接； """
    def __init__(self, in_channel, out_channel, strides=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.strides = strides

        self.block = self.get_residual_block()
        self.residual_connect = self.get_residual_connect()
        self.final_act = nn.ReLU()

    def forward(self, x):
        residual = self.residual_connect(x)
        x = self.block(x)
        x = x + residual
        return self.final_act(x)

    def get_residual_block(self):
        """ 单个 residual 模块的主分支；"""
        layers = nn.Sequential()
        # conv + BN + ReLU
        layers.append(nn.Conv2d(self.in_channel, out_channels=self.out_channel, kernel_size=3,
                                stride=self.strides, padding=1))
        layers.append(nn.BatchNorm2d(self.out_channel))
        layers.append(nn.ReLU())
        # conv + BN
        layers.append(nn.Conv2d(self.out_channel, out_channels=self.out_channel, kernel_size=3,
                                stride=1, padding=1))
        layers.append(nn.BatchNorm2d(self.out_channel))
        return layers

    def get_residual_connect(self):
        """ 单个 residual 模块的残差连接分支；"""
        if self.in_channel != self.out_channel or self.strides != 1:
            return nn.Conv2d(self.in_channel, out_channels=self.out_channel, kernel_size=1,
                             stride=self.strides)
        else:
            return nn.Identity()


class BottleneckResidualBlock(ResidualBlock):
    """使用2个 1*1卷积代替 3*3卷积的 Residual 模块，包含2个 conv + BN + ReLU 组合，以及残差连接； """
    def __init__(self, in_channel, bottleneck_channel, out_channel, strides=1):
        self.bottleneck_channel = bottleneck_channel  # 中间结果的通道数
        super().__init__(in_channel, out_channel, strides)

    def get_residual_block(self):
        """ 单个 residual 模块的主分支；"""
        layers = nn.Sequential()
        # 1*1 conv + BN + ReLU
        layers.append(nn.Conv2d(self.in_channel, out_channels=self.bottleneck_channel, kernel_size=1,
                                stride=1))
        layers.append(nn.BatchNorm2d(self.bottleneck_channel))
        layers.append(nn.ReLU())
        # 3*3 conv + BN + ReLU
        layers.append(nn.Conv2d(self.bottleneck_channel, out_channels=self.bottleneck_channel, kernel_size=3,
                                stride=self.strides, padding=1))
        layers.append(nn.BatchNorm2d(self.bottleneck_channel))
        layers.append(nn.ReLU())
        # 1*1 conv + BN
        layers.append(nn.Conv2d(self.bottleneck_channel, out_channels=self.out_channel, kernel_size=1,
                                stride=1))
        layers.append(nn.BatchNorm2d(self.out_channel))
        return layers


class ResNet18(nn.Module):
    def __init__(self, conf: dict):
        """ 初始化 ResNet 网络结构；"""
        super().__init__()

        self.n_classes = conf["n_classes"]  # 分类的类别数量
        self.in_channel = 3  # 输入图像的通道处，默认为RGB三通道
        self.input_shape = conf["input_shape"]  # 默认 [224, 224]

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


class ResNet152(ResNet18):
    def __init__(self, conf: dict):
        """ 初始化 ResNet 网络结构；"""
        super().__init__(conf)

        self.block_1 = self.get_bottleneck_residual_block(64, 64, 256, num_blocks=3, down_sample=False)
        self.block_2 = self.get_bottleneck_residual_block(256, 128, 512, num_blocks=8, down_sample=True)
        self.block_3 = self.get_bottleneck_residual_block(512, 256, 1024, num_blocks=36, down_sample=True)
        self.block_4 = self.get_bottleneck_residual_block(1024, 512, 2048, num_blocks=3, down_sample=True)

        self.final_block = self.get_final_block(in_features=2048)

    def get_bottleneck_residual_block(
            self,
            in_channel,
            bottleneck_channel,
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
                BottleneckResidualBlock(
                    in_channel=in_channel,
                    bottleneck_channel=bottleneck_channel,
                    out_channel=out_channel,
                    strides=strides
                )
            )
        return blocks


if __name__ == "__main__":
    conf = {
        "n_classes": 170,
        "input_shape": [224, 224],
    }

    model = ResNet18(conf)

    print(model)