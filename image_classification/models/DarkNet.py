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
        self.mid_channels = [32, 64, 128, 256, 512, 1024]

        self.begin_block = self.get_begin_block()  # ResNet 模型的第一个模块

        self.down_sample_block_1 = self.get_down_sample_block(in_channel=self.mid_channels[0], out_channel=self.mid_channels[1])
        self.residual_block_1 = self.get_residual_block(in_out_channel=self.mid_channels[1], num_blocks=1)

        self.down_sample_block_2 = self.get_down_sample_block(in_channel=self.mid_channels[1], out_channel=self.mid_channels[2])
        self.residual_block_2 = self.get_residual_block(in_out_channel=self.mid_channels[2], num_blocks=2)

        self.down_sample_block_3 = self.get_down_sample_block(in_channel=self.mid_channels[2], out_channel=self.mid_channels[3])
        self.residual_block_3 = self.get_residual_block(in_out_channel=self.mid_channels[3], num_blocks=8)

        self.down_sample_block_4 = self.get_down_sample_block(in_channel=self.mid_channels[3], out_channel=self.mid_channels[4])
        self.residual_block_4 = self.get_residual_block(in_out_channel=self.mid_channels[4], num_blocks=8)

        self.down_sample_block_5 = self.get_down_sample_block(in_channel=self.mid_channels[4], out_channel=self.mid_channels[5])
        self.residual_block_5 = self.get_residual_block(in_out_channel=self.mid_channels[5], num_blocks=4)

        self.final_block = self.get_final_block()  # DarkNet 模型的最后一个模块

    def forward(self, x, model_type: str = "classification"):
        assert model_type in ["classification", "yolo"], f"DarkNet53 只能用于分类任务和YOLOv3任务；"

        yolo_outputs = []

        # 第零个卷积：channel: 3 -> 32，图像尺寸 416*416
        x = self.begin_block(x)

        # 第一个下采样 + 残差模块，图像尺寸下降一半，即 228*228，channel -> 64
        x = self.down_sample_block_1(x)
        x = self.residual_block_1(x)

        # 第二个下采样 + 残差模块，图像尺寸下降至 1/4，即 124*124，channel -> 128
        x = self.down_sample_block_2(x)
        x = self.residual_block_2(x)

        # 第三个下采样 + 残差模块，图像尺寸下降至 1/8，即 52*52，channel -> 256
        x = self.down_sample_block_3(x)
        x = self.residual_block_3(x)
        yolo_outputs.append(x.clone())

        # 第四个下采样 + 残差模块，图像尺寸下降至 1/16，即 26*26，channel -> 512
        x = self.down_sample_block_4(x)
        x = self.residual_block_4(x)
        yolo_outputs.append(x.clone())

        # 第五个下采样 + 残差模块，图像尺寸下降至 1/32，即 13*13，channel -> 1024
        x = self.down_sample_block_5(x)
        x = self.residual_block_5(x)
        yolo_outputs.append(x.clone())

        # 分类任务的输出模块，YOLO不需要
        x = self.final_block(x)

        if model_type == "classification":
            return x
        else:
            return yolo_outputs

    @staticmethod
    def get_residual_block(
            in_out_channel,
            num_blocks=1,
    ):
        """ DarkNet 核心的残差模块；"""
        blocks = nn.Sequential()
        for block_i in range(num_blocks):
            blocks.append(
                ResidualBlock(
                    in_out_channel=in_out_channel,
                )
            )
        return blocks

    @staticmethod
    def get_down_sample_block(
            in_channel: int,
            out_channel: int,
    ):
        """ DarkNet 的下采样模块，使用3*3卷积、不使用pooling层；"""
        blocks = nn.Sequential()
        blocks.append(
            BasicBlock(
                in_channel=in_channel,
                out_channel=out_channel,
                kernel_size=3,
                stride=2,
            )
        )
        return blocks

    def get_begin_block(self):
        """ ResNet的第一个模块；"""
        layers = nn.Sequential()
        layers.append(
            BasicBlock(
                in_channel=self.in_channel,
                out_channel=self.mid_channels[0],
                kernel_size=3,
                stride=1,
            )
        )
        return layers

    def get_final_block(self):
        """ ResNet的最后一个；"""
        layers = nn.Sequential()
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # 无论输入尺寸是多少，输出尺寸都是[1, 1]
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=self.mid_channels[-1], out_features=self.n_classes))
        return layers


if __name__ == "__main__":
    conf = {
        "n_classes": 160,
        "input_shape": [224, 224],
    }

    model = DarkNet53(conf)

    print(model)