""" 定义 YOLO v3 模型的核心 DarkNet 网络结构；"""

import torch.nn as nn

from image_classification.models import DarkNet53, CBLBlock, ResBlock


class YOLOv3(nn.Module):
    def __init__(self, conf: dict):
        """ 初始化 YOLO v3 网络结构；"""
        super().__init__()

        self.n_classes = conf["n_classes"]  # 分类的类别数量
        self.in_channel = 3  # 输入图像的通道处，默认为RGB三通道
        self.input_shape = conf["input_shape"]  # 默认 [416, 416]

        # 核心的 DarkNet53 结构
        self.backbone = DarkNet53(conf)

        # 输出部分的三个连续CBL块：
        self.CBL_blocks_0 = self.get_CBL_blocks(num_blocks=5, in_channel=256)
        self.CBL_blocks_1 = self.get_CBL_blocks(num_blocks=5, in_channel=256)
        self.CBL_blocks_2 = self.get_CBL_blocks(num_blocks=5, in_channel=1024, out_channel=256)

    def get_CBL_blocks(self, num_blocks=5, in_channel=1024, out_channel=1024):
        """ YOLO3 输出部分的 连续卷积模块；默认为 5个 Conv + BN + Leaky ReLU 结构； """
        blocks = nn.Sequential()
        for block_i in range(num_blocks):
            blocks.append(
                ResBlock(
                    in_channel=in_channel,
                    out_channel=out_channel,
                )
            )
            in_channel = out_channel
        return blocks

    def forward(self, x):

        backbone_outputs = self.backbone(x, "yolo")
        backbone_outputs_0 = backbone_outputs[0]  # [batch, channel=256, 52*52]
        backbone_outputs_1 = backbone_outputs[1]  # [batch, channel=512, 26*26]
        backbone_outputs_2 = backbone_outputs[2]  # [batch, channel=1024, 13*13]

        return x
