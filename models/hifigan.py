""" 定义 HiFiGAN 声码器；"""

import torch
import torch.nn as nn


class HiFiMultiResBlock(nn.Module):
    """ HiFiGAN 声码器 生成器 的多级残差模块； """
    def __init__(self,
                 upsample_rate: int = 5,
                 in_channels: int = 128,
                 out_channels: int = 64,
                 block_kernel_size: list = [3, 7, 11],
                 block_dilation_rate: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 act: str = "LeakyReLU",
                 ):
        super().__init__()

        if act == "LeakyReLU":
            self.input_act = nn.LeakyReLU(0.2)
        else:
            self.input_act = nn.ReLU()

        self.upsample = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_rate*2,
            stride=upsample_rate,
            padding=upsample_rate,
            # device="cuda",  # 这里好像必须得声明是 cuda，要不会默认成 cpu
        )

    def forward(self, x):
        """
        :param x: [batch, in_channel, time]
        :return: [batch, out_channel, time]
        """
        x = self.input_act(x)
        x = self.upsample(x)

        return x


class HiFiGAN(nn.Module):
    """ HiFiGAN 声码器 """
    def __init__(self, conf: dict):
        super().__init__()

        self.upsample_rate = [5, 5, 4, 2]  # 乘起来等于 hop_size, 200 for 16K Hz
        self.out_channels = [64, 32, 16, 8]
        self.block_kernel_size = [3, 7, 11]
        self.block_dilation_rate = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.input_out_channel = self.out_channels[0] * 2
        self.input_kernel_size = 7
        self.conv1d_input = nn.Conv1d(
            in_channels=80,
            out_channels=self.input_out_channel,
            kernel_size=self.input_kernel_size,
            padding=self.input_kernel_size // 2,
            padding_mode="reflect",
        )

        self.multi_res_block = []
        in_channel_i = self.input_out_channel
        for upsample_rate_i, out_channel_i in zip(self.upsample_rate, self.out_channels):
            self.multi_res_block.append(
                HiFiMultiResBlock(
                    upsample_rate=upsample_rate_i,
                    in_channels=in_channel_i,
                    out_channels=out_channel_i,
                    block_kernel_size=self.block_kernel_size,
                    block_dilation_rate=self.block_dilation_rate,
                    act="LeakyReLU",
                )
            )
            in_channel_i = out_channel_i

    def forward(self, x):
        """
        :param x: [batch, channel=80, time]
        :return audio: [batch, channel=1, new_time]
        """
        x = self.conv1d_input(x)
        print(x.shape)

        # HiFiGAN Multi-Res-Block
        for block in self.multi_res_block:
            x = block(x)
            print(x.shape)

        return x
