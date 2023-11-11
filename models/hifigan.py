""" 定义 HiFiGAN 声码器；"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """ HiFiGAN 声码器 生成器 的单个残差模块； """
    def __init__(self,
                 in_out_channel: int,
                 kernel_size: int,
                 dilation_rates: list[int]):
        super().__init__()

    def forward(self, x):  # TODO
        """
        :param x:
        :return:
        """
        return x


class MultiResBlock(nn.Module):
    """ HiFiGAN 声码器 生成器 的多个残差模块； """
    def __init__(self,
                 in_out_channel: int,
                 block_kernel_size: list = [3, 7, 11],
                 block_dilation_rate: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 ):
        super().__init__()

        assert len(block_kernel_size) == len(block_dilation_rate)

        self.blocks = []
        for kernel_size, dilation_rates in zip(block_kernel_size, block_dilation_rate):
            self.blocks.append(
                ResBlock(
                    in_out_channel=in_out_channel,
                    kernel_size=kernel_size,
                    dilation_rates=dilation_rates,
                )
            )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        res = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for block in self.blocks:
            res += block(x)

        return res / len(self.blocks)


class UpSampleBlock(nn.Module):
    """ HiFiGAN 声码器 生成器 的上采样模块； """
    def __init__(self,
                 up_sample_rate: int,
                 in_channels: int,
                 out_channels: int,
                 block_kernel_size: list = [3, 7, 11],
                 block_dilation_rate: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 device: str = "cuda",
                 ):
        super().__init__()

        self.input_act = nn.LeakyReLU(0.2)

        self.up_sample_layer = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=up_sample_rate * 2,
            stride=up_sample_rate,
            padding=up_sample_rate,
            device=device,  # 这里好像必须得声明是 cuda，要不会默认成 cpu、导致和输入tensor不一致
        )

        self.multi_res_block = MultiResBlock(
            in_out_channel=out_channels,
            block_kernel_size=block_kernel_size,
            block_dilation_rate=block_dilation_rate,
        )

    def forward(self, x):
        """
        :param x: [batch, in_channel, time]
        :return: [batch, out_channel, time]
        """
        x = self.input_act(x)
        x = self.up_sample_layer(x)
        x = self.multi_res_block(x)

        return x


class HiFiGAN(nn.Module):
    """ HiFiGAN 声码器 """
    def __init__(self, conf: dict, device="cuda"):
        super().__init__()

        assert device in ["cpu", "cuda"]

        self.up_sample_rate = [5, 5, 4, 2]  # 乘起来等于 hop_size, 200 for 16K Hz
        self.out_channels = [64, 32, 16, 8]  # 随着逐次上采样，通道数逐渐降低
        self.block_kernel_size = [3, 7, 11]
        self.block_dilation_rate = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.kernel_size = 7

        # first conv
        self.input_out_channel = self.out_channels[0] * 2
        self.conv1d_first = nn.Conv1d(
            in_channels=80,
            out_channels=self.input_out_channel,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="reflect",
        )

        # up sample blocks
        self.up_sample_block = []
        in_channel_i = self.input_out_channel
        for upsample_rate_i, out_channel_i in zip(self.up_sample_rate, self.out_channels):
            self.up_sample_block.append(
                UpSampleBlock(up_sample_rate=upsample_rate_i,
                              in_channels=in_channel_i,
                              out_channels=out_channel_i,
                              block_kernel_size=self.block_kernel_size,
                              block_dilation_rate=self.block_dilation_rate,
                              device=device)
            )
            in_channel_i = out_channel_i

        # final conv
        self.non_linear_act = nn.LeakyReLU(0.2)
        self.conv1d_final = nn.Conv1d(
            in_channels=self.out_channels[-1],
            out_channels=1,  # audio
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="reflect",
        )
        self.act_final = nn.Tanh()

    def forward(self, x):
        """
        :param x: [batch, channel=80, time]
        :return audio: [batch, channel=1, new_time]
        """
        # first conv
        x = self.conv1d_first(x)

        # HiFiGAN Multi-UpSample-Res-Block
        for block in self.up_sample_block:  # 通道数逐渐降低、逐渐从一个点上采样至hop_size
            x = block(x)  # [batch, channel = self.out_channels[-1], time * hop_size]

        # final conv
        x = self.non_linear_act(x)
        x = self.conv1d_final(x)  # [batch, 1, time * hop_size]
        x = self.act_final(x)

        return x
