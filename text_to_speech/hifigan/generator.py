""" 定义 HiFiGAN 声码器的生成器部分；"""

import torch
import torch.nn as nn


class HiFiGANGenerator(nn.Module):
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
            device=device,
        )

        # up sample blocks
        self.up_sample_block_1 = UpSampleBlock(
            up_sample_rate=self.up_sample_rate[0],
            in_channels=self.input_out_channel,
            out_channels=self.out_channels[0],
            block_kernel_size=self.block_kernel_size,
            block_dilation_rate=self.block_dilation_rate,
            device=device
        )
        self.up_sample_block_2 = UpSampleBlock(
            up_sample_rate=self.up_sample_rate[1],
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[1],
            block_kernel_size=self.block_kernel_size,
            block_dilation_rate=self.block_dilation_rate,
            device=device
        )
        self.up_sample_block_3 = UpSampleBlock(
            up_sample_rate=self.up_sample_rate[2],
            in_channels=self.out_channels[1],
            out_channels=self.out_channels[2],
            block_kernel_size=self.block_kernel_size,
            block_dilation_rate=self.block_dilation_rate,
            device=device
        )
        self.up_sample_block_4 = UpSampleBlock(
            up_sample_rate=self.up_sample_rate[3],
            in_channels=self.out_channels[2],
            out_channels=self.out_channels[3],
            block_kernel_size=self.block_kernel_size,
            block_dilation_rate=self.block_dilation_rate,
            device=device
        )

        # final conv
        self.non_linear_act = nn.LeakyReLU(0.2)
        self.conv1d_final = nn.Conv1d(
            in_channels=self.out_channels[-1],
            out_channels=1,  # audio
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="reflect",
            device=device,
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
        # for block in self.up_sample_block:  # 通道数逐渐降低、逐渐从一个点上采样至hop_size
        #     x = block(x)  # [batch, channel = self.out_channels[-1], time * hop_size]
        x = self.up_sample_block_1(x)
        x = self.up_sample_block_2(x)
        x = self.up_sample_block_3(x)
        x = self.up_sample_block_4(x)

        # final conv
        x = self.non_linear_act(x)
        x = self.conv1d_final(x)  # [batch, 1, time * hop_size]
        x = self.act_final(x)

        return x


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

        self.backbone = nn.Sequential()

        self.backbone.append(
            nn.LeakyReLU(0.2)
        )

        self.backbone.append(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=up_sample_rate * 2,
                stride=up_sample_rate,
                padding=up_sample_rate,
                device=device,  # 这里好像必须得声明是 cuda，要不会默认成 cpu、导致和输入tensor不一致
            )
        )

        self.backbone.append(
            MultiResBlock(
                in_out_channel=out_channels,
                block_kernel_size=block_kernel_size,
                block_dilation_rate=block_dilation_rate,
                device=device,
            )
        )

    def forward(self, x):
        """
        :param x: [batch, in_channel, time]
        :return: [batch, out_channel, time]
        """
        x = self.backbone(x)
        return x


class MultiResBlock(nn.Module):
    """ HiFiGAN 声码器 生成器 的多个残差模块； """
    def __init__(self,
                 in_out_channel: int,
                 block_kernel_size: list = [3, 7, 11],
                 block_dilation_rate: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 device: str = "cuda",
                 ):
        super().__init__()

        assert len(block_kernel_size) == len(block_dilation_rate)

        self.block_1 = ResBlock(
            in_out_channel=in_out_channel,
            kernel_size=block_kernel_size[0],
            dilation_rates=block_dilation_rate[0],
            device=device,
        )
        self.block_2 = ResBlock(
            in_out_channel=in_out_channel,
            kernel_size=block_kernel_size[1],
            dilation_rates=block_dilation_rate[1],
            device=device,
        )
        self.block_3 = ResBlock(
            in_out_channel=in_out_channel,
            kernel_size=block_kernel_size[2],
            dilation_rates=block_dilation_rate[2],
            device=device,
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        output_1 = self.block_1(x)
        output_2 = self.block_2(x)
        output_3 = self.block_3(x)
        return (output_1 + output_2 + output_3) / 3


class ResBlock(nn.Module):
    """ HiFiGAN 声码器 生成器 的单个残差模块； """
    def __init__(self,
                 in_out_channel: int,
                 kernel_size: int,
                 dilation_rates: list[int] = [1, 3, 5],
                 device: str = "cuda",
                 ):
        super().__init__()

        self.input_act = nn.LeakyReLU(0.2)

        self.blocks = []
        for i, dilation_rate in enumerate(dilation_rates):
            block_i = nn.Sequential()

            block_i.append(
                nn.LeakyReLU(0.2)
            )
            block_i.append(
                nn.Conv1d(
                    in_channels=in_out_channel,
                    out_channels=in_out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2 * dilation_rate,
                    padding_mode="reflect",
                    dilation=dilation_rate,
                    device=device,
                )
            )
            block_i.append(
                nn.Conv1d(
                    in_channels=in_out_channel,
                    out_channels=in_out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    padding_mode="reflect",
                    dilation=1,
                    device=device,
                )
            )

            self.blocks.append(block_i)

    def forward(self, x):  # TODO
        """
        :param x:
        :return:
        """
        x = self.input_act(x)
        for block in self.blocks:
            xo = block(x)
            x = x.clone() + xo

        return x
