""" 定义 HiFiGAN 声码器；"""

import torch
import torch.nn as nn


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


class HiFiGANPeriodDiscriminator(nn.Module):
    """ HiFiGAN 声码器的：多周期判别器的单个输出； """
    def __init__(self,
                 period: int,
                 num_blocks: int = 5,
                 max_channel: int = 512,
                 kernel_size: int = 5,
                 device="cuda"
                 ):
        super().__init__()

        assert device in ["cpu", "cuda"]

        self.period = period  # 2, 3, 5, 7 or 11
        print(f"period = {period}")

        # multi-conv2D
        self.backbone = nn.Sequential()
        in_channel = 1
        for i in range(num_blocks):
            out_channel = min(8 * (4 ** (i + 1)), max_channel)
            self.backbone.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=(kernel_size, 1),
                    stride=(3, 1),
                    padding=(kernel_size // 2, 0),
                )
            )
            self.backbone.append(
                nn.LeakyReLU(0.2)
            )
            in_channel = out_channel

        # final conv2D
        self.final_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=1,
            kernel_size=(3, 1),
            stride=(3, 1),
            padding=(3 // 2, 0),
        )

    def forward(self, audio):
        """
        :param audio: [batch, channel=1, time]
        :return new_audio: [batch, channel=1, new_time]
        """
        batch, channel, time = audio.shape

        # padding 到 period 的整数倍
        if time % self.period != 0:
            zeros_pad = torch.zeros([batch, channel, self.period - time % self.period],
                                    dtype=audio.dtype, device=audio.device)
            audio = torch.concat([audio.clone(), zeros_pad], dim=-1)

        # reshape: -> [batch, channel=1, -1, period]
        audio = torch.reshape(audio.clone(), [batch, channel, -1, self.period])

        # multi-conv2D
        audio = self.backbone(audio)

        # final conv2D
        audio = self.final_conv(audio)

        # reshape: -> [batch, channel=1, new_time]
        audio = torch.reshape(audio.clone(), [batch, channel, -1])

        return audio


class HiFiGANMultiPeriodDiscriminator(nn.Module):
    """ HiFiGAN 声码器的：多周期判别器； """
    def __init__(self, conf: dict, device="cuda"):
        super().__init__()

        assert device in ["cpu", "cuda"]

        self.period_discriminator_2 = HiFiGANPeriodDiscriminator(period=2, device=device)
        self.period_discriminator_3 = HiFiGANPeriodDiscriminator(period=3, device=device)
        self.period_discriminator_5 = HiFiGANPeriodDiscriminator(period=5, device=device)
        self.period_discriminator_7 = HiFiGANPeriodDiscriminator(period=7, device=device)
        self.period_discriminator_11 = HiFiGANPeriodDiscriminator(period=11, device=device)

    def forward(self, audio):
        """
        :param audio: [batch, channel=1, time]
        :return new_audio_list: [[batch, channel=1, new_time1], ...]
        """
        new_audio_2 = self.period_discriminator_2(audio)
        new_audio_3 = self.period_discriminator_3(audio)
        new_audio_5 = self.period_discriminator_5(audio)
        new_audio_7 = self.period_discriminator_7(audio)
        new_audio_11 = self.period_discriminator_11(audio)
        return [
            new_audio_2,
            new_audio_3,
            new_audio_5,
            new_audio_7,
            new_audio_11,
        ]


class HiFiGANMultiScaleDiscriminator(nn.Module):
    """ HiFiGAN 声码器的：多尺度判别器； """
    def __init__(self,
                 conf: dict,
                 kernel_sizes: list = [5, 3],
                 down_sample_scales: list = [4, 4, 4, 4],
                 init_channel: int = 16,
                 max_channel: int = 512,
                 device="cuda"
                 ):
        super().__init__()

        assert device in ["cpu", "cuda"]

        self.blocks = []

        self.blocks.append(
            nn.Conv1d(
                in_channels=1,
                out_channels=init_channel,
                kernel_size=kernel_sizes[0] * kernel_sizes[1],
                stride=1,
                padding=(kernel_sizes[0] * kernel_sizes[1]) // 2,
                padding_mode="reflect",
            )
        )

        self.blocks.append(
            nn.LeakyReLU(0.2)
        )

        in_channel = init_channel
        for i, down_sample_scale_i in enumerate(down_sample_scales):
            out_channel = max(in_channel * down_sample_scale_i, max_channel)

            self.blocks.append(
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=down_sample_scale_i * 10 + 1,
                    stride=down_sample_scale_i,
                    padding=(down_sample_scale_i * 10 + 1) // 2,
                    groups=4,
                )
            )

            in_channel = out_channel

            self.blocks.append(
                nn.LeakyReLU(0.2)
            )

        # final_conv1D_1
        out_channel = max(in_channel * 2, max_channel)
        self.blocks.append(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_sizes[0],
                stride=1,
                padding=kernel_sizes[0] // 2,
            )
        )
        self.blocks.append(
            nn.LeakyReLU(0.2)
        )

        # final_conv1D_2
        self.blocks.append(
            nn.Conv1d(
                in_channels=out_channel,
                out_channels=1,
                kernel_size=kernel_sizes[1],
                stride=1,
                padding=kernel_sizes[1] // 2,
            )
        )

    def forward(self, audio):
        """
        :param audio: [batch, channel=1, time]
        :return new_audio_list: [[batch, channel=1, new_time1], ...]
        """
        outputs = []
        for block in self.blocks:
            new_audio = block(audio)
            outputs.append(new_audio)  # 将每一层的结果都收集起来、都输出；
        return outputs


class HiFiGAN(nn.Module):
    """ HiFiGAN 声码器，包含：生成器 + 判别器； """
    def __init__(self, conf: dict, device="cuda"):
        super().__init__()

        assert device in ["cpu", "cuda"]

        self.generator = HiFiGANGenerator(conf, device=device)
        self.multi_period_discriminator = HiFiGANMultiPeriodDiscriminator(conf, device=device)
        self.multi_scale_discriminator = HiFiGANMultiScaleDiscriminator(conf, device=device)

    def forward(self, x):
        """
        :param x: [batch, channel=80, time]
        :return tuple(audio, new_audio_list): [batch, channel=1, time] and [[batch, channel=1, new_time1], ...]
        """
        # 生成器
        audio = self.generator(x)

        # 判别器
        discriminator_outputs = self.forward_discriminator(audio)

        return audio, discriminator_outputs

    def forward_discriminator(self, audio):
        """
        :param audio: [batch, channel=1, time]
        :return new_audio_list: [[batch, channel=1, new_time1], ...]
        """
        # 判别器
        discriminator_outputs = []
        discriminator_outputs.extend(self.multi_period_discriminator(audio))
        discriminator_outputs.extend(self.multi_scale_discriminator(x))

        return discriminator_outputs
