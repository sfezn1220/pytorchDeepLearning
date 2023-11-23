""" 定义 HiFiGAN 声码器的判别器部分；"""

import torch
import torch.nn as nn


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
                    device=device,
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
            device=device,
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
                device=device,
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
                    device=device,
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
                device=device,
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
                device=device,
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
