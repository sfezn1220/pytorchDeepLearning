""" 定义 HiFiGAN 声码器；"""

import torch
import torch.nn as nn

from text_to_speech.hifigan.generator import HiFiGANGenerator
from text_to_speech.hifigan.discriminator import HiFiGANMultiPeriodDiscriminator, HiFiGANMultiScaleDiscriminator


class HiFiGAN(nn.Module):
    """ HiFiGAN 声码器，包含：生成器 + 判别器； """
    def __init__(self, conf: dict, device="cuda"):
        super().__init__()

        assert device in ["cpu", "cuda"]

        self.generator = HiFiGANGenerator(conf, device=device)
        self.multi_scale_discriminator = HiFiGANMultiScaleDiscriminator(conf, device=device)
        self.multi_period_discriminator = HiFiGANMultiPeriodDiscriminator(conf, device=device)

    def forward(self, mel):
        """
        :param mel: [batch, channel=80, time]
        :return tuple(audio, new_audio_list): [batch, channel=1, time] and [[batch, channel=1, new_time1], ...]
        """
        # 生成器
        audio_gen = self.generator(mel)

        # 判别器
        discriminator_outputs = self.forward_discriminator(audio_gen)

        return audio_gen, discriminator_outputs

    def forward_discriminator(self, audio):
        """
        :param audio: [batch, channel=1, time]
        :return new_audio_list: [[batch, channel=1, new_time1], ...]
        """
        # 判别器
        if len(audio.shape) == 2:
            audio = audio.clone().unsqueeze(1)  # [batch, time] -> [batch, channel=1, time]

        discriminator_outputs = []
        discriminator_outputs.append(self.multi_scale_discriminator(audio))
        discriminator_outputs.append(self.multi_period_discriminator(audio))

        return discriminator_outputs

    def inference(self, mel):
        """
        :param mel: [batch, channel=80, time]
        :return audio: [batch, channel=1, time]
        """
        # 生成器
        audio_gen = self.generator(mel)
        return audio_gen
