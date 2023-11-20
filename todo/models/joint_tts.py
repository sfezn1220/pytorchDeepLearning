""" 级联： 声学模型 + 声码器 """

import torch
import torch.nn as nn

from .fastspeech2 import FastSpeech2
from .hifigan import HiFiGAN


class JointTTS(nn.Module):
    """ 级联： 声学模型 + 声码器 """
    def __init__(self, conf: dict, device="cuda"):
        super().__init__()

        assert device in ["cpu", "cuda"]

        self.acoustic_model = FastSpeech2(conf)
        self.vocoder = HiFiGAN(conf, device=device)

        print('Parameters Count of acoustic_model = ',
              sum(p.numel() for p in self.acoustic_model.parameters() if p.requires_grad))
        print('Parameters Count of acoustic_model = ',
              sum(p.numel() for p in self.vocoder.parameters() if p.requires_grad))

    def forward(self,
                phoneme_ids,
                spk_id,
                audio_gt=None,
                duration_gt=None,
                f0_gt=None,
                energy_gt=None,
                mel_length=None,
                f0_length=None,
                energy_length=None
                ):
        """
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :param spk_id: [batch] 输入的音色ID
        :param duration_gt: [batch, time] 输入的每个音素对应的帧数；
        :param audio_gt: [batch, 1, time] 输入的音频；
        :param f0_gt: [batch, time] 输入的每帧对应的F0值；
        :param energy_gt: [batch, time] 输入的每帧对应的F0值；
        :param mel_length: [batch, ] 输入的帧数；
        :param f0_length: [batch, ] 输入的F0长度；
        :param energy_length: [batch, ] 输入的energy长度；
        :return:
        """

        # 声学模型
        mel_after, mel_before, f0_predict, energy_predict, duration_predict = self.acoustic_model(phoneme_ids, spk_id, duration_gt, f0_gt, energy_gt, mel_length, f0_length, energy_length)

        # 声码器
        audio_pred, discriminator_outputs = self.vocoder(mel_after)

        # 声码器的判别器
        if audio_gt is not None:
            true_features = self.vocoder.forward_discriminator(audio_gt)
        else:
            true_features = None

        return audio_pred, mel_after, mel_before, f0_predict, energy_predict, duration_predict, discriminator_outputs, true_features
