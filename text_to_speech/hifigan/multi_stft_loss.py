""" HiFiGAN 用到的 多尺度STFT loss """

import torch
import torch.nn as nn


class MultiSTFTLoss:

    def __init__(self,
                 win_lengths: list = [240, 600, 1200],
                 hop_sizes: list = [50, 120, 240],
                 fft_sizes: list = [512, 1024, 2048],
                 ):
        """ HiFiGAN 用到的 多尺度STFT loss """

        assert len(win_lengths) == len(hop_sizes) == len(fft_sizes) == 3

        self.stft_0 = STFTLoss(win_length=win_lengths[0], hop_size=hop_sizes[0], fft_size=fft_sizes[0])
        self.stft_1 = STFTLoss(win_length=win_lengths[1], hop_size=hop_sizes[1], fft_size=fft_sizes[1])
        self.stft_2 = STFTLoss(win_length=win_lengths[2], hop_size=hop_sizes[2], fft_size=fft_sizes[2])

    def cal_loss(self, audio_gt: torch.tensor, audio_gen: torch.tensor):
        """
        计算多尺度STFT loss；
        :param audio_gt: [batch, time]
        :param audio_gen: [batch, time]
        :return: sc_loss, mag_loss
        """
        sc_loss_0, mag_loss_0 = self.stft_0.cal_loss(audio_gt, audio_gen)
        sc_loss_1, mag_loss_1 = self.stft_1.cal_loss(audio_gt, audio_gen)
        sc_loss_2, mag_loss_2 = self.stft_2.cal_loss(audio_gt, audio_gen)

        sc_loss = (sc_loss_0 + sc_loss_1 + sc_loss_2) / 3.0
        mag_loss = (mag_loss_0 + mag_loss_1 + mag_loss_2) / 3.0

        return sc_loss, mag_loss


class STFTLoss:
    def __init__(self,
                 win_length: int = 240,
                 hop_size: int = 50,
                 fft_size: int = 512,
                 ):
        """ HiFiGAN 用到的 单个STFT loss """

        self.win_length = win_length
        self.hop_size = hop_size
        self.fft_size = fft_size

    def cal_feature(self, audio: torch.tensor) -> torch.tensor:
        """ 输入音频，提取特征； """

        stft = torch.stft(
            audio,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            return_complex=True,
        )

        feature = torch.abs(stft)
        feature = torch.clip(feature, min=1e-7, max=1e3)
        return feature

    def cal_loss(self, audio_gt: torch.tensor, audio_gen: torch.tensor):
        """
        计算STFT loss；
        :param audio_gt: [batch, time]
        :param audio_gen: [batch, time]
        :return: sc_loss, mag_loss
        """

        if len(audio_gt.shape) == 3:
            audio_gt = audio_gt.clone().squeeze(1)  # [batch, 1, time] -> [batch, time]
        if len(audio_gen.shape) == 3:
            audio_gen = audio_gen.clone().squeeze(1)  # [batch, 1, time] -> [batch, time]

        # 裁剪到相同长度
        if audio_gt.shape[-1] > audio_gen.shape[-1]:
            audio_gt = audio_gt[:, :audio_gen.shape[-1]]
        elif audio_gt.shape[-1] < audio_gen.shape[-1]:
            audio_gen = audio_gen[:, :audio_gt.shape[-1]]

        feature_gt = self.cal_feature(audio_gt)
        feature_gen = self.cal_feature(audio_gen)

        mag_loss = nn.L1Loss()(
            torch.log(feature_gt), torch.log(feature_gen)
        )

        sc_loss = torch.norm(feature_gt - feature_gen, dim=[1, 2]) / torch.norm(feature_gt, dim=[1, 2])
        sc_loss = torch.mean(sc_loss)

        return sc_loss, mag_loss
