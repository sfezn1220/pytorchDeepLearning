""" 定义 FastSpeech2 声学模型；"""

import random

import torch
import torch.nn as nn

from text_to_speech.fastspeech2.conformer import ConformerEncoder, ConformerDecoder
from text_to_speech.fastspeech2.length_regulator import LengthRegulator
from text_to_speech.fastspeech2.variant_predictor import VariantPredictor


class FastSpeech2(nn.Module):
    """ FastSpeech2 声学模型；"""
    def __init__(self, conf: dict):
        super().__init__()

        self.conf = conf
        self.channel = conf.get('encoder_channels', 256)  # encoder、decoder 等的 channel 数，默认是256；

        # phoneme embedding, speaker embedding
        self.get_phoneme_embedding = nn.Embedding(
            num_embeddings=conf.get('phonemes_size', 213),  # 音素的数量
            embedding_dim=self.channel,  # 默认和 encoder 的 channel 一致，256；
            padding_idx=0,  # 音素“0”表示pad
        )
        self.get_speaker_embedding = nn.Embedding(
            num_embeddings=conf.get('speaker_size', 64),  # 音色的数量
            embedding_dim=self.channel,  # 默认和 encoder 的 channel 一致，256；
        )
        self.linear_speaker_embedding = nn.Linear(
            in_features=self.channel + self.channel,  # encoder 的 channel + spk-emb 的 channel；它们可以不一样；
            out_features=self.channel,
        )

        # encoder
        self.encoder = ConformerEncoder(conf)

        # decoder
        self.decoder = ConformerDecoder(conf)

        # length regulator
        self.length_regulator = LengthRegulator()

        # variant predictor
        self.f0_predictor = VariantPredictor()
        self.energy_predictor = VariantPredictor()
        self.duration_predictor = VariantPredictor()

        # f0, energy: make embedding
        f0_conv_kernel_size = conf.get('f0_conv_kernel_size', 9)
        self.f0_conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.channel,
            kernel_size=f0_conv_kernel_size,
            stride=1,
            padding=f0_conv_kernel_size // 2,
        )
        self.f0_dropout = nn.Dropout(0.2)
        self.energy_conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.channel,
            kernel_size=f0_conv_kernel_size,
            stride=1,
            padding=f0_conv_kernel_size // 2,
        )
        self.energy_dropout = nn.Dropout(0.2)

        # mel_before
        self.linear_mel_before = nn.Linear(
            in_features=self.channel,
            out_features=80,  # 即Mel谱维度
        )

        # post net
        self.post_net = PostNet()

    def forward(self, phoneme_ids, spk_id, duration_gt=None, f0_gt=None, energy_gt=None,
                mel_length=None, f0_length=None, energy_length=None, complete_percent: float = 0.5):
        """
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :param spk_id: [batch] 输入的音色ID
        :param duration_gt: [batch, time] 输入的每个音素对应的帧数；
        :param f0_gt: [batch, time] 输入的每帧对应的F0值；
        :param energy_gt: [batch, time] 输入的每帧对应的F0值；
        :param mel_length: int, Mel谱长度，padding前，仅用于调试；
        :param f0_length: int, f0长度，padding前，仅用于调试；
        :param energy_length: int, energy长度，padding前，仅用于调试；
        :param complete_percent: float，训练steps数量占全部数量的比例，比例越高，生成的duration的权重就越大；
        :return:
        """

        phoneme_mask = self.get_phoneme_mask(phoneme_ids).transpose(1, 2)  # [batch, 1, time]
        phoneme_embedded = self.get_phoneme_embedding(phoneme_ids).transpose(1, 2)  # [batch, channel, time]

        # encoder
        encoder_outputs = self.encoder(phoneme_embedded, phoneme_mask)  # [batch, channel, time]

        # add spk-emb
        speaker_embedding = self.get_speaker_embedding(spk_id).unsqueeze(-1)  # [batch, channel, 1]
        encoder_outputs = self.add_speaker_embedding(encoder_outputs, speaker_embedding, phoneme_mask)

        # Length Regulation
        duration_predict = self.duration_predictor(encoder_outputs, phoneme_mask)  # [batch, 1, time]
        duration_predict = duration_predict.squeeze(1)  # [batch, time]
        if duration_gt is not None:
            r = random.uniform(0, complete_percent)
            # 预测的duration
            duration_predict_exp = nn.ReLU()(torch.exp(duration_predict) - 1)
            duration_predict_exp = duration_predict_exp
            # 真实的duration
            duration_gt = duration_gt
            # 加到一起
            duration = (1 - r) * duration_gt + r * duration_predict_exp
            duration = duration.int()
        else:  # train
            duration = nn.ReLU()(torch.exp(duration_predict) - 1)
            duration = duration.int()
        lr_outputs, lr_mask = self.length_regulator(encoder_outputs, duration)  # [batch, channel, new_time], [batch, 1, new_time]

        # f0, energy 的计算放在 LR 后面
        f0_predict = self.f0_predictor(lr_outputs, lr_mask)  # [batch, 1, new_time]
        energy_predict = self.energy_predictor(lr_outputs, lr_mask)  # [batch, 1, new_time]

        # f0 embedding, energy embedding
        f0_embedding, energy_embedding = self.get_f0_energy_embedding(f0_gt, energy_gt, f0_predict, energy_predict)
        lr_outputs = lr_outputs.clone() + f0_embedding + energy_embedding

        # decoder
        decoder_outputs = self.decoder(lr_outputs, lr_mask)

        # mel before
        mel_before = self.linear_mel_before(decoder_outputs.transpose(1, 2)).transpose(2, 1)  # [batch, mel=80, time]

        # mel after
        mel_after = self.post_net(mel_before, lr_mask)  # [batch, mel=80, time]

        return mel_after, mel_before, f0_predict, energy_predict, duration_predict

    def get_f0_energy_embedding(self, f0_gt, energy_gt, f0_predict, energy_predict):
        """
        使用 f0_gt, energy_gt or f0_predict, energy_predict 来计算出 embedding；
        :param f0_gt: [batch, time] or [batch, 1, time]
        :param energy_gt: [batch, time] or [batch, 1, time]
        :param f0_predict: [batch, 1, time]
        :param energy_predict: [batch, 1, time]
        :return: ([batch, 1, time], [batch, 1, time])
        """
        # f0 embedding
        if f0_gt is not None:  # inference
            if len(f0_gt.shape) == 2:
                f0_gt = f0_gt.unsqueeze(1)  # [batch, 1, time]
            f0 = f0_gt[:, :, :f0_predict.shape[-1]]
        else:  # train
            f0 = f0_predict
        f0_embedding = self.f0_dropout(self.f0_conv(f0))

        # energy embedding
        if energy_gt is not None:  # inference
            if len(energy_gt.shape) == 2:
                energy_gt = energy_gt.unsqueeze(1)  # [batch, 1, time]
            energy = energy_gt[:, :, :energy_predict.shape[-1]]
        else:  # train
            energy = energy_predict
        energy_embedding = self.energy_dropout(self.energy_conv(energy))

        return f0_embedding, energy_embedding

    def add_speaker_embedding(self, encoder_outputs, speaker_embedding, phoneme_mask):
        """
        将 spk-emb 加到 encoder 的输出上；
        :param encoder_outputs: [batch, channel, time]
        :param speaker_embedding: [batch, channel, 1]
        :param phoneme_mask: [batch, 1, time]
        :return: [batch, channel, time]
        """
        time = encoder_outputs.shape[2]
        speaker_embedding = torch.repeat_interleave(speaker_embedding, time, dim=-1)  # [batch, channel, time]
        outputs = torch.concat([encoder_outputs, speaker_embedding], dim=1)  # [batch, channel * 2, time]
        outputs = self.linear_speaker_embedding(outputs.transpose(1, 2)).transpose(2, 1)  # [batch, channel, time]
        return outputs * phoneme_mask

    @staticmethod
    def get_phoneme_mask(phoneme_ids):
        """
        输入音素序列的mask，用于后续的 attention；
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :return: [batch, time] 与输入的尺寸相同，当音素=0时取0，当音素!=0时取1；
        """
        mask = torch.not_equal(phoneme_ids, 0)
        return mask.unsqueeze(-1).int()


class PostNet(nn.Module):
    """ FastSpeech2 最后处理 Mel谱的模块，包含残差结构；"""
    def __init__(self, n_blocks=5, in_out_channel=80, mid_channel=512, kernel_size=5, dropout_rate=0.2):
        super().__init__()

        self.backbone = nn.Sequential()
        for i in range(n_blocks):
            in_channel = in_out_channel if i == 0 else mid_channel
            out_channel = in_out_channel if i == n_blocks-1 else mid_channel
            self.backbone.append(
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
            )
            self.backbone.append(nn.BatchNorm1d(out_channel))
            if i != n_blocks-1:
                self.backbone.append(nn.Tanh())
            self.backbone.append(nn.Dropout(dropout_rate))

    def forward(self, x, mask):
        """
        :param x: [batch, mel_channel=80, time]
        :param mask: [batch, 1, time]
        :return: [batch, mel_channel=80, time]
        """
        residual = nn.Identity()(x)
        for layer in self.backbone:
            x = layer(x)
        x = torch.mul(x, mask)
        return residual + x
