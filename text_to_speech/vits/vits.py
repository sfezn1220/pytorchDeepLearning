""" 定义 VITS 模型；"""

import random

import torch
import torch.nn as nn

from text_to_speech.utils.get_mask import get_attention_mask


class VITS(nn.Module):
    """ VITS 端到端模型；"""
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

    def forward(self, phoneme_ids, spk_id, audio_gt):
        """
        for training
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :param spk_id: [batch] 输入的音色ID
        :param audio_gt: [batch, time] 输出的真实音频；
        :return:
        """

        phoneme_mask = get_attention_mask(phoneme_ids).unsqueeze(-1).transpose(1, 2)  # [batch, 1, time]
        phoneme_embedded = self.get_phoneme_embedding(phoneme_ids).transpose(1, 2)  # [batch, channel, time]

        x, m_p, logs_p, x_mask = self.text_encoder(phoneme_embedded, phoneme_mask)  # TODO

        return

    def infer(self, phoneme_ids, spk_id):
        """
        for inference
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :param spk_id: [batch] 输入的音色ID
        :return:
        """

        return
