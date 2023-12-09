""" 定义 VITS 模型；"""

import random

import torch
import torch.nn as nn

from text_to_speech.fastspeech2.conformer import ConformerEncoder


class VITS(nn.Module):
    """ VITS 端到端模型；"""
    def __init__(self, conf: dict):
        super().__init__()

        self.conf = conf
        self.channel = conf.get('encoder_channels', 256)  # encoder、decoder 等的 channel 数，默认是256；

        # speaker embedding
        self.get_speaker_embedding = nn.Embedding(
            num_embeddings=conf.get('speaker_size', 64),  # 音色的数量
            embedding_dim=self.channel,  # 默认和 encoder 的 channel 一致，256；
        )

