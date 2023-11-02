""" 定义 FastSpeech2 声学模型；"""

import torch
import torch.nn as nn


class FastSpeech2(nn.Module):
    def __init__(self, conf: dict):
        """ 初始化 FastSpeech2 声学模型的网络结构；"""
        super().__init__()

    def forward(self, phoneme_ids, spk_id, duration):
        """
        :param phoneme_ids: [batch_size, text_padding_length] 输入的音素序列；
        :param spk_id: [batch_size] 输入的音色ID
        :param duration: [batch_size, text_padding_length] 输入的每个音素对应的帧数；
        :return:
        """

        phoneme_mask = self.get_phoneme_mask(phoneme_ids)

        return phoneme_mask

    @staticmethod
    def get_phoneme_mask(phoneme_ids):
        """
        输入音素序列的mask，用于后续的 attention；
        :param phoneme_ids: [batch_size, text_padding_length] 输入的音素序列；
        :return: [batch_size, text_padding_length] 与输入的尺寸相同，当音素=0时取0，当音素!=0时取1；
        """
        mask = torch.not_equal(phoneme_ids, 0)
        return mask.int()
