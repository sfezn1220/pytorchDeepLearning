""" 定义 FastSpeech2 声学模型；"""

import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    """TTS transformer Block."""
    def __init__(self, conf: dict):
        super().__init__()

        self.start_feed_forward = FeedForwardModule(conf)

    def forward(self, x, mask):

        return x


class FeedForwardModule(nn.Module):
    """ Conformer 模型的 feed-forward 模块，包含残差结构；"""
    def __init__(self, conf: dict):
        super().__init__()

        self.cond_1 = nn.Conv1d()  # TODO 定义 macaron feed-forward 模块

    def forward(self, x):
        """
        :param x: [batch, time, in_channel]
        :return: [batch, time, out_channel]
        """
        return x


class ConformerEncoder(nn.Module):
    """TTS transformer Encoder."""
    def __init__(self, conf: dict):
        super().__init__()

        self.conf = conf
        self.num_blocks = conf.get("encoder_num_blocks", 4)  # 包含多少层 conformer
        self.blocks = self.get_blocks()

    def get_blocks(self):
        """定义 conformer encoder 的网络结构；"""
        blocks = nn.Sequential()
        for block_i in range(self.num_blocks):
            blocks.append(
                ConformerBlock(self.conf)
            )
        return blocks

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """
        :param x: [batch, time] or [batch, time, 1]
        :param mask: 与 x 尺寸一致
        :return: [batch, time, out_channel]
        """
        # 检查尺寸
        assert len(x.shape) == len(mask.shape)
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, 2)  # [batch, time] -> [batch, time, 1]
            mask = torch.unsqueeze(mask, 1)  # [batch, time] > [batch, 1, time]

        for block in self.blocks:
            x = block(x, mask)
            print(f"{x.shape}")

        return x


class FastSpeech2(nn.Module):
    """ FastSpeech2 声学模型；"""
    def __init__(self, conf: dict):
        super().__init__()

        self.encoder = ConformerEncoder(conf)

    def forward(self, phoneme_ids, spk_id, duration):
        """
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :param spk_id: [batch] 输入的音色ID
        :param duration: [batch, time] 输入的每个音素对应的帧数；
        :return:
        """

        phoneme_mask = self.get_phoneme_mask(phoneme_ids)

        encoder_outputs = self.encoder(phoneme_ids, phoneme_mask)

        return phoneme_mask

    @staticmethod
    def get_phoneme_mask(phoneme_ids):
        """
        输入音素序列的mask，用于后续的 attention；
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :return: [batch, time] 与输入的尺寸相同，当音素=0时取0，当音素!=0时取1；
        """
        mask = torch.not_equal(phoneme_ids, 0)
        return mask.int()
