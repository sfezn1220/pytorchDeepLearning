""" 定义 FastSpeech2 声学模型；"""

import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    """TTS transformer Block."""
    def __init__(self, conf: dict):
        super().__init__()

        self.start_feed_forward = FeedForwardModule()
        self.convolution_module = ConvolutionModule()

    def forward(self, x, mask):

        x = self.start_feed_forward(x)

        x *= mask

        return x


class ConvolutionModule(nn.Module):
    """ Conformer 摸瞎的 卷积模块；包含残差结构；"""
    def __init__(self, in_channel=256, out_channel=256):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_channel)
        self.point_wise_conv1d = nn.Conv1d(
            in_channels=in_channel,
            out_channels=in_channel * 2,
            kernel_size=1,
            stride=1,
        )
        self.act_1 = nn.GLU()
        self.deep_wise_conv1d = # TODO 继续搭建 conformer Conv Module

    def forward(self, x):
        """
        :param x: [batch, time, in_channel]
        :return: [batch, time, out_channel]
        """
        residual = nn.Identity()(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  # layer norm 需要 channel last
        x = x  # TODO
        x = residual + 0.5 * x
        return x


class FeedForwardModule(nn.Module):
    """ Conformer 模型的 feed-forward 模块，包含残差结构；"""
    def __init__(self, in_channel=256, out_channel=256, mid_channel=1024, kernel_size=3, dropout_rate=0.2):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_channel)
        self.conv1d_1 = nn.Conv1d(
            in_channels=in_channel,
            out_channels=mid_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=mid_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        :param x: [batch, time, in_channel]
        :return: [batch, time, out_channel]
        """
        residual = nn.Identity()(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  # layer norm 需要 channel last
        x = self.dropout(self.conv1d_2(self.dropout(self.act(self.conv1d_1(x)))))
        x = residual + 0.5 * x
        return x


class ConformerEncoder(nn.Module):
    """TTS conformer Encoder，包含多层 conformer 结构；"""
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
        :param x: [batch, in_channel, time]
        :param mask: 与 x 尺寸一致
        :return: [batch, out_channel, time]
        """
        # 检查尺寸
        assert len(x.shape) == len(mask.shape) == 3, \
            f"输入音素序列的尺寸应该是三维 [batch, in_channel, time]，而 len(inputs.shape) = {len(x.shape)}"

        for block in self.blocks:
            x = block(x, mask)
            print(f"x.shape = {x.shape}, mask.shape = {mask.shape}")

        return x


class FastSpeech2(nn.Module):
    """ FastSpeech2 声学模型；"""
    def __init__(self, conf: dict):
        super().__init__()

        self.conf = conf

        self.phoneme_embedding = self.get_phoneme_embedding()
        self.encoder = ConformerEncoder(conf)

    def forward(self, phoneme_ids, spk_id, duration):
        """
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :param spk_id: [batch] 输入的音色ID
        :param duration: [batch, time] 输入的每个音素对应的帧数；
        :return:
        """

        phoneme_mask = self.get_phoneme_mask(phoneme_ids).transpose(1, 2)
        phoneme_embedded = self.phoneme_embedding(phoneme_ids).transpose(1, 2)  # [batch, channel, time], channel first

        encoder_outputs = self.encoder(phoneme_embedded, phoneme_mask)

        return encoder_outputs

    @staticmethod
    def get_phoneme_mask(phoneme_ids):
        """
        输入音素序列的mask，用于后续的 attention；
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :return: [batch, time] 与输入的尺寸相同，当音素=0时取0，当音素!=0时取1；
        """
        mask = torch.not_equal(phoneme_ids, 0)
        return mask.unsqueeze(-1).int()

    def get_phoneme_embedding(self):
        """ 对音素序列进行 embedding，在 encoder 之前；"""
        num_embeddings = self.conf.get('phonemes_size', 213)  # 音素的数量
        embedding_dim = 256  # 默认取256，和后面encoder、decoder的channel一致
        return nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0,  # 音素“0”表示pad
        )
