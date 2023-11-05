""" 定义 FastSpeech2 声学模型；"""

import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    """TTS transformer Block."""
    def __init__(self, conf: dict):
        super().__init__()

        self.start_feed_forward = FeedForwardModule()
        self.multi_head_self_attention = MultiHeadSelfAttention()
        self.convolution_module = ConvolutionModule()
        self.end_feed_forward = FeedForwardModule()

    def forward(self, x, mask):

        x = self.start_feed_forward(x)
        x = self.multi_head_self_attention(x, mask)
        x = self.convolution_module(x)
        x = self.end_feed_forward(x)

        x *= mask

        return x


class MultiHeadSelfAttention(nn.Module):
    """" Conformer 模型的 自注意力模块；包含残差结构； """
    def __init__(self, in_out_channel=256, head_nums=2, dropout_rate=0.2):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_out_channel)

        self.head_nums = head_nums
        self.channel = in_out_channel
        self.linear_q = nn.Linear(in_features=in_out_channel, out_features=in_out_channel)
        self.linear_k = nn.Linear(in_features=in_out_channel, out_features=in_out_channel)
        self.linear_v = nn.Linear(in_features=in_out_channel, out_features=in_out_channel)
        self.dropout = nn.Dropout(dropout_rate)

    def calculate_qkv(self, x):
        """ [batch, channels, time] -> [batch, head_nums * head_size, time] -> [batch, head_nums, time, head_size] """
        batch_size = x.shape[0]
        time = x.shape[2]

        queue = self.linear_q(x.transpose(1, 2)).transpose(1, 2)  # nn.Linear 要求 channel last
        key = self.linear_k(x.transpose(1, 2)).transpose(1, 2)
        value = self.linear_v(x.transpose(1, 2)).transpose(1, 2)

        queue = queue.view([batch_size, self.head_nums, x, -1])
        key = key.view([batch_size, self.head_nums, x, -1])
        value = value.view([batch_size, self.head_nums, x, -1])

        return queue, key, value

    def forward(self, x, mask):
        """
        :param x: [batch, in_out_channel, time]
        :return: [batch, in_out_channel, time]
        """
        residual = nn.Identity()(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  # layer norm 需要 channel last
        # self-attention
        queue, key, value = self.calculate_qkv(x)
        position_embedding = self.calculate_postion_embdding(x)  # TODO: 生成相对位置编码的emb，并进行卷积
        attention_score = self.calculate_maxtrix(x)  # TODO: 计算相对位置编码所需的 matrix
        x = self.calculate_masked_attention_score(x)  # TODO: 计算attention score

        x = self.dropout(x)
        x += residual
        return x


class ConvolutionModule(nn.Module):
    """ Conformer 模型的 卷积模块；包含残差结构；"""
    def __init__(self, in_out_channel=256, deep_wise_conv_kernel_size=7, dropout_rate=0.2):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_out_channel)
        self.point_wise_conv1d_1 = nn.Conv1d(
            in_channels=in_out_channel,
            out_channels=in_out_channel * 2,
            kernel_size=1,
            stride=1,
        )
        self.glu = nn.GLU(dim=1)  # 门控激活函数
        self.deep_wise_conv1d = nn.Conv1d(
            in_channels=in_out_channel,
            out_channels=in_out_channel,
            kernel_size=deep_wise_conv_kernel_size,
            stride=1,
            padding=deep_wise_conv_kernel_size // 2,
            groups=1,
        )
        self.batch_norm = nn.BatchNorm1d(in_out_channel)
        self.swish = nn.SiLU()  # swish 激活函数
        self.point_wise_conv1d_2 = nn.Conv1d(
            in_channels=in_out_channel,
            out_channels=in_out_channel,
            kernel_size=1,
            stride=1,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        :param x: [batch, in_out_channel, time]
        :return: [batch, in_out_channel, time]
        """
        residual = nn.Identity()(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  # layer norm 需要 channel last
        # point-wise-conv1d + GLU + deep-wise-conv1d + BN + swish + point-wise-conv1d + dropout
        x = self.point_wise_conv1d_1(x)
        x = self.glu(x)
        x = self.deep_wise_conv1d(x)
        x = self.point_wise_conv1d_2(self.swish(self.batch_norm(x)))
        x = self.dropout(x)
        x += residual
        return x


class FeedForwardModule(nn.Module):
    """ Conformer 模型的 feed-forward 模块，包含残差结构；"""
    def __init__(self, in_out_channel=256, mid_channel=1024, kernel_size=3, dropout_rate=0.2):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_out_channel)
        self.conv1d_1 = nn.Conv1d(
            in_channels=in_out_channel,
            out_channels=mid_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=mid_channel,
            out_channels=in_out_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()  # 按照PaddleSpeech、ESPnet，这里是ReLU，而按照conformer论文，这里是swish；
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        :param x: [batch, in_out_channel, time]
        :return: [batch, in_out_channel, time]
        """
        residual = nn.Identity()(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  # layer norm 需要 channel last
        # 3*3-conv1d-增加channel + relu + dropout + 3*3-conv1d-减小channel + dropout
        x = self.dropout(self.conv1d_2(self.dropout(self.relu(self.conv1d_1(x)))))
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
