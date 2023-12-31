""" FastSpeech2 声学模型的 Conformer 编解码器； """

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from text_to_speech.utils.layer_norm import LayerNorm


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
        :param mask: [batch, 1, time]
        :return: [batch, out_channel, time]
        """
        # 检查尺寸
        assert len(x.shape) == len(mask.shape) == 3, \
            f"输入音素序列的尺寸应该是三维 [batch, in_channel, time]，而 len(inputs.shape) = {len(x.shape)}"

        for block in self.blocks:
            x = block(x, mask)

        return x


class ConformerDecoder(ConformerEncoder):
    """TTS conformer Decoder，相当于 一个线性层 + Encoder；"""
    def __init__(self, conf: dict):
        super().__init__(conf)

        self.channel = conf.get('encoder_channels', 256)  # encoder、decoder 等的 channel 数，默认是256；
        self.linear_input = nn.Linear(
            in_features=self.channel,
            out_features=self.channel,
        )

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """
        :param x: [batch, in_channel, time]
        :param mask: [batch, 1, time]
        :return: [batch, out_channel, time]
        """
        x = self.linear_input(x.transpose(1, 2)).transpose(1, 2)  # [batch, channel, time], nn.Linear 需要 channel last
        x = torch.mul(x, mask)
        return super().forward(x, mask)


class ConformerBlock(nn.Module):
    """TTS transformer Block."""
    def __init__(self, conf: dict):
        super().__init__()

        self.start_feed_forward = FeedForwardModule()
        self.multi_head_self_attention = MultiHeadSelfAttention()
        self.convolution_module = ConvolutionModule()
        self.end_feed_forward = FeedForwardModule()

    def forward(self, x, mask):
        """
        :param x: [batch, in_channel, time]
        :param mask: [batch, 1, time]
        :return: [batch, out_channel, time]
        """

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

        self.layer_norm = LayerNorm(in_out_channel, dim=1)

        self.channel = in_out_channel
        self.head_nums = head_nums
        self.head_size = in_out_channel // head_nums
        self.linear_q = nn.Linear(in_features=in_out_channel, out_features=self.head_size * head_nums)
        self.linear_k = nn.Linear(in_features=in_out_channel, out_features=self.head_size * head_nums)
        self.linear_v = nn.Linear(in_features=in_out_channel, out_features=in_out_channel)

        self.linear_pos_emb = nn.Linear(in_features=in_out_channel, out_features=self.head_size * head_nums)

        self.bias_u = nn.Parameter(torch.Tensor(1, head_nums, 1, self.head_size))
        self.bias_v = nn.Parameter(torch.Tensor(1, head_nums, 1, self.head_size))

        self.dropout = nn.Dropout(dropout_rate)

        self.linear_output = nn.Linear(in_features=in_out_channel, out_features=in_out_channel)

    def forward(self, x, mask):
        """
        :param x: [batch, in_out_channel, time]
        :param mask: [batch, 1, time]
        :return: [batch, in_out_channel, time]
        """
        residual = nn.Identity()(x)
        x = self.layer_norm(x)
        # self-attention
        with autocast(enabled=False):
            query, key, value = self.calculate_qkv(x)
            position_embedding = self.get_position_embedding(x)
            attention_score = self.calculate_matrix(position_embedding, query, key)
            x = self.calculate_masked_attention_score(attention_score, mask, value)
        x = self.dropout(x)
        x = x.clone() + residual
        return x

    @staticmethod
    def get_position_embedding(x):
        """
        生成 相对位置编码所需的 position embedding；
        :param x: [batch, channel, time]
        :return: [batch, channel, time]
        """
        batch, channel, time = x.shape

        # 先生成一个 [time, channel] 的系数矩阵：
        factor = torch.arange(0, channel, 2, dtype=x.dtype).unsqueeze(1)  # [channel // 2, 1]
        factor = torch.pow(torch.tensor(10000.0), -1 * factor) * channel
        position = torch.arange(0, time, dtype=x.dtype).unsqueeze(0)  # [1, time]
        position = position * factor  # [channel // 2, time]

        # 将position分成两部分，分别初始化：
        position_positive = torch.zeros(channel, time, dtype=x.dtype)  # [channel, time]
        position_negative = torch.zeros(channel, time, dtype=x.dtype)  # [channel, time]

        # 计算position的两个部分：
        position_positive[0::2, :] = torch.sin(position)
        position_positive[1::2, :] = torch.cos(position)
        position_negative[0::2, :] = torch.sin(-1 * position)
        position_negative[1::2, :] = torch.cos(-1 * position)

        # 拼接出position embedding
        position_positive = torch.flip(position_positive, dims=[1])  # [channel, time]
        position_negative = position_negative[:, 1:]  # [channel, times-1]
        position_embedding = torch.cat([position_positive, position_negative], dim=1)  # [channel, time * 2 - 1]
        position_embedding = position_embedding.unsqueeze(0)  # [1, channel, time * 2 - 1]

        return position_embedding.to(x.device)

    def calculate_qkv(self, x):
        """ [batch, channels, time] -> [batch, head_nums * head_size, time] -> [batch, head_nums, time, head_size] """
        batch_size = x.shape[0]
        time = x.shape[2]

        query = self.linear_q(x.transpose(1, 2)).transpose(1, 2)  # nn.Linear 要求 channel last
        key = self.linear_k(x.transpose(1, 2)).transpose(1, 2)
        value = self.linear_v(x.transpose(1, 2)).transpose(1, 2)

        query = torch.reshape(query, [batch_size, self.head_nums, time, -1])
        key = torch.reshape(key, [batch_size, self.head_nums, time, -1])
        value = torch.reshape(value, [batch_size, self.head_nums, time, -1])

        return query, key, value  # [batch, head_nums, time, head_size]

    @staticmethod
    def relative_shift(matrix):
        """
        计算相对位置编码；
        :param matrix: [batch, head_nums, time, time * 2 - 1]
        :return: [batch, head_nums, time, time]
        """
        batch, heads, time, _ = matrix.shape

        zeros_pad = torch.zeros([batch, heads, time, 1], dtype=matrix.dtype, device=matrix.device)
        matrix_padded = torch.cat([zeros_pad, matrix], dim=-1)
        matrix_padded = torch.reshape(matrix_padded, [batch, heads, -1, time])

        matrix = matrix_padded[:, :, 1:, :]
        matrix = torch.reshape(matrix, [batch, heads, time, -1])

        return matrix[:, :, :, :time]

    def calculate_matrix(self, position_embedding, query, key):
        """
        计算：基于相对位置编码的 attention 打分结果；
        :param position_embedding: [1, channel, time * 2 - 1]
        :param query: [batch, head_nums, time, head_size]
        :param key: [batch, head_nums, time, head_size]
        :return: attention_score: [batch, head_nums, time, time]
        """
        position_embedding = self.linear_pos_emb(position_embedding.transpose(1, 2)).transpose(1, 2)  # [1, head_nums * head_size, time*2-1]
        position_embedding = torch.reshape(position_embedding, [1, self.head_nums, -1, self.head_size])  # [1, head_nums, time*2-1, head_size]

        query_with_u = query + self.bias_u  # [batch, head_nums, time, head_size]
        query_with_v = query + self.bias_v  # [batch, head_nums, time, head_size]

        query_with_u = query_with_u.to(torch.float32)  # fp16 -> fp32
        query_with_v = query_with_v.to(torch.float32)
        key = key.to(torch.float32)
        position_embedding = position_embedding.to(torch.float32)

        matrix_ac = torch.matmul(query_with_u, key.transpose(2, 3))  # [batch, head_nums, time, time]
        matrix_bd = torch.matmul(query_with_v, position_embedding.transpose(2, 3))  # [batch, head_nums, time, time*2-1]
        matrix_bd = self.relative_shift(matrix_bd)  # [batch, head_nums, time, time]

        return matrix_bd + matrix_ac

    def calculate_masked_attention_score(self, attention_score, mask, value):
        """
        对输入的 attention_score 进行 mask；
        :param attention_score: [batch, head_nums, time, time]
        :param mask: [batch, 1, time]
        :param value: [batch, head_nums, time, head_size]
        :return: [batch, channel, time]
        """
        batch, _, time, _ = attention_score.shape

        attention_score /= torch.sqrt(torch.tensor(self.head_size, dtype=torch.int32, device=attention_score.device))  # 除以根号下文本长度

        mask = mask.unsqueeze(dim=-1)  # [batch, 1, time, 1]
        mask = mask.eq(0)

        dtype = str(attention_score.dtype).split(".")[-1]
        min_value = float(np.finfo(dtype).min)
        attention_score = attention_score.masked_fill(mask, min_value)  # 将mask=0位置的分数，设置为很小的值

        attention_probs = torch.softmax(attention_score, dim=-1)
        attention_probs = attention_probs.masked_fill(mask, 0.0)  # 将mask=0位置的概率，设置为零
        attention_probs = self.dropout(attention_probs)

        attention_probs = attention_probs.to(torch.float32)
        value = value.to(torch.float32)

        outputs = torch.matmul(attention_probs, value)  # [batch, head_nums, time, head_size]
        outputs = outputs.transpose(1, 2)
        outputs = torch.reshape(outputs, [batch, self.channel, time])
        outputs = self.linear_output(outputs.transpose(1, 2)).transpose(2, 1)  # nn.linear 要求 channel last

        return outputs  # [batch, channel, time]


class ConvolutionModule(nn.Module):
    """ Conformer 模型的 卷积模块；包含残差结构；"""
    def __init__(self, in_out_channel=256, deep_wise_conv_kernel_size=7, dropout_rate=0.2):
        super().__init__()

        self.layer_norm = LayerNorm(in_out_channel, dim=1)
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
        x = self.layer_norm(x)  # layer norm 需要 channel last
        # point-wise-conv1d + GLU + deep-wise-conv1d + BN + swish + point-wise-conv1d + dropout
        x = self.point_wise_conv1d_1(x)
        x = self.glu(x)
        x = self.deep_wise_conv1d(x)
        x = self.point_wise_conv1d_2(self.swish(self.batch_norm(x)))
        x = self.dropout(x)
        x = x.clone() + residual
        return x


class FeedForwardModule(nn.Module):
    """ Conformer 模型的 feed-forward 模块，包含残差结构；"""
    def __init__(self, in_out_channel=256, mid_channel=1024, kernel_size=3, dropout_rate=0.2):
        super().__init__()

        self.layer_norm = LayerNorm(in_out_channel, dim=1)
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
        x = self.layer_norm(x)  # layer norm 需要 channel last
        # 3*3-conv1d-增加channel + relu + dropout + 3*3-conv1d-减小channel + dropout
        x = self.dropout(self.conv1d_2(self.dropout(self.relu(self.conv1d_1(x)))))
        x = residual + 0.5 * x.clone()
        return x
