""" 定义 FastSpeech2 声学模型；"""

import torch
import torch.nn as nn


class LayerNorm(torch.nn.LayerNorm):
    """ 对 nn.LayerNorm 进行包装，允许指定维度； """
    def __init__(self, channel, dim=-1):
        super(LayerNorm, self).__init__(channel)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        else:
            x = x.transpose(self.dim, -1)
            x = super(LayerNorm, self).forward(x)
            return x.transpose(self.dim, -1)


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
        query, key, value = self.calculate_qkv(x)
        position_embedding = self.get_position_embedding(x)
        attention_score = self.calculate_matrix(position_embedding, query, key)
        x = self.calculate_masked_attention_score(attention_score, mask, value)
        x = self.dropout(x)
        x += residual
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

        attention_score = attention_score.masked_fill(mask, -1e34)  # 将mask=0位置的分数，设置为很小的值

        attention_probs = torch.softmax(attention_score, dim=-1)
        attention_probs = attention_probs.masked_fill(mask, 0.0)  # 将mask=0位置的概率，设置为零
        attention_probs = self.dropout(attention_probs)

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
        x += residual
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
        :param mask: [batch, 1, time]
        :return: [batch, out_channel, time]
        """
        # 检查尺寸
        assert len(x.shape) == len(mask.shape) == 3, \
            f"输入音素序列的尺寸应该是三维 [batch, in_channel, time]，而 len(inputs.shape) = {len(x.shape)}"

        for block in self.blocks:
            x = block(x, mask)

        print(f"x.shape = {x.shape}, mask.shape = {mask.shape}")
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


class VariantPredictor(nn.Module):
    """ FastSpeech2 用来预测 F0、Energy、Duration 的模块；"""
    def __init__(self, num_blocks=2, in_out_channel=256, kernel_size=3, dropout_rate=0.2):
        super().__init__()

        self.backbone = nn.Sequential()
        for _ in range(num_blocks):
            self.backbone.append(
                nn.Conv1d(
                    in_channels=in_out_channel,
                    out_channels=in_out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
            )
            self.backbone.append(nn.ReLU())
            self.backbone.append(LayerNorm(in_out_channel, dim=1))
            self.backbone.append(nn.Dropout(dropout_rate))

        self.linear_output = nn.Linear(
            in_features=in_out_channel,
            out_features=1,
        )

    def forward(self, x, mask):
        """
        :param x: [batch, channel, time]
        :param mask: [batch, 1, time]
        :return: [batch, 1, time]
        """
        for layer in self.backbone:
            x = layer(x)  # [batch, channel, time]
        x = self.linear_output(x.transpose(1, 2)).transpose(1, 2)  # [batch, 1, time]
        return torch.mul(x, mask)  # [batch, 1, time]


class LengthRegulator(nn.Module):
    """ FastSpeech2 的长度规范模块； """
    def __init__(self):
        super().__init__()

    def forward(self, x, duration):
        """
        :param x: [batch, channel, time]
        :param duration: [batch, 1, time], masked;
        :return: [batch, channel, new_time], [batch, 1, new_time]
        """
        length = torch.sum(duration, dim=1)  # [batch, ]
        max_length = torch.max(length)  # [1, ]
        outputs = []
        for x_i, duration_i in zip(x, duration):
            output_i = torch.repeat_interleave(x_i, duration_i, dim=-1)  # [channel, time]
            if output_i.shape[-1] < max_length:
                zeros_pad = torch.zeros([output_i.shape[0], max_length - output_i.shape[-1]], dtype=output_i.dtype, device=output_i.device)
                output_i = torch.concat([output_i, zeros_pad], dim=-1)  # [channel, time]
            output_i = output_i.unsqueeze(0)  # [1, channel, time]
            outputs.append(output_i)

        outputs = torch.concat(outputs, dim=0)
        new_mask = torch.not_equal(outputs, 0)[:, 0, :].unsqueeze(1).int()
        return outputs, new_mask


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

    def forward(self, phoneme_ids, spk_id, duration_gt=None, f0_gt=None, energy_gt=None, mel_length=None, f0_length=None, energy=None):
        """
        :param phoneme_ids: [batch, time] 输入的音素序列；
        :param spk_id: [batch] 输入的音色ID
        :param duration_gt: [batch, time] 输入的每个音素对应的帧数；
        :param f0_gt: [batch, time] 输入的每帧对应的F0值；
        :param energy_gt: [batch, time] 输入的每帧对应的F0值；
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
        if duration_gt is not None:  # inference
            duration = duration_gt.int()
        else:  # train
            duration = nn.ReLU()(torch.exp(duration_predict) - 1)
            duration = duration.int()
        lr_outputs, lr_mask = self.length_regulator(encoder_outputs, duration)  # [batch, channel, new_time], [batch, 1, new_time]

        # 检查长度/帧数是否对得上：frames_pred, frames_duration_gts, mel_length, f0_length, energy_length
        # frames_pred = torch.sum(decoder_mask, dim=1)
        # frames_duration_gts = torch.sum(duration_gt, dim=1)

        # f0, energy 放在 LR 后面
        f0_predict = self.f0_predictor(lr_outputs, lr_mask)
        energy_predict = self.energy_predictor(lr_outputs, lr_mask)

        # f0 embedding
        if f0_gt is not None:  # inference
            f0_embedding = self.f0_dropout(self.f0_conv(f0_predict))
        else:  # train
            f0_embedding = self.f0_dropout(self.f0_conv(f0_predict))

        # energy embedding
        if energy_gt is not None:  # inference
            energy_embedding = self.energy_dropout(self.energy_conv(energy_predict))
        else:  # train
            energy_embedding = self.energy_dropout(self.energy_conv(energy_predict))

        lr_outputs += f0_embedding + energy_embedding

        # decoder
        decoder_outputs = self.decoder(lr_outputs, lr_mask)

        return decoder_outputs

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
