""" FastSpeech 声学模型预测 F0、energy、duration 的模块； """

import torch
import torch.nn as nn

from text_to_speech.utils.layer_norm import LayerNorm


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
        y = self.linear_output(x.transpose(1, 2)).transpose(1, 2)  # [batch, 1, time]
        return torch.mul(y.clone(), mask)  # [batch, 1, time]
