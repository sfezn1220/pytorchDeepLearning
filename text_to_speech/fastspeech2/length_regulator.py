""" FastSpeech2 声学模型的长度规范模块； """

import torch
import torch.nn as nn


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
