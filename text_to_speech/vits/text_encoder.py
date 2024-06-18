""" 定义 VITS 模型的 音素编码前；"""

import torch
import torch.nn as nn

from text_to_speech.fastspeech2.conformer import ConformerEncoder


class TextEncoder(nn.Module):
    """ VITS 文本编码器模块； """
    def __init__(self, conf: dict):
        super().__init__()

        self.conf = conf
        self.conformer_encoder = ConformerEncoder(conf)

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """
        :param x: [batch, in_channel, time]
        :param mask: [batch, 1, time]
        :return: [batch, out_channel, time]
        """
        x = self.conformer_encoder(x * mask, mask)

        return x


class Solution:
    def search(self, nums: List[int], target: int) -> int:

        left = 0
        right = len(nums) - 1
        if right - left <= 1:
            if target in nums:
                return True
            else:
                return False
        else:
            mid = (right - left) // 2
            # 搜索左半边
            if nums[left] <= target and target <= nums[mid]:
                return True
            elif nums[mid + 1] <= target and target <= nums[right]:
                return True
            else:
                if nums[left] < nums[mid]:
                    return self.search(nums[mid + 1:right + 1])
                else:
                    return self.search(nums[:mid])

