""" 生成 attention 所需的 mask """


import torch


def get_attention_mask(phoneme_ids, padding_id: int = 0):
    """
    输入音素序列的mask，用于后续的 attention；
    :param phoneme_ids: [batch, time] 输入的音素序列；
    :param padding_id: padding 时的数字ID；
    :return: [batch, time] 与输入的尺寸相同，当音素=0时取0，当音素!=0时取1；
    """
    mask = torch.not_equal(phoneme_ids, padding_id)
    return mask.int()
