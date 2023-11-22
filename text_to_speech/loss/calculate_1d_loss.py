""" 计算两个一维向量之间的 loss """

import torch
import torch.nn as nn


def calculate_1d_loss(gt: torch.tensor, predict: torch.tensor, loss_func: str = "MSE"):
    """
    计算两个一维向量之间的 loss
    :param gt: [batch, time1] or [batch, 1, time1]
    :param predict: [batch, time2] or [batch, 1, time2]
    :param loss_func: MSE or MAE
    :return: float32
    """
    # [batch, 1, time1] -> [batch, time1]
    if len(gt.shape) == 3:
        gt = gt.squeeze(1)  # [batch, time]
    if len(predict.shape) == 3:
        predict = predict.squeeze(1)  # [batch, time]

    # 裁剪到相同长度
    if gt.shape[-1] > predict.shape[-1]:
        gt = gt[:, :predict.shape[-1]]
    elif gt.shape[-1] < predict.shape[-1]:
        predict = predict[:, :gt.shape[-1]]

    if loss_func == "MSE":
        return nn.MSELoss()(predict, gt)
    elif loss_func == "MAE":
        return nn.L1Loss()(predict, gt)
    else:
        raise ValueError(f"loss_function must in ['MSE', 'MAE']")
