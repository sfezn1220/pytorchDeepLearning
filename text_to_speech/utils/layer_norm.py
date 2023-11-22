""" 包装一下 pytorch 的 LayerNorm 模块； """

import torch


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
