""" 定义：基础的数据集；与任务无关； """

import random
from typing import Dict
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader


class BaseDataList(IterableDataset):

    def __init__(self, conf: Dict, data_type: str = "train"):
        """
        定义：基础的数据集；与任务无关；
        :param conf: 数据集的参数；以字典的形式转入；
        :param data_type: ["train", "valid"]
        :return: 可迭代的数据集；
        """
        super().__init__()

        self.shuffle = conf['shuffle']  # 是否在每个 epoch 打乱数据集
        self.epoch = -1  # 当前epoch数量；用于打乱数据集；

        self.data_type = data_type  # train or valid dataset

        self.data_list = []  # 数据集存储在这里

    def set_epoch(self, epoch):
        """ 模型训练时，在每个epochs开始时，设置 epochs: """
        self.epoch = epoch
        print(f"{self.data_type} dataset: set epoch = {epoch}")  # type = train or valid

    def __iter__(self):
        """ 模型加载数据集的入口； """
        if self.shuffle is True:
            random.Random(self.epoch).shuffle(self.data_list)  # 按照epoch设置random的随机种子，保证可复现
        for data in self.data_list:
            yield data

    def __len__(self):
        """ 用于统计数据的量； """
        return len(self.data_list)
