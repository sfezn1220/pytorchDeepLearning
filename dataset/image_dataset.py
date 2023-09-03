""" 定义：图像数据集；"""
import random
import torch
from torch.utils.data import IterableDataset
from utils import read_json_lists
from torch.utils.data import DataLoader


class ImageDataList(IterableDataset):
    def __init__(self, data_list: list, shuffle: bool=False):
        super(ImageDataList).__init__()
        self.data_list = data_list  # list 格式的输入数据集
        self.shuffle = shuffle  # 是否在每个epoch都打乱数据集
        self.epoch = -1  # 按照epcoh设置random的随机种子，保证可复现

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle is True:
            random.Random(self.epoch).shuffle(self.data_list)  # 按照epcoh设置random的随机种子，保证可复现
        for data in self.data_list:
            yield data


def image_dataset(
        data_list_file: str,
        conf: dict,
):
    """
    定义数据集：输入数据数据的格式；
    :param data_list_file: 训练集/验证集的label文件路径；
    :param conf: 数据集的参数；
    :return:
    """
    data_list = read_json_lists(data_list_file)
    shuffle = conf.get('shuffle', False)

    dataset = ImageDataList(data_list, shuffle)
    return dataset


if __name__ == "__main__":
    data_list_file = "G:\\Images\\2.labeled_json_0902.txt"  # 测试数据，每行就是一条数据
    conf = {"shuffle": True}  # 测试 shuffle 功能

    # 测试 dataset 的功能
    dataset = image_dataset(
        data_list_file,
        conf,
    )

    # 测试 dataloader 的功能
    data_loader = DataLoader(
        dataset,
        batch_size=16,
    )

    # 测试以上功能，尤其是 shuffle 功能：
    for epoch in range(5):
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx == 0:
                print(batch_idx, batch)
