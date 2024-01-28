""" 定义：图像检测的数据集；"""
import os
import tqdm

import yaml
import random
import torch
import cv2
import numpy as np
from typing import Dict

from bin.base_image_dataset import ImageBaseDataList
from text_to_speech.utils import dict2numpy, numpy2dict  # 字典和np.array的转换

from image_classification.utils import read_json_lists
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


class ImageDataList(ImageBaseDataList):
    def __init__(self, data_dir: str, conf: Dict, data_type: str = "train"):
        """
        定义数据集：输入数据数据的格式；
        :param data_dir: 训练集/验证集的文件夹，里面包含：图像、同名的txt标注文件；
        :param conf: 数据集的参数；
        :param data_type: ["train", "valid"]
        :return:
        """
        super().__init__(conf, data_type)

        self.data_list = self.get_images_and_labels(data_dir)  # 输入数据集

    def get_images_and_labels(self, data_dir):
        """ 最开始的 读取 label 文件的操作；"""
        data_list = []
        for img_file in os.listdir(data_dir):
            if not img_file.endswith(".png"):
                continue
            # 找到图像、及其对应的label文件
            label_file = img_file.replace(".png", ".txt")
            label_file_path = os.path.join(data_dir, label_file)
            img_file_path = os.path.join(data_dir, img_file)
            if not os.path.exists(label_file_path):
                continue
            # 生成一条数据
            data = dict()
            data["path"] = str(img_file_path)
            data["basename"] = img_file.replace(".png", "")
            data["anchor"] = [["", "", "", "", ""] for _ in range(10)]
            with open(label_file_path, 'r', encoding='utf-8') as r1:
                for i, line in enumerate(r1.readlines()):
                    person, *xxyy = line.strip().split(" ", 4)
                    x1, x2, y1, y2 = [float(p) for p in xxyy]
                    data["anchor"][i] = [person, x1, x2, y1, y2]
            data_list.append(data)
        return data_list

    def __iter__(self):
        if self.shuffle is True:
            random.Random(self.epoch).shuffle(self.data_list)  # 按照epoch设置random的随机种子，保证可复现

        self.data_aug_transforms = self.get_data_aug_transforms()  # 数据输入、增强

        for data in self.data_list:
            basename = data["basename"]
            # 先看有没有本地缓存数据
            load_path = os.path.join(self.images_cache_dir, str(basename) + ".npz")
            if os.path.exists(load_path):
                new_data_np = np.load(load_path, allow_pickle=True)["data"]
                data = numpy2dict(new_data_np)
                img = data["image"]
                del new_data_np
            else:
                # print(f"Ignore non-existing npz file and use opencv to read.: {load_path}")
                img = cv2.imread(data["path"])

            img = self.data_aug_transforms(img)  # 尺寸变为 416 * 416、转换成 tensor；
            data["image"] = img
            yield data

    def get_data_aug_transforms(self):
        """定义：数据增强函数；"""

        # 读取图像
        aug_func_list = [
            transforms.ToPILImage(),
            # transforms.Resize(self.input_shape),
        ]

        # 将图像转换为 tensor
        aug_func_list += [
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
        ]

        return transforms.Compose(aug_func_list)


def get_image_dataloader(
        data_dir: str,
        data_conf: dict,
        num_workers: int = 0,
        data_type: str = "train",
):
    """
    生成 train_dataloader、valid_dataloader、test_dataloader；
    :param data_dir: 图像机器label文件路径；
    :param data_conf: 数据集的参数；
    :param num_workers: 默认为 0；
    :param data_type: ["train", "valid"]
    :return:
    """
    dataset = ImageDataList(
        data_dir=data_dir,
        conf=data_conf,
        data_type=data_type,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=data_conf["batch_size"],
        num_workers=num_workers,
    )

    print(f"{data_type} steps_per_epoch = {len(data_loader)}")

    return data_loader


if __name__ == "__main__":
    # config 文件
    conf_file = "../example/QianGu/configs/yolov3_demo.yaml"
    with open(conf_file, 'r', encoding='utf-8') as r1:
        configs = yaml.load(r1, Loader=yaml.FullLoader)

    # 测试 dataloader 的功能
    train_data_loader = get_image_dataloader(
        data_dir=configs["train_data"],
        data_conf=configs,
    )

    # 测试是 shuffle 功能：
    for epoch in range(1):
        for batch_idx, batch in enumerate(train_data_loader):
            # if batch_idx == 0:
            print(f"batch[{batch_idx}]")
            print(f"shape = {batch['image'].shape}")
            print()
