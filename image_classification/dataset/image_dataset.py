""" 定义：图像数据集；"""
import os
import tqdm

import yaml
import random
import torch
import cv2
import numpy as np
from typing import Dict

from bin.base_dataset import BaseDataList
from text_to_speech.utils import dict2numpy, numpy2dict  # 字典和np.array的转换

from image_classification.utils import read_json_lists
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


class ImageDataList(BaseDataList):
    def __init__(self, data_list_file: str, conf: Dict, data_type: str = "train"):
        """
        定义数据集：输入数据数据的格式；
        :param data_list_file: 训练集/验证集的label文件路径；
        :param conf: 数据集的参数；
        :param data_type: ["train", "valid"]
        :return:
        """
        super().__init__(conf, data_type)

        self.conf = conf

        self.input_shape = conf['input_shape']  # 默认 224*224
        self.n_classes = conf['n_classes']
        self.data_list = self.get_images_and_labels(data_list_file)  # 输入数据集，list 格式

        # 数据增强部分
        self.data_aug_transforms = self.get_data_aug_transforms()

        # 图像先保存在这里，减少训练时的内存占用
        self.images_cache_dir = os.path.join(conf["ckpt_path"], "images_cache_dir")  # npy格式的图像保存在这里

    def get_images_and_labels(self, data_list_file):
        """ 最开始的 读取 label 文件的操作；"""
        data_list = []
        for data in read_json_lists(data_list_file):
            data["label_id"] = torch.tensor(int(data["label_id"]))
            data["label_one_hot"] = F.one_hot(data["label_id"], self.n_classes).float()
            data["basename"] = os.path.basename(data['path']).replace(".png", "")
            data_list.append(data)
        return data_list

    def __iter__(self):
        if self.shuffle is True:
            random.Random(self.epoch).shuffle(self.data_list)  # 按照epoch设置random的随机种子，保证可复现

        self.data_aug_transforms = self.get_data_aug_transforms()  # 数据增强

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
                print(f"Ignore non-existing npz file and use opencv to read.: {load_path}")
                img = cv2.imread(data["path"])

            img = self.data_aug_transforms(img)  # 尺寸变为 224 * 224、转换成 tensor；
            data["image"] = img
            yield data

    def save_images(self):
        # 保存路径
        save_dir = self.images_cache_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # 去重
        basename_list = []

        # 开始保存数据
        print(f"开始保存 {self.data_type} 集的 numpy格式的训练数据：")
        for data in tqdm.tqdm(self.data_list):
            basename = data["basename"]
            # 查重
            if basename not in basename_list:
                basename_list.append(basename)
            else:
                print(f"重复的 basename: {basename}")

            # 如果已经保存了，就不再重复计算了
            save_path = os.path.join(save_dir, str(basename) + ".npz")
            if os.path.exists(save_path):
                continue

            img = cv2.imread(data["path"])
            data["image"] = img

            data_np = dict2numpy(data)  # 转换成特定格式的np.array
            np.savez(file=save_path, data=data_np)

            del data_np
            del data

        del basename_list
        return

    def get_data_aug_transforms(self):
        """定义：数据增强函数；"""

        # 读取图像
        aug_func_list = [
            transforms.ToPILImage(),
            # transforms.Resize(self.input_shape),
        ]

        # 水平翻转
        if self.conf['aug_horizontal_flip'] is True:
            p = self.conf['aug_horizontal_flip_p']
            aug_func_list.append(
                transforms.RandomHorizontalFlip(p=p)
            )
            print(f"use aug_horizontal_flip")

        # 竖直翻转
        if self.conf['aug_vertical_flip'] is True:
            p = self.conf['aug_vertical_flip_p']
            aug_func_list.append(
                transforms.RandomVerticalFlip(p=p)
            )
            print(f"use aug_vertical_flip")

        # 随机 pad
        if self.conf['aug_pad'] is True:
            p = self.conf['aug_pad_p']
            min_pad = self.conf['aug_pad_min']
            max_pad = self.conf['aug_pad_max']
            if random.random() < p:
                aug_func_list.append(
                    transforms.Pad((  # 随机 pad
                        random.randint(min_pad, max_pad),
                        random.randint(min_pad, max_pad),
                        random.randint(min_pad, max_pad),
                        random.randint(min_pad, max_pad),
                    ))
                )
            print(f"use aug_pad")

        # 随机旋转
        if self.conf['aug_rotation'] is True:
            p = self.conf['aug_rotation_p']
            min_rot = self.conf['aug_rotation_min']
            max_rot = self.conf['aug_rotation_max']
            if random.random() < p:
                aug_func_list.append(
                    transforms.RandomRotation([min_rot, max_rot])
                )
            print(f"use aug_rotation")

        # 随机高斯模糊
        if self.conf['aug_GaussianBlur'] is True:
            p = self.conf['aug_GaussianBlur_p']
            gaussian_blur_list = self.conf['aug_GaussianBlur_list']  # [3, 5, 7, 9, 11]
            if random.random() < p:
                aug_func_list.append(
                    transforms.GaussianBlur(
                        random.choice(gaussian_blur_list),
                    )
                )
            print(f"use aug_GaussianBlur")

        # 随机亮度调整
        if self.conf['aug_ColorJitter'] is True:
            p = self.conf['aug_ColorJitter_p']
            value = self.conf['aug_ColorJitter_value']  # 0.3
            if random.random() < p:
                aug_func_list.append(
                    transforms.ColorJitter(
                        brightness=(1-value, 1+value),  # 随机：亮度
                        contrast=(1-value, 1+value),  # 随机：对比度
                        saturation=(1-value, 1+value),  # 随机：饱和度
                        hue=(-value, value),  # 随机：色调
                    )
                )
            print(f"use aug_ColorJitter")

        # 将图像转换为 tensor
        aug_func_list += [
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
        ]

        return transforms.Compose(aug_func_list)


def get_image_dataloader(
        data_path: str,
        data_conf: dict,
        num_workers: int = 0,
        data_type: str = "train",
):
    """
    生成 train_dataloader、valid_dataloader、test_dataloader；
    :param data_path: label文件路径；
    :param data_conf: 数据集的参数；
    :param num_workers: 默认为 0；
    :param data_type: ["train", "valid"]
    :return:
    """
    dataset = ImageDataList(
        data_list_file=data_path,
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
    conf_file = "../example/commics/configs/resnet/resnet152_ft1.yaml"
    with open(conf_file, 'r', encoding='utf-8') as r1:
        configs = yaml.load(r1, Loader=yaml.FullLoader)

    # 测试 dataloader 的功能
    train_data_loader = get_image_dataloader(
        data_path=configs["train_data"],
        data_conf=configs,
    )

    # 测试是 shuffle 功能：
    for epoch in range(5):
        for batch_idx, batch in enumerate(train_data_loader):
            # if batch_idx == 0:
            print(f"batch[{batch_idx}]")
            print(f"ids[{len(batch['label_id'])}] = {batch['label_id']}")
            print(f"shape = {batch['image'].shape}")
            print()
