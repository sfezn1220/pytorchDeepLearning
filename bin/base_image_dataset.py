""" 定义：图像相关的数据集；"""
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


class ImageBaseDataList(BaseDataList):
    def __init__(self, conf: Dict, data_type: str = "train"):
        """
        定义数据集：输入数据数据的格式；
        :param conf: 数据集的参数；
        :param data_type: ["train", "valid"]
        :return:
        """
        super().__init__(conf, data_type)

        self.conf = conf

        self.input_shape = conf['input_shape']  # 默认 224*224
        self.n_classes = conf['n_classes']

        # 图像先保存在这里，减少训练时的内存占用
        self.images_cache_dir = os.path.join(conf["ckpt_path"], "images_cache_dir")  # npy格式的图像保存在这里

    def __iter__(self):
        super().__iter__()

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
