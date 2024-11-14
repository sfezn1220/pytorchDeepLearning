# -*- coding: utf-8 -*-
""" 预处理图像数据：结合 iqiyi数据 & 已有数据； """

import os
import json
import shutil
import time
import tqdm
import random
import cv2
from PIL import Image
import pandas as pd
from image_classification.utils.avif2png import avif2png


class ImgItem:
    def __init__(self, anime_type, anime_type_path, anime_dir, anime_dir_path, charactor, charactor_path, file_name, file_full_path):
        self.anime_type = anime_type
        self.anime_type_path = anime_type_path
        self.anime_dir = anime_dir
        self.anime_dir_path = anime_dir_path
        self.charactor = charactor
        self.charactor_path = charactor_path
        self.file_name = file_name
        self.file_full_path = file_full_path


class Dataset:
    def __init__(self):
        # 输出路径
        self.output_root_dir = "G:\\Images"
        # 随机字符的长度
        self.random_length = 11
        return

    def _remove_file(self, file_path):
        """ 删除一张图片 """
        for i in range(5):
            if not os.path.exists(file_path):
                break
            try:
                os.remove(file_path)
                break
            except:
                time.sleep(1)
        return

    def _norm_img_name(self, charactor="",
                       output_prefix="train", output_suffix="png",
                       date="20240721"):
        """ 规范化命名文件； """
        # 先随机取字母，组成id：
        alpha_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z']
        assert len(alpha_list) == 26
        random_str = ""
        for _ in range(self.random_length):
            random_str += random.choice(alpha_list)

        # 再拼完整的名称
        assert output_prefix in ["train", "test", "valid"], f'output_prefix must in ["train", "test", "valid"]'
        if not output_suffix.startswith("."):
            output_suffix = "." + output_suffix

        new_file_name = "_".join(
            [output_prefix, charactor, date, random_str]
        )

        new_file_name += output_suffix
        new_file_name = new_file_name.replace(" ", "")

        return new_file_name

    def _copy_and_norm(
            self,
            ori_path: str,
            new_path: str,
            delete_origin_file: bool = False,
    ):
        """ 复制一张图片，并规范化为 PNG 格式； """
        # 动图的规范方法：
        if ori_path.endswith(".gif"):
            i = 0
            basename = os.path.basename(new_path).replace(".png", "")
            gif = Image.open(ori_path)
            while True:
                current = gif.tell()
                image = gif.convert('RGB')
                if i % 5 == 0:  # 每 5帧保存一帧
                    new_file_name_i = basename[:-3] + str(i).rjust(3, "0") + ".png"
                    new_path_i = os.path.join(os.path.dirname(new_path), new_file_name_i)
                    image.save(new_path_i)
                try:
                    gif.seek(current + 1)
                    i += 1
                except:
                    break
                if i >= 20 * 5:
                    break
            gif.close()
        # 非动图的规范方法：
        elif ori_path.endswith(".avif"):
            cache_path1 = "./" + os.path.basename("cache.avif")
            cache_path2 = "./" + os.path.basename(new_path)
            shutil.copy(ori_path, cache_path1)
            image = Image.open(cache_path1)
            image.save(cache_path2, "PNG")
            shutil.copy(cache_path2, new_path)
            self._remove_file(cache_path1)
            self._remove_file(cache_path2)
        else:
            image = Image.open(ori_path)
            image.save(new_path, "PNG")

        if delete_origin_file is True and os.path.exists(ori_path):
            self._remove_file(ori_path)
        return

    def _collect_all_files(self, dataset_dir: str = "G:\\Images"):
        """ 收集所有的角色的路径 """
        res = []
        for anime_type in os.listdir(dataset_dir):
            anime_type_path = os.path.join(dataset_dir, anime_type)
            if not os.path.isdir(anime_type_path):
                continue

            for anime_dir in os.listdir(anime_type_path):
                anime_dir_path = os.path.join(dataset_dir, anime_type, anime_dir)
                if not os.path.isdir(anime_dir_path):
                    continue

                for charactor in os.listdir(anime_dir_path):
                    charactor_path = os.path.join(dataset_dir, anime_type, anime_dir, charactor)
                    if not os.path.isdir(charactor_path):
                        continue

                    for file in os.listdir(charactor_path):
                        file_full_path = os.path.join(dataset_dir, anime_type, anime_dir, charactor, file)
                        # 一个图片文件的所有信息
                        item = ImgItem(anime_type, anime_type_path, anime_dir, anime_dir_path, charactor, charactor_path, file, file_full_path)
                        res.append(item)
        return res

    def norm_dir(self, file_list: list[ImgItem]):
        """ 输入一个数据集，将 charactor 名称标准化、文件名标准化 """
        for img_item in tqdm.tqdm(file_list):
            file_name = img_item.file_name
            ori_full_path = img_item.file_full_path
            if file_name.endswith(".gif") or file_name.endswith(".GIF"):
                self._remove_file(ori_full_path)
                continue
            elif (((file_name.startswith("train") or file_name.startswith("test"))
                  and len(file_name.split("_")) == 4)
                  and len(file_name.split("_")[-1].split(".")[0]) == self.random_length):
                continue
            else:
                if file_name.startswith("test"):
                    output_prefix = "test"
                else:
                    output_prefix = "train"
            new_basename = self._norm_img_name(charactor=img_item.charactor, output_prefix=output_prefix, date="20240913")
            new_full_path = os.path.join(img_item.charactor_path, new_basename)
            try:
                self._copy_and_norm(ori_full_path, new_full_path, delete_origin_file=True)
                print(f"ori_basename = {file_name} , new_basename = {new_basename}")
            except Exception as e:
                self._remove_file(ori_full_path)
                print(f"skip and delete ERROR \"{e}\" img: \"{ori_full_path}\"\"")

        print(f"Done! norm_dir")
        return

    def divide_train_test(self, file_list: list[ImgItem]):
        """ 统计每个角色的训练集数量、测试集数量；测试集不足的数据，自动生成测试集 """
        charactor_train_map = {}
        charactor_test_map = {}
        for img_item in tqdm.tqdm(file_list, desc="正在划分训练集和测试集..."):
            # 拼接：角色唯一名称
            charactor_full = "_".join([img_item.anime_type, img_item.anime_dir, img_item.charactor])
            charactor_train_map.setdefault(charactor_full, [])
            charactor_test_map.setdefault(charactor_full, [])
            # 收集训练集和测试集
            if str(img_item.file_name).startswith("train"):
                charactor_train_map[charactor_full].append(img_item.file_name)
            elif str(img_item.file_name).startswith("test"):
                charactor_test_map[charactor_full].append(img_item.file_name)
            else:
                raise ValueError(f"这个文件既不是测试集也不是训练集：{img_item.file_name}")
            # TODO 划分测试集
            # TODO 记录到excel文件中
        return

    def main(self):
        # stage 1 收集所有文件
        file_list = self._collect_all_files("G:\\Images")
        # stage 2 规范化命名
        self.norm_dir(file_list)
        # stage 3 ing 统计每个角色的训练集数量、测试集数量；测试集不足的数据，自动生成测试集
        del file_list
        file_list = self._collect_all_files("G:\\Images")
        self.divide_train_test(file_list)
        # stage 4 TODO 每个角色，任意两个图片，去重

        return


if __name__ == "__main__":
    Dataset().main()
