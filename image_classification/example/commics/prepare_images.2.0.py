# -- coding: utf-8 --
""" 预处理图像数据：结合 iqiyi数据 & 已有数据； """

import os
import json
import shutil
import tqdm
import random
import cv2
from PIL import Image
import pandas as pd


class Dataset:
    def __init__(self):
        # 输入路径：
        self.iqiyi_dir = "G:\\爱奇艺_动画截图"
        self.iqiyi_excel = os.path.join(self.iqiyi_dir, "0.所有类别的汇总.xlsx")
        self.iqiyi_img_dir = os.path.join(self.iqiyi_dir, "personai_icartoonface_rectrain\\icartoonface_rectrain")
        # 输出路径
        self.output_root_dir = "G:\\Images"
        return

    @staticmethod
    def _norm_img_name(charactor="",
                       output_prefix="train", output_suffix="png",
                       date="20240721", random_length=11):
        """ 规范化命名文件； """
        # 先随机取字母，组成id：
        alpha_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z']
        assert len(alpha_list) == 26
        random_str = ""
        for _ in range(random_length):
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

    @staticmethod
    def _copy_and_norm(
            ori_path: str,
            new_path: str,
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
        else:
            image = Image.open(ori_path)
            image.save(new_path, "PNG")
        return

    def read_iqiyi_excel(self):
        """ 读取 iqiyi excel数据集（标注后） """
        data = pd.ExcelFile(self.iqiyi_excel)
        # 先读取类别名：
        df_1 = data.parse("类别名")
        anime_used = list(df_1.loc[:, "类别名"])
        # 再读取所需的角色名
        df_1 = data.parse("Sheet1")
        titles = list(df_1.loc[:, "title"])
        names = list(df_1.loc[:, "name"])
        ids = list(df_1.loc[:, "id"])

        item_used_list = []
        for t, n, i in zip(titles, names, ids):
            if t in anime_used:
                item_used_list.append(
                    {
                        "anime_type": "动画",
                        "anime_name": t,
                        "charactor": n,
                        "id": i,
                    }
                )
        return item_used_list

    def copy_and_norm_dir(self, item_used_list, min_cnt=50):
        """ 复制所需的图片，并重命名； """
        print(f"start copy and norm images...")
        total_cnt = 0
        for item in tqdm.tqdm(item_used_list):
            anime_type, anime_name, charactor, id_i = item["anime_type"], item["anime_name"], item["charactor"], item["id"]

            # 先确定输入输出目录
            ori_dir = os.path.join(self.iqiyi_img_dir, id_i)
            new_dir = os.path.join(self.output_root_dir, anime_type, anime_name, charactor).replace(" ", "")

            # 然后看输入的图片数量是否足够
            cnt_i = 0
            all_file_list = []
            for f in os.listdir(ori_dir):
                if not f.startswith("train"):
                    all_file_list.append(f)
            if len(all_file_list) < min_cnt:
                continue
            else:
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)

            # 正式复制数据
            for file in all_file_list:
                new_basename = self._norm_img_name(charactor=charactor,
                                                   output_prefix="train", date="20240721")
                ori_full_path = os.path.join(ori_dir, file)
                new_full_path = os.path.join(new_dir, new_basename)
                try:
                    self._copy_and_norm(ori_full_path, new_full_path)
                    cnt_i += 1
                except Exception as e:
                    print(f"skip ERROR \"{e}\" img: \"{ori_full_path}\"\"")

            print(f"copy and norm {str(cnt_i)} image of charactor {anime_name} : {charactor}")
            total_cnt += cnt_i

        print(f"\ncopy and norm {str(total_cnt)} images all.")
        return

    def main(self):
        # item_used_list = self.read_iqiyi_excel()
        # self.copy_and_norm_dir(item_used_list)

        return


if __name__ == "__main__":
    Dataset().main()
