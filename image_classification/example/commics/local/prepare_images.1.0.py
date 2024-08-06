# -- coding: utf-8 --
""" 预处理图像数据； """

import os
import json
import shutil
import tqdm
import random
import cv2
import imageio
import pillow_avif
from PIL import Image


def label_file_to_json(
        read_file: str,
        write_file: str,
        spk_map_file: str,
        valid_max_cnt: int = 1000,
):
    """读取 label 文件，生成 json 格式的 label 文件；"""
    assert read_file != write_file, f"输入、输出文件不能相同；"
    # 建立从类别到ID的映射：
    character_id_to_id_dict = {}
    character_id_to_ch_dict = {}
    with open(spk_map_file, 'r', encoding='utf-8') as r1:
        for line in r1.readlines():
            line = str(line).strip()
            if line.startswith("场景ID") or len(line.strip()) <= 1:  # 去掉标题行
                continue
            try:
                scene_id, scene, character, character_id, label_id = line.split("\t")[:5]
                character_id_to_id_dict[character_id] = label_id
                character_id_to_ch_dict[character_id] = scene + "_" + character
            except Exception as e:
                print(f"load character ID error: {line.strip()} with {e}")

    # 正式生成 json 格式的 label 文件
    train_cnt = 0
    valid_cnt = 0
    test_cnt = 0
    with open(read_file, 'r', encoding='utf-8') as r1, \
            open(write_file.replace(".txt", ".train.txt"), 'w', encoding='utf-8') as w1, \
            open(write_file.replace(".txt", ".valid.txt"), 'w', encoding='utf-8') as w2, \
            open(write_file.replace(".txt", ".test.txt"), 'w', encoding='utf-8') as w3:
        # "character_id", "path", "train_or_test"
        all_data = r1.readlines()
        random.shuffle(all_data)  # 打乱数据集
        for line in all_data:
            line = line.strip()
            if line.startswith("character_id"):
                continue

            character_id, path, train_or_test = line.split("\t")
            label_id = character_id_to_id_dict[character_id]
            character = character_id_to_ch_dict[character_id]

            if not os.path.exists(path):
                # print(f"Skip not exists path: {path}")
                continue

            json_line = {
                "path": path,
                "label_id": label_id,
                "character_id": character_id,
                "character": character,
            }
            json_line = json.dumps(json_line, ensure_ascii=False)  # 加这个参数，让中文正常显示

            if train_or_test == "train":
                if valid_cnt < valid_max_cnt:
                    w2.write(json_line + "\n")
                    valid_cnt += 1
                else:
                    w1.write(json_line + "\n")
                    train_cnt += 1
            elif train_or_test == "test":
                w3.write(json_line + "\n")
                test_cnt += 1
            print(f"write {train_cnt + valid_cnt + test_cnt} json line...", end="\r", flush=True)

    print(f"totally write {train_cnt + valid_cnt + test_cnt} json line.")

    return


def random_basename(
        lens: int = 27,
) -> str:
    """随机生成字符串，长度 = lens"""
    alpha_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    assert len(alpha_list) == 26

    res = ""
    for i in range(lens):
        random.shuffle(alpha_list)
        res += alpha_list[0]

    return res


def get_image_size(
        path: str,
) -> tuple[int, int]:
    """ 输入图像路径，返回长、宽数值；"""
    if not os.path.exists(path):
        return -1, -1
    else:
        image = cv2.imread(path)
        length, width, channels = image.shape
        return length, width


def copy_and_norm(
        input_path: str,
        output_path: str,
):
    """ 将图像重命名、规范化PNG格式图像；"""
    cache_path = os.path.join(
        os.path.dirname(output_path),
        str(os.path.basename(output_path)).replace(".png", ".tmp.") + str(os.path.basename(input_path)).split(".")[-1],
    )

    shutil.copy(input_path, cache_path)

    # 动图的规范方法：
    if cache_path.endswith(".gif"):
        i = 0
        basename = os.path.basename(output_path).replace(".png", "")
        gif = Image.open(cache_path)
        while True:
            current = gif.tell()
            image = gif.convert('RGB')
            if i % 5 == 0:  # 每 5帧保存一帧
                image.save(
                    os.path.join(os.path.dirname(output_path), basename[:-3] + str(i).rjust(3, "0") + ".png")
                )
            try:
                gif.seek(current + 1)
                i += 1
            except:
                break
            if i >= 20 * 5:
                break
        gif.close()
    # 非动图的规范方法：
    elif cache_path.endswith(".avif"):
        image = Image.open(cache_path)
        image.save(output_path, "PNG")
    else:
        image = cv2.imread(cache_path)
        cv2.imwrite(output_path, image)

    os.remove(cache_path)

    return


def merge_new_image_to_old_dir(
        ori_data_dir: str,
        copy_data_dir: str,
):
    """ 将一批新到的数据，合并到所有原始数据的公共文件夹中； """
    # 统计一共有多少可用的数据、可用的类别
    total_files = 0
    total_character = 0

    # 遍历每个场景
    for scene in tqdm.tqdm(os.listdir(ori_data_dir)):
        scene = str(scene)

        # 遍历当前场景下的每个类别：
        for character in os.listdir(os.path.join(ori_data_dir, scene)):
            character = str(character)

            ori_full_dir = os.path.join(ori_data_dir, scene, character)
            new_full_dir = os.path.join(copy_data_dir, scene, character)

            # 统计当前类别的所有数据：
            file_list = os.listdir(ori_full_dir)

            # 建立复制路径
            if not os.path.exists(new_full_dir):
                os.makedirs(new_full_dir)

            # 复制到新目录下
            for file_select in file_list:
                ori_full_path = os.path.join(ori_full_dir, file_select)
                new_full_path = os.path.join(new_full_dir, file_select)
                shutil.copy(ori_full_path, new_full_path)

            # 统计数据
            total_character += 1
            total_files += len(file_list)

            print(f"Done: {character}")

    # 展示统计结果
    print(f"total_character = {total_character}")
    print(f"total_files = {total_files}")

    return


def random_select_test_dataset(
        ori_data_dir: str,
        copy_data_dir: str,
        min_cnt_per: int = 100,
        select_cnt: int = 10,
        suffix: str = ","
):
    """
        从训练数据中，随机选取一些数据，复制出来，作为测试集；需要人工检查；
        ori_data_dir: 从这里读取图像数据；
        copy_data_dir: 将随机选的数据放在这里，等待人工查验；
        min_cnt_per: 每个类别最少的数据量，默认为 100；少于这个数据量的话，不会读取这个类别数据、也不会用于训练；
        select_cnt: 每个类别随机选取的数据量，默认为 10；
        suffix: 新一批的类别名的标志符，用于和以前的分开；
    """

    # 统计一共有多少可用的数据、可用的类别
    total_files = 0
    total_character = 0

    # 遍历每个场景
    for scene in tqdm.tqdm(os.listdir(ori_data_dir)):
        scene = str(scene)

        if scene.startswith("00"):  # 特殊类别，不用处理
            continue

        # 遍历当前场景下的每个类别：
        for character in os.listdir(os.path.join(ori_data_dir, scene)):
            character = str(character)

            ori_full_dir = os.path.join(ori_data_dir, scene, character)
            new_full_dir = os.path.join(copy_data_dir, scene, character)

            # 统计当前类别的所有数据：
            file_list = os.listdir(ori_full_dir)

            if len(file_list) < min_cnt_per:
                print(f"{character} has {len(file_list)} data, less than {min_cnt_per}")
                continue

            # 建立复制路径
            if os.path.exists(new_full_dir):
                # 已有这个训练数据了
                continue
            else:
                new_full_dir += suffix
                os.makedirs(new_full_dir)

            # 随机取数据
            random.shuffle(file_list)
            file_select_list = file_list[:select_cnt]

            # 复制到新目录下
            for file_select in file_select_list:
                ori_full_path = os.path.join(ori_full_dir, file_select)
                new_full_path = os.path.join(new_full_dir, file_select)
                shutil.copy(ori_full_path, new_full_path)

            # 统计数据
            total_character += 1
            total_files += len(file_list)

    # 展示统计结果
    print(f"total_character = {total_character}")
    print(f"total_files = {total_files}")

    return


def split_train_test(
        ori_data_dir: str,
        test_data_dir: str,
        new_data_dir: str,
        label_path: str,
        spk_map_file: str,
):
    """
        根据前面生成与检查的测试数据，将所有数据进行规范化、重命名，并复制到新路径下；
        ori_data_dir: 所有的、原始数据路径；
        test_data_dir: 前面生产并检查的测试数据路径；
        new_data_dir: 规范化、重命名后的所有数据，复制到这里；
        label_path: 生成 label 文件，保存规范化、重命名后的所有数据
        spk_map_file: 从类别到ID的映射文件；
    """

    # 建立从类别到ID的映射：
    spk_to_id_dict = {}
    with open(spk_map_file, 'r', encoding='utf-8') as r1:
        for line in r1.readlines():
            line = str(line).strip()
            if line.startswith("场景ID") or len(line.strip()) <= 1:  # 去掉标题行
                continue
            try:
                scene_id, scene, character, character_id = line.split("\t")[:4]
                full_character = scene + "_" + character
                spk_to_id_dict[full_character] = character_id
            except Exception as e:
                print(f"load character ID error: {line.strip()} with {e}")

    # 统计一共有多少可用的数据、可用的类别
    total_character = 0
    total_train_files = 0
    total_test_files = 0

    # 统计最大、最小尺寸：length: size[0]; width: size[1];
    max_length = 0
    max_width = 0
    max_size = 0
    min_length = 1e10
    min_width = 1e10
    min_size = 1e10

    # 先检查下是否所有类别都在 map 里
    all_character_list = []
    for scene in os.listdir(test_data_dir):
        scene = str(scene)
        if not os.path.isdir(os.path.join(test_data_dir, scene)):
            continue
        for character in os.listdir(os.path.join(test_data_dir, scene)):
            character = str(character).replace("动画_", "_")
            assert character in spk_to_id_dict, f"不在 map 里的类别：{character}"
            all_character_list.append(character)
    # 再检查哪些类别的数量不足：
    for character in spk_to_id_dict:
        if character not in all_character_list:
            print(f"训练数据不足的 character：{character}")

    # 遍历每个有测试数据的场景、写入 label 文件：
    with open(label_path, 'w', encoding='utf-8') as w1:
        w1.write("\t".join(["character_id", "path", "train_or_test"]) + "\n")

        for scene in tqdm.tqdm(os.listdir(test_data_dir), desc="场景数量"):
            scene = str(scene)

            if not os.path.isdir(os.path.join(test_data_dir, scene)):
                continue

            # 遍历当前场景下的每个类别：
            for character in tqdm.tqdm(
                os.listdir(os.path.join(test_data_dir, scene)),
                desc=f"场景：{scene}",
                leave=False,
            ):
                character = str(character)
                character_id = spk_to_id_dict[character.replace("动画_", "_")]

                # 统计每个类别有多少可用的数据
                character_train_files = 0
                character_test_files = 0

                ori_full_dir = os.path.join(ori_data_dir, scene, character)  # 原始数据
                test_full_dir = os.path.join(test_data_dir, scene, character)  # 选中的测试数据

                # 统计当前类别的所有 测试数据：
                test_file_list = os.listdir(test_full_dir)

                # 将原始数据复制到新路径下：
                for file in os.listdir(ori_full_dir):
                    file = str(file)

                    # 新的文件名
                    if file in test_file_list:
                        new_file = "test_" + random_basename(lens=27) + ".png"
                        total_test_files += 1
                        character_test_files += 1
                        new_full_dir = os.path.join(new_data_dir, character_id, "test")  # 复制、重命名后，放在这里
                        if not os.path.exists(new_full_dir):
                            os.makedirs(new_full_dir)
                        new_full_path = os.path.join(new_full_dir, new_file)
                        w1.write("\t".join([character_id, new_full_path, "test"]) + "\n")
                    else:
                        new_file = "train_" + random_basename(lens=27) + ".png"
                        total_train_files += 1
                        character_train_files += 1
                        new_full_dir = os.path.join(new_data_dir, character_id, "train")  # 复制、重命名后，放在这里
                        if not os.path.exists(new_full_dir):
                            os.makedirs(new_full_dir)
                        new_full_path = os.path.join(new_full_dir, new_file)
                        w1.write("\t".join([character_id, new_full_path, "train"]) + "\n")

                    # 正式复制数据
                    ori_full_path = os.path.join(ori_full_dir, file)
                    try:
                        copy_and_norm(input_path=ori_full_path, output_path=new_full_path)
                    except Exception as e:
                        print(f"Error: {ori_full_path}")
                        print(f"{e}")

                    # 统计长、宽
                    length, width = get_image_size(new_full_path)
                    if length < 0 or width < 0:
                        continue
                    max_length = max(max_length, length)
                    max_width = max(max_width, width)
                    max_size = max(max_size, length * width)
                    min_length = min(min_length, length)
                    min_width = min(min_width, width)
                    min_size = min(min_size, length * width)

                # 统计数据
                total_character += 1

                # 展示每个类别的统计结果
                print(f"{character} has {character_train_files} train data and {character_test_files} test data.")

    # 展示所有统计结果
    print(f"total_character = {total_character}")
    print(f"total_train_files = {total_train_files}")
    print(f"total_test_files = {total_test_files}")

    print(f"max_length = {max_length}")
    print(f"max_width = {max_width}")
    print(f"max_size = {max_size}")
    print(f"min_length = {min_length}")
    print(f"min_width = {min_width}")
    print(f"min_size = {min_size}")

    return


if __name__ == "__main__":
    start_stage = 3
    stop_stage = 3

    # stage 0: 将一批新到的数据，合并到所有原始数据的公共文件夹中；
    if 0 <= start_stage <= 0:
        merge_new_image_to_old_dir(
            ori_data_dir="G:\Images\\10.todo",
            copy_data_dir="G:\Images\\0.ori_downloads",
        )

    # stage 1: 从训练数据中，随机选取一些数据，复制出来，作为测试集；需要人工检查；
    if 1 <= start_stage <= 1:
        random_select_test_dataset(
            ori_data_dir="G:\Images\\0.ori_downloads",
            copy_data_dir="G:\Images\\1.test_dataset_cache",
            min_cnt_per=100,
            select_cnt=10,
            suffix="(新）"
        )

    # stage 2: 根据前面生成与检查的测试数据，将所有数据进行规范化、重命名，并复制到新路径下；
    if 2 <= start_stage <= 2:
        split_train_test(
            ori_data_dir="G:\Images\\0.ori_downloads",
            test_data_dir="G:\Images\\1.test_dataset_cache",
            new_data_dir="G:\Images\\2.norm_rename",
            label_path="G:\Images\\2.labeled_0622.txt",
            spk_map_file="G:\Images\\spk-map.txt",
        )

    # stage 3: label 文件 -> json label 文件；
    if 3 <= start_stage <= 3:
        label_file_to_json(
            read_file="G:\Images\\2.labeled_0622.txt",
            write_file="G:\Images\\3.labeled_json_0622.txt",
            spk_map_file="G:\Images\\spk-map.txt",
            valid_max_cnt=1000,
        )
