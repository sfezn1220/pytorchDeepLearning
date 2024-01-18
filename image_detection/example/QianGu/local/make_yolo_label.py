"""
制作YOLO训练所需的label文件；
输入：
    原始的每帧截图；
    YOLO检测结果的校验结果（文件名中包含边框信息、人工过滤后，每个人物单独一个文件夹）；
输出：
    原始的每帧截图（过滤后、仅包含需要的人物）；
    带边框的每帧截图，用于人工校对；
    label文件，记录边框信息
"""

import os
import cv2
import copy
import tqdm
import numpy as np
import pypinyin


def person2pinyin(person):
    """ 汉字转拼音，并首字母大写"""
    pinyin = pypinyin.lazy_pinyin(person)
    res = ""
    for p in pinyin:
        res += p[0].upper() + p[1:].lower()
    return res


def collect_all_ori_images(
        input_ori_images: str,
):
    """
    收集：原始的每帧截图；
    :param input_ori_images:
    :return: {key = basename, value = full_path}
    """
    res = {}
    for file in os.listdir(input_ori_images):
        if not file.endswith(".png"):
            continue
        full_path = os.path.join(input_ori_images, file)
        basename = os.path.basename(file).replace(".png", "")

        res[basename] = full_path

    return res


def collect_all_labeled_images(
        input_labeled_images: str,
):
    """
    收集：YOLO切分好的、标注后的图像集，每个人物一个子文件夹；
    :param input_labeled_images:
    :return: {key = basename, value = [person_pinyin, full_path, x1, x2, y1, y2]}
    """
    # 先遍历每个人物的文件夹
    res = {}
    for person in os.listdir(input_labeled_images):
        personal_dir = os.path.join(input_labeled_images, person)
        # 再遍历当前人物的所有数据
        for file in os.listdir(personal_dir):
            if not file.endswith(".png"):
                continue
            full_path = os.path.join(personal_dir, file)
            basename, position = os.path.basename(file).replace(".png", "").split("-person-", 1)
            x1, x2, y1, y2 = [int(p) for p in position.split("-", 3)]

            person_pinyin = person2pinyin(person)

            if basename not in res:
                res[basename] = []
            res[basename].append([person_pinyin, full_path, x1, x2, y1, y2])

    return res


def main_make_yolo_label(
        input_ori_images: str,
        input_labeled_images: str,
        output_images: str,
        output_images_show: str,
):
    """
    制作YOLO训练所需的label文件；
    :param input_ori_images: 输入的原始图像的文件夹；
    :param input_labeled_images: YOLO检测并标注后的文件夹，每个人物一个子文件夹；
    :param output_images: 输出的图像的文件夹，也就是过滤后的；
    :param output_images_show: 输出的图像的文件夹，添加标注框，用于校对；
    :return:
    """

    # 检查输出目录
    if not os.path.exists(output_images):
        os.mkdir(output_images)
    if not os.path.exists(output_images_show):
        os.mkdir(output_images_show)

    # 收集：原始的每帧截图；
    basename_dic = collect_all_ori_images(input_ori_images)

    # 收集：YOLO切分好的、标注后的图像集，每个人物一个子文件夹；
    labeled_dic = collect_all_labeled_images(input_labeled_images)

    # 遍历每个标注好的图像
    for basename in tqdm.tqdm(labeled_dic):
        if basename not in basename_dic:
            continue

        # 读取原始图像
        ori_full_path = basename_dic[basename]
        image = cv2.imread(ori_full_path)
        x_max, y_max = np.shape(image)[:2]  # 1080, 1920

        # 保存标注后的图像
        save_path = os.path.join(output_images, basename + ".png")
        cv2.imwrite(save_path, image)
        # 保存label文件
        label_path = os.path.join(output_images, basename + ".txt")

        # 保存带有标注框的图像、便于核对
        image_labeled = copy.deepcopy(image)
        with open(label_path, "w", encoding="utf-8") as w1:
            for person, _, x1, x2, y1, y2 in labeled_dic[basename]:
                image_labeled = cv2.rectangle(image, [y1, x1], [y2, x2], (0, 0, 255), 10)
                image_labeled = cv2.putText(img=image_labeled, text=str(person), org=(y1, x1+20), fontFace=1,
                                            fontScale=3, color=(0, 255, 0), thickness=2)
                w1.write(" ".join([person, str(x1/x_max), str(x2/x_max), str(y1/y_max), str(y2/y_max)]) + "\n")

        save_path = os.path.join(output_images_show, basename + ".png")
        cv2.imwrite(save_path, image_labeled)

    return


if __name__ == "__main__":
    main_make_yolo_label(
        input_ori_images="G:\\Images_QianGu\\1.extract_images",
        input_labeled_images="G:\\Images_QianGu\\3.labeled_demo",
        output_images="G:\\Images_QianGu\\4.labeled_norm",
        output_images_show="G:\\Images_QianGu\\4.labeled_norm_show"
    )
