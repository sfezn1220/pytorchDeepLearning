"""
1、将 avif 图片，替换为 png 图片；
2、将一个文件夹里的所有 avif 图片，替换为 png 图片；
"""
import os
import tqdm
import pillow_avif
from PIL import Image


def avif2png(
        input_file_or_dir: str,
        delete_ori_img: bool = False,
):
    # 找出 avif 文件
    avif_list = []

    if os.path.isdir(input_file_or_dir):
        for file in os.listdir(input_file_or_dir):
            if file.endswith("avif"):
                full_path = os.path.join(input_file_or_dir, file)
                avif_list.append(full_path)
    elif os.path.isfile(input_file_or_dir):
        if input_file_or_dir.endswith("avif"):
            avif_list.append(input_file_or_dir)
    else:
        raise ValueError(f"输入的变量 \"input_file_or_dir\" 必须是 avif 文件，或包含 avif 文件。")
    print(f"Totally has {len(avif_list)} avif files.")

    # 遍历所有 avif 文件
    for avif_file in tqdm.tqdm(avif_list):
        ori_dir_name = os.path.dirname(avif_file)
        ori_basename = os.path.basename(avif_file)
        new_basename = str(ori_basename).replace(".avif", ".png")
        new_full_path = os.path.join(ori_dir_name, new_basename)

        try:
            img = Image.open(avif_file)
            img.save(new_full_path, "PNG")
        except:
            continue

        if delete_ori_img is True and os.path.exists(new_full_path) and os.path.exists(avif_file):
            os.remove(avif_file)

    return


if __name__ == "__main__":
    avif2png(
        input_file_or_dir="D:\\Desktop",
        delete_ori_img=True,
    )
