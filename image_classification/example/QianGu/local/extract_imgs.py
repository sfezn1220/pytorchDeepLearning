"""" 读取一个路径下的所有视频，提取图像。并根据图像相似度，进行过滤； """
import os
import threading
from image_classification.utils import extract_img_from_mp4


def main_extract_img(
        input_mp4_dir: str,
        output_mp4_dir: str,
    ):
    """" 读取一个路径下的所有视频，提取图像。并根据图像相似度，进行过滤； """

    # 收集mp4文件
    mp4_list = []
    for file in os.listdir(input_mp4_dir):
        if file.endswith(".mp4"):
            mp4_list.append(
                os.path.join(input_mp4_dir, file)
            )

    # 排序，升序
    mp4_list.sort()

    for mp4_file in mp4_list:
        prefix = os.path.basename(mp4_file).replace(".mp4", "").replace("千古玦尘", "QianGuJueChen")

        start_sec = 0  # 1.5 * 60
        end_sec = 70 * 60

        part = 1  # 每个视频分成几份
        for part_i in range(part):

            timestamp = (end_sec - start_sec) / part
            start_sec_i = start_sec + timestamp * part_i
            end_sec_i = start_sec_i + timestamp

            thread_i = threading.Thread(target=extract_img_from_mp4, args=(mp4_file, output_mp4_dir, prefix, start_sec_i, end_sec_i, 0.5, ))
            thread_i.start()
            thread_i.join(5)

        # extract_img_from_mp4(
        #     input_mp4_file=mp4_file,
        #     output_dir=output_mp4_dir,
        #     prefix=prefix,
        #     start_sec=90,
        #     ssim_score_threshold=0.5,
        # )

    return


if __name__ == "__main__":
    main_extract_img(
        input_mp4_dir="G:\\Images_AllOfUsAreDead\\0.ori_mp4",
        output_mp4_dir="G:\\Images_AllOfUsAreDead\\1.extract_images"
    )
