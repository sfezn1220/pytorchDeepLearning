""" 从mp4文件中，提取出每一帧图像； """
import os
import tqdm

import cv2
from .compute_ssim import compute_ssim


def extract_img_from_mp4(
        input_mp4_file: str,
        output_dir: str = "",
        prefix: str = "",
        start_sec: float = 0.0,
        end_sec: float = 100000000,
        ssim_score_threshold: float = 0.5,
):
    """
    从mp4文件中，提取出每一帧图像；
    :param input_mp4_file: 输入的mp3文件的路径；
    :param output_dir: 输出路径；
    :param prefix: 输出图像的文件民的前缀；
    :param start_sec: 从第几秒开始；
    :param end_sec: 到第几 秒结束；
    :param ssim_score_threshold: 判断图像和前一帧的相似度的阈值；
    :return:
    """

    # 输入视频
    assert os.path.exists(input_mp4_file), f"mp4 file not exist: {input_mp4_file} ."
    print(f"processing video file: {input_mp4_file}")
    cap = cv2.VideoCapture(input_mp4_file)

    # 输出路径
    if len(output_dir) < 1:
        output_dir = os.path.join(os.path.dirname(input_mp4_file), "extracted_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 计算：总帧数
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"frames_total = {frames_total}")

    # 计算：帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps = {fps}")

    # 从第几秒开始
    cap.set(cv2.CAP_PROP_POS_MSEC, int(1000 * start_sec))
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), frames_total)

    # 从第几帧开始，如果和前面一张图像的相似度较高，就不提取
    last_image = None
    # for frame_num in tqdm.tqdm(range(start_frame, end_frame)):
    for frame_num in range(start_frame, end_frame):
        # 获取一帧图像
        success, image = cap.read()
        if success is False:
            break
        # 输出文件名
        image_save_basename = "-".join([prefix, "{:06d}.png".format(frame_num)])
        image_save_path = os.path.join(output_dir, image_save_basename)

        # 判断与前一帧的相似度
        if last_image is None:
            cv2.imwrite(image_save_path, image)
            last_image = image
        else:
            ssim_score = compute_ssim(image, last_image)
            if ssim_score > ssim_score_threshold:
                continue
            else:
                cv2.imwrite(image_save_path, image)
                last_image = image

    print(f"Done! processed video file: {input_mp4_file}")
    return

