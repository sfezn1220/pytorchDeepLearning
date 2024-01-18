""" 输入两个cv2读取后的图像，输出ssim 结构相似度； """
import cv2
import numpy as np


def compute_ssim(image_1, image_2):
    """ 
    输入两个cv2读取后的图像，输出ssim 结构相似度； 
    ref: https://blog.51cto.com/u_16213384/7305492
    """

    # 转灰度图，即单通道
    image_1_grey = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2_grey = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    # 计算均值、方差、协方差
    mean_1, var_1 = np.mean(image_1_grey), np.var(image_1_grey)
    mean_2, var_2 = np.mean(image_2_grey), np.var(image_2_grey)
    co_var = np.cov(image_1_grey.flatten(), image_2_grey.flatten())[0][1]

    # 设置分母不为零的常数
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # 计算SSIM
    p1 = (2 * mean_1 * mean_2 + c1) * (2 * co_var + c2)
    p2 = (mean_1 ** 2 + mean_2 ** 2 + c1) * (var_1 + var_2 + c2)
    ssim = p1 / p2

    return ssim
