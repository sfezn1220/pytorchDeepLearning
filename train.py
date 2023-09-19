""" 训练、测试的主函数； """

import os
import torch
import argparse
import copy
import yaml
import logging

from dataset import get_image_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='training models.')

    parser.add_argument('--gpu', type=int, default=-1, help='0 for gpu and -1 for cpu')
    parser.add_argument('--config', type=str, required=True, help='yaml 格式的配置文件')
    parser.add_argument('--train_data', type=str, required=True, help='训练集label文件，每行一条json格式数据')
    parser.add_argument('--valid_data', type=str, required=True, help='验证集label文件，每行一条json格式数据')

    args = parser.parse_args()
    return args


class HyperParameters:
    """ 所有训练用的超参数；"""
    gpu = "0"
    config = {
        "shuffle": True,
    }

    train_data = "G:\\Images\\2.labeled_json_0902.txt"
    valid_data = train_data


def main():
    # 超参数
    hp = HyperParameters

    # GPU or CPU
    gpu = str(hp.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if gpu == "-1":
        logging.info(f"Use device: CPU.")
    elif gpu == "0":
        logging.info(f"Use device: GPU {gpu}.")
    else:
        raise ValueError(f"\"--gpu\" must in [-1, 0], while input is {gpu}")

    # Set random seed
    torch.manual_seed(777)

    # 读取 configs.yaml
    train_conf = hp.config
    # with open(, 'r', encoding='utf-8') as r1:
    #     configs = yaml.load(r1, Loader=yaml.FullLoader)

    valid_conf = copy.deepcopy(train_conf)
    valid_conf['shuffle'] = False

    # 数据集：
    train_data_loader, valid_data_loader = get_image_dataloader(
        train_data=hp.train_data,
        valid_data=hp.valid_data,
        train_conf=train_conf,
        valid_conf=valid_conf,
        batch_size=16,
        num_workers=0,
    )

    steps_per_epoch = len(train_data_loader)
    print(f"steps_per_epoch = {steps_per_epoch}")


if __name__ == "__main__":
    main()
