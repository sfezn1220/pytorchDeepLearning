""" 训练、测试的主函数； """

import os
import torch
import argparse
import copy
from torch.utils.data import DataLoader
import yaml
import logging

from dataset import image_dataset


def get_args():
    parser = argparse.ArgumentParser(description='training models.')

    parser.add_argument('--gpu', type=int, default=-1, help='0 for gpu and -1 for cpu')
    parser.add_argument('--config', type=str, required=True, help='yaml 格式的配置文件')
    parser.add_argument('--train_data', type=str, required=True, help='训练集label文件，每行一条json格式数据')
    parser.add_argument('--valid_data', type=str, required=True, help='验证集label文件，每行一条json格式数据')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # GPU or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if str(args.gpu) == "-1":
        logging.info(f"Use device: CPU.")
    elif str(args.gpu) == "0":
        logging.info(f"Use device: GPU {str(args.gpu)}.")
    else:
        raise ValueError(f"\"--gpu\" must in [-1, 0], while input is {str(args.gpu)}")

    # Set random seed
    torch.manual_seed(777)

    # 读取 configs.yaml
    with open(args.config, 'r', encoding='utf-8') as r1:
        configs = yaml.load(r1, Loader=yaml.FullLoader)

    train_conf = configs['dataset_conf']
    valid_conf = copy.deepcopy(train_conf)
    valid_conf['shuffle'] = False

    # 数据集：
    train_dataset = image_dataset(
        data_list_file=args.train_data,
        conf=train_conf,
    )
    valid_dataset = image_dataset(
        data_list_file=args.valid_data,
        conf=valid_conf,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=0,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        num_workers=0,
    )

    