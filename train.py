""" 训练、测试的主函数； """

import os
import torch
import copy
import yaml
import logging
import torch.nn as nn

from dataset import get_image_dataloader
from models import VGG16
from executor import Executor


def train(model="vgg_" + "new21"):
    """训练的代码"""
    # config 文件
    conf_file = f"configs\\{model}.yaml"
    with open(conf_file, 'r', encoding='utf-8') as r1:
        configs = yaml.load(r1, Loader=yaml.FullLoader)

    # GPU or CPU
    gpu = str(configs["gpu"])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if gpu == "-1":
        device = "cpu"
        logging.info(f"Use device: CPU.")
    elif gpu == "0":
        device = "cuda"
        logging.info(f"Use device: GPU {gpu}.")
    else:
        raise ValueError(f"\"--gpu\" must in [-1, 0], while input is {gpu}")

    # Set random seed
    torch.manual_seed(777)

    # 读取 configs.yaml
    train_data_conf = configs
    valid_data_conf = copy.deepcopy(train_data_conf)
    valid_data_conf['shuffle'] = False
    valid_data_conf['aug_horizontal_flip'] = False
    valid_data_conf['aug_vertical_flip'] = False
    valid_data_conf['aug_pad'] = False
    valid_data_conf['aug_rotation'] = False
    valid_data_conf['aug_GaussianBlur'] = False
    valid_data_conf['aug_ColorJitter'] = False

    # 数据集：
    train_data_loader = get_image_dataloader(
        data_path=train_data_conf["train_data"],
        data_conf=train_data_conf,
    )
    valid_data_loader = get_image_dataloader(
        data_path=valid_data_conf["valid_data"],
        data_conf=valid_data_conf,
    )

    # 模型
    model = VGG16(configs).to(device)
    print(model)

    # 损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(configs["lr"]))

    # 正式训练
    executor = Executor(
        trainer_conf=configs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    executor.run(
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
    )


def test(model):
    """测试的代码"""
    # config 文件
    conf_file = f"configs\\{model}.yaml"
    with open(conf_file, 'r', encoding='utf-8') as r1:
        configs = yaml.load(r1, Loader=yaml.FullLoader)

    # GPU or CPU
    gpu = str(configs["gpu"])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if gpu == "-1":
        device = "cpu"
        logging.info(f"Use device: CPU.")
    elif gpu == "0":
        device = "cuda"
        logging.info(f"Use device: GPU {gpu}.")
    else:
        raise ValueError(f"\"--gpu\" must in [-1, 0], while input is {gpu}")

    test_data_conf = copy.deepcopy(configs)
    test_data_conf['log_every_steps'] = 1
    test_data_conf['shuffle'] = False
    test_data_conf['aug_horizontal_flip'] = False
    test_data_conf['aug_vertical_flip'] = False
    test_data_conf['aug_pad'] = False
    test_data_conf['aug_rotation'] = False
    test_data_conf['aug_GaussianBlur'] = False
    test_data_conf['aug_ColorJitter'] = False

    # 测试数据：
    test_data_loader = get_image_dataloader(
        data_path=test_data_conf["test_data"],
        data_conf=test_data_conf,
    )

    # 模型
    model = VGG16(configs).to(device)
    print(model)

    # 损失函数、优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=float(configs["lr"]))

    # 正式测试
    executor = Executor(
        trainer_conf=test_data_conf,
        criterion=None,
        optimizer=optimizer,
        device=device,
    )
    executor.test(
        model=model,
        data_loader=test_data_loader,
    )

    return


if __name__ == "__main__":
    # train(model="vgg_" + "new22")

    test(model="vgg_" + "new21")
