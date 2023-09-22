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


def main():
    # config 文件
    conf_file = "configs\\vgg_base.yaml"
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
    # valid_data_conf['shuffle'] = False

    # 数据集：
    train_data_loader, valid_data_loader = get_image_dataloader(
        train_data=train_data_conf["train_data"],
        valid_data=valid_data_conf["valid_data"],
        train_conf=train_data_conf,
        valid_conf=valid_data_conf,
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


if __name__ == "__main__":
    main()
