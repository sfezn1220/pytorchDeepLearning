""" 训练、测试的主函数； """

import os
import torch
import argparse
import copy
import yaml
import logging
import torch.nn as nn

from dataset import get_image_dataloader
from models import VGG16


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
        "lr": 1e-3,
        "epochs": 1000,
    }

    train_data = "G:\\Images\\3.labeled_json_0921.train.txt"
    valid_data = "G:\\Images\\3.labeled_json_0921.valid.txt"

    ckpt_path = "F:\\models_images\\demo"


def main():
    # 超参数
    hp = HyperParameters

    # GPU or CPU
    gpu = str(hp.gpu)
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
        batch_size=64,
        num_workers=0,
    )

    # 模型
    model = VGG16(n_classes=160).to(device)
    print(model)

    # 损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf["lr"])

    # 正式开始训练
    for epoch in range(train_conf["epochs"]):

        model.train()
        correct_ids = 0
        total_ids = 0
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_data_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # 前向计算
            pred = model(images)
            loss = criterion(pred, labels)

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 计算 accuracy
            pred_id = pred.argmax(0)
            correct_ids += pred_id.eq(labels).sum().item()
            total_ids += labels.size(0)
            train_loss += loss.item()

            # 展示日志
            if batch_idx % 10 == 0:
                print(f"epoch[{epoch}], steps[{batch_idx}]: loss = {loss.item()}")

        # end of epoch
        train_accuracy = 100. * correct_ids / total_ids
        train_loss = 100. * train_loss / total_ids
        print(f"\nepoch[{epoch}] end: train_total_loss = {train_loss}, train_accuracy = {train_accuracy}\n")

        # save ckpt
        path = os.path.join(hp.ckpt_path, 'epoch_{:04d}.pth'.format(epoch))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, path
        )

        # eval
        model.eval()
        correct_ids = 0
        total_ids = 0
        valid_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_data_loader):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                # 前向计算
                pred = model(images)
                loss = criterion(pred, labels)

                # 计算 accuracy
                pred_id = pred.argmax(0)
                correct_ids += pred_id.eq(labels).sum().item()
                total_ids += labels.size(0)
                valid_loss += loss.item()

        valid_accuracy = 100. * correct_ids / total_ids
        valid_loss = 100. * valid_loss / total_ids
        print(f"\nvalid: valid_total_loss = {valid_loss}, valid_accuracy = {valid_accuracy}\n")


if __name__ == "__main__":
    main()
