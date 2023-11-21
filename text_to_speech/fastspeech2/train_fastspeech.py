""" 训练、测试 FastSpeech2 声学模型的主函数； """

import os
import torch
import copy
import yaml
import logging
import torch.nn as nn

from text_to_speech.fastspeech2.fastspeech_dataset import get_tts_dataloader
from text_to_speech.fastspeech2.fastspeech_executor import FastSpeechExecutor
from text_to_speech.fastspeech2.fastspeech_models import FastSpeech2


def train(conf_file: str):
    """训练的代码"""
    # config 文件
    # conf_file = f"configs\\tts_fs+hifi\\{model}.yaml"
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
    train_data_loader = get_tts_dataloader(
        data_path=train_data_conf["train_data"],
        data_conf=train_data_conf,
    )
    valid_data_loader = get_tts_dataloader(
        data_path=valid_data_conf["valid_data"],
        data_conf=valid_data_conf,
    )

    # 模型
    model = FastSpeech2(configs).to(device)
    # print(model)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=float(configs["lr"]))

    # 正式训练
    executor = FastSpeechExecutor(
        trainer_conf=configs,
        criterion=None,
        optimizer=optimizer,
        device=device,
        name="fastspeech2",
    )

    executor.run(
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
    )


if __name__ == "__main__":
    # 最多使用90%的显存；需要设置一下，要不显存使用过多，会强制重启windows
    torch.cuda.set_per_process_memory_fraction(0.93, 0)

    train(conf_file=f"../examples/Yuanshen/configs/fs+hifi/demo.yaml")
