""" HiFiGAN声码器的训练过程； """

import os
import shutil
import copy
import torch
import torch.nn as nn
import time
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from bin.base_executor import BaseExecutor
from text_to_speech.fastspeech2.fastspeech_dataset import get_tts_dataloader
from text_to_speech.hifigan.hifigan import HiFiGAN

from text_to_speech.loss import calculate_1d_loss, calculate_2d_loss


class HiFiGANExecutor(BaseExecutor):

    def __init__(self, conf_file: str, name: str = "hifigan"):
        super().__init__(conf_file, name)

        print(f"self.device = {self.device}")

        # 读取 configs.yaml
        train_data_conf = copy.deepcopy(self.trainer_conf)
        valid_data_conf = copy.deepcopy(self.trainer_conf)
        # valid_data_conf['shuffle'] = False

        # 数据集：
        self.train_data_loader = get_tts_dataloader(
            data_path=train_data_conf["train_data"],
            data_conf=train_data_conf,
        )
        self.valid_data_loader = get_tts_dataloader(
            data_path=valid_data_conf["valid_data"],
            data_conf=valid_data_conf,
        )

        # 模型
        self.model = HiFiGAN(self.trainer_conf, device=self.device).to(self.device)
        # print(model)

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.trainer_conf["lr"]))

    def run(self):
        super().run()

    def train_one_epoch(self):
        """ 训练一个 epoch """

        self.model.train()
        flag = "train"

        epoch_total_loss = 0.0
        epoch_sc_loss = 0.0
        epoch_mag_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_fm_loss = 0.0
        epoch_real_loss = 0.0
        epoch_fake_loss = 0.0

        batch_per_epoch = len(self.train_data_loader)
        log_every_steps = min(batch_per_epoch, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(self.train_data_loader):

            uttids = batch["uttid"]
            spk_id = batch["spk_id"].to(self.device)
            mel_gt = batch["mel"].to(self.device)
            audio_gt = batch['audio'].to(self.device)

            mel_gt = mel_gt.transpose(1, 2)

            # 前向计算
            audio_gen, features_gen = self.model(mel_gt)
            features_gt = self.model.forward_discriminator(audio_gt)

            # loss
            adv_loss = calculate_2d_loss(
                torch.ones_like(features_gen[-1], dtype=features_gen[-1].dtype),
                features_gen[-1],
                "MSE"
            )

            total_loss = adv_loss  # TODO 补充更多 loss

            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_total_loss += total_loss.item()

            # 展示日志
            if batch_idx % log_every_steps == 0:
                log = (f"{flag}: epoch[{self.epoch}], steps[{batch_idx}/{batch_per_epoch}]: "
                       f"total_loss = {round(total_loss.item(), 3)}, "
                       f"")
                print(log)
                self.write_training_log(log, "a")

        # end of epoch
        epoch_total_loss /= batch_per_epoch
        et = time.time()
        log = (f"{flag} epoch end, {round((et - st) / 60, 2)} minutes,"
               f"\n"
               f"epoch[{self.epoch}]: "
               f"total_loss = {round(epoch_total_loss, 3)}, "
               f"\n")
        print(log)
        self.write_training_log(log, "a")