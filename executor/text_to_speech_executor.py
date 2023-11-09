"""语音合成任务的训练过程；"""

import os
import shutil
import torch
import time


class Executor:

    def __init__(self, trainer_conf: dict, device: str = "gpu"):
        self.device = device  # gpu or cpu

    def train_one_epoch(self, model, data_loader, epoch):
        """ 训练一个 epoch """

        model.train()

        batch_per_epoch = len(data_loader)

        st = time.time()
        for batch_idx, batch in enumerate(data_loader):

            phoneme_ids = batch["phoneme_ids"].to(self.device)
            spk_id = batch["spk_id"].to(self.device)
            duration = batch["duration"].to(self.device)
            mel = batch["mel"].to(self.device)
            f0 = batch["f0"].to(self.device)
            energy = batch["energy"].to(self.device)
            mel_length = batch["mel_length"].to(self.device)
            f0_length = batch["f0_length"].to(self.device)
            energy_length = batch["energy_length"].to(self.device)

            # 前向计算
            mel_after, mel_before, f0_predict, energy_predict, duration_predict = model(phoneme_ids, spk_id, duration, f0, energy, mel_length, f0_length, energy_length)

        return
