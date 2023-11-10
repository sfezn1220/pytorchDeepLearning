"""语音合成任务的训练过程；"""

import os
import shutil
import torch
import torch.nn as nn
import time
from .image_classfication_executor import Executor as BaseExecutor


class Executor(BaseExecutor):

    def __init__(self, trainer_conf: dict, criterion, optimizer, device: str = "gpu"):
        super().__init__(trainer_conf, None, optimizer, device)

    @staticmethod
    def calculate_1d_loss(gt, predict, loss_func="MSE"):
        """
        计算两个一维向量之间的 loss
        :param gt: [batch, time1] or [batch, 1, time1]
        :param predict: [batch, time2] or [batch, 1, time2]
        :param loss_func: MSE or MAE
        :return: float32
        """
        if len(gt.shape) == 3:
            gt = gt.squeeze(1)  # [batch, time]
        if len(predict.shape) == 3:
            predict = predict.squeeze(1)  # [batch, time]

        if gt.shape[-1] > predict.shape[-1]:
            gt = gt[:, :predict.shape[-1]]
        elif gt.shape[-1] < predict.shape[-1]:
            predict = predict[:, :gt.shape[-1]]

        if loss_func == "MSE":
            return nn.MSELoss()(predict, gt)
        elif loss_func == "MAE":
            return nn.L1Loss()(predict, gt)
        else:
            raise ValueError(f"loss_function must in ['MSE', 'MAE']")

    @staticmethod
    def calculate_2d_loss(gt, predict, loss_func="MSE"):
        """
        计算两个二维向量之间的 loss
        :param gt: [batch, channel, time1]
        :param predict: [batch, channel, time1]
        :param loss_func: MSE or MAE
        :return: float32
        """
        if gt.shape[-1] > predict.shape[-1]:
            gt = gt[:, :, :predict.shape[-1]]
        elif gt.shape[-1] < predict.shape[-1]:
            predict = predict[:, :, :gt.shape[-1]]

        if loss_func == "MSE":
            return nn.MSELoss()(predict, gt)
        elif loss_func == "MAE":
            return nn.L1Loss()(predict, gt)
        else:
            raise ValueError(f"loss_function must in ['MSE', 'MAE']")

    def run(self, model, train_data_loader, valid_data_loader):
        super().run(model, train_data_loader, valid_data_loader)

    def train_one_epoch(self, model, data_loader, epoch):
        """ 训练一个 epoch """

        model.train()

        epoch_total_loss = 0.0
        epoch_f0_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_dur_loss = 0.0
        epoch_mel_before_loss = 0.0
        epoch_mel_after_loss = 0.0

        batch_per_epoch = len(data_loader)
        log_every_steps = min(batch_per_epoch, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(data_loader):

            phoneme_ids = batch["phoneme_ids"].to(self.device)
            spk_id = batch["spk_id"].to(self.device)
            duration_gt = batch["duration"].to(self.device)
            mel_gt = batch["mel"].to(self.device)
            f0_gt = batch["f0"].to(self.device)
            energy_gt = batch["energy"].to(self.device)
            mel_length = batch["mel_length"].to(self.device)
            f0_length = batch["f0_length"].to(self.device)
            energy_length = batch["energy_length"].to(self.device)

            # 前向计算
            audio, mel_after, mel_before, f0_predict, energy_predict, duration_predict = model(phoneme_ids, spk_id, duration_gt, f0_gt, energy_gt, mel_length, f0_length, energy_length)

            # loss
            f0_loss = self.calculate_1d_loss(f0_gt, f0_predict, "MSE")
            energy_loss = self.calculate_1d_loss(energy_gt, energy_predict, "MSE")

            duration_gt = torch.log(duration_gt + 1)
            duration_loss = self.calculate_1d_loss(duration_gt, duration_predict, "MSE")

            mel_gt = mel_gt.transpose(1, 2)
            mel_before_loss = self.calculate_2d_loss(mel_gt, mel_before, "MAE")
            mel_after_loss = self.calculate_2d_loss(mel_gt, mel_after, "MAE")

            total_loss = f0_loss + energy_loss + duration_loss + mel_before_loss + mel_after_loss

            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_total_loss += total_loss.item()
            epoch_f0_loss += f0_loss.item()
            epoch_energy_loss += energy_loss.item()
            epoch_dur_loss += duration_loss.item()
            epoch_mel_before_loss += mel_before_loss.item()
            epoch_mel_after_loss += mel_after_loss.item()

            # 展示日志
            if batch_idx % log_every_steps == 0:
                log = (f"train: epoch[{epoch}], steps[{batch_idx}/{batch_per_epoch}]: "
                       f"total_loss = {round(total_loss.item(), 2)}, "
                       f"f0_loss = {round(f0_loss.item(), 2)}, "
                       f"energy_loss = {round(energy_loss.item(), 2)}, "
                       f"dur_loss = {round(duration_loss.item(), 2)}, "
                       f"mel_before_loss = {round(mel_before_loss.item(), 2)}, "
                       f"mel_after_loss = {round(mel_after_loss.item(), 2)}.")
                print(log)
                self.write_training_log(log, "a")

        # end of epoch
        epoch_total_loss /= batch_per_epoch
        epoch_f0_loss /= batch_per_epoch
        epoch_energy_loss /= batch_per_epoch
        epoch_dur_loss /= batch_per_epoch
        epoch_mel_before_loss /= batch_per_epoch
        epoch_mel_after_loss /= batch_per_epoch
        et = time.time()
        log = (f"epoch end, {round((et - st)/60, 2)} minutes,"
               f"\n"
               f"train: epoch[{epoch}]: "
               f"total_loss = {round(epoch_total_loss, 2)}, "
               f"f0_loss = {round(epoch_f0_loss, 2)}, "
               f"energy_loss = {round(epoch_energy_loss, 2)}, "
               f"dur_loss = {round(epoch_dur_loss, 2)}, "
               f"mel_before_loss = {round(epoch_mel_before_loss, 2)}, "
               f"mel_after_loss = {round(epoch_mel_after_loss, 2)}.\n")
        print(log)
        self.write_training_log(log, "a")

    def valid_one_epoch(self, model, data_loader, epoch):
        """ 验证一个 epoch """
        print(f"skip valid")
        pass
