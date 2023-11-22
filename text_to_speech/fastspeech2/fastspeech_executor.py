"""语音合成任务的训练过程；"""

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
from text_to_speech.fastspeech2.fastspeech_models import FastSpeech2
from text_to_speech.fastspeech2.fastspeech_dataset import get_tts_dataloader

from text_to_speech.loss import calculate_1d_loss, calculate_2d_loss


class FastSpeechExecutor(BaseExecutor):

    def __init__(self, conf_file: str, name: str = "fastspeech2"):
        super().__init__(conf_file, name)

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
        self.model = FastSpeech2(self.trainer_conf).to(self.device)
        # print(model)

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.trainer_conf["lr"]))

    def save_mel_images(self, mel_gt, mel_before, mel_after, epoch, uttids):
        """ 保存 Mel谱的图片，用于对比； """

        mel_gt = mel_gt[:, :, :mel_after.shape[-1]]

        mel_gt = mel_gt.transpose(1, 2).detach().cpu().numpy()  # -> [batch, time, channel=80]
        mel_before = mel_before.transpose(1, 2).detach().cpu().numpy()  # -> [batch, time, channel=80]
        mel_after = mel_after.transpose(1, 2).detach().cpu().numpy()  # -> [batch, time, channel=80]
        uttids = uttids  # [batch, ]

        # Mel谱保存在这里
        save_dir = os.path.join(self.ckpt_path, self.name + 'predict_epoch-{:04d}'.format(epoch))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # 依次保存batch内的每张Mel谱
        for mel_gt_i, mel_before_i, mel_after_i, uttids_i in zip(mel_gt, mel_before, mel_after, uttids):
            image_path = os.path.join(save_dir, str(uttids_i) + ".png")

            # TODO 仔细修改下下面的代码：
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt_i), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title("Predicted Mel-before-Spectrogram")
            im = ax2.imshow(np.rot90(mel_before_i), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title("Predicted Mel-after-Spectrogram")
            im = ax3.imshow(np.rot90(mel_after_i), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()

    def run(self):
        super().run()

    def train_one_epoch(self):
        """ 训练一个 epoch """

        self.model.train()
        flag = "train"

        epoch_total_loss = 0.0
        epoch_f0_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_dur_loss = 0.0
        epoch_mel_before_loss = 0.0
        epoch_mel_after_loss = 0.0

        batch_per_epoch = len(self.train_data_loader)
        log_every_steps = min(batch_per_epoch, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(self.train_data_loader):

            uttids = batch["uttid"]
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
            mel_after, mel_before, f0_predict, energy_predict, duration_predict \
                = self.model(phoneme_ids, spk_id, duration_gt, f0_gt, energy_gt, mel_length, f0_length, energy_length)

            # loss
            f0_loss = calculate_1d_loss(f0_gt, f0_predict, "MSE")
            energy_loss = calculate_1d_loss(energy_gt, energy_predict, "MSE")

            duration_gt = torch.log(duration_gt + 1)
            duration_loss = calculate_1d_loss(duration_gt, duration_predict, "MSE")

            mel_gt = mel_gt.transpose(1, 2)
            mel_before_loss = calculate_2d_loss(mel_gt, mel_before, "MAE")
            mel_after_loss = calculate_2d_loss(mel_gt, mel_after, "MAE")

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
                log = (f"{flag}: epoch[{self.epoch}], steps[{batch_idx}/{batch_per_epoch}]: "
                       f"total_loss = {round(total_loss.item(), 3)}, "
                       f"f0_loss = {round(f0_loss.item(), 3)}, "
                       f"energy_loss = {round(energy_loss.item(), 3)}, "
                       f"dur_loss = {round(duration_loss.item(), 3)}, "
                       f"mel_before_loss = {round(mel_before_loss.item(), 3)}, "
                       f"mel_after_loss = {round(mel_after_loss.item(), 3)}.")
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
        log = (f"{flag} epoch end, {round((et - st)/60, 2)} minutes,"
               f"\n"
               f"epoch[{self.epoch}]: "
               f"total_loss = {round(epoch_total_loss, 3)}, "
               f"f0_loss = {round(epoch_f0_loss, 3)}, "
               f"energy_loss = {round(epoch_energy_loss, 3)}, "
               f"dur_loss = {round(epoch_dur_loss, 3)}, "
               f"mel_before_loss = {round(epoch_mel_before_loss, 3)}, "
               f"mel_after_loss = {round(epoch_mel_after_loss, 3)}.\n")
        print(log)
        self.write_training_log(log, "a")

    def valid_one_epoch(self):
        """ 验证一个 epoch """

        self.model.eval()
        flag = "valid"

        epoch_total_loss = 0.0
        epoch_f0_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_dur_loss = 0.0
        epoch_mel_before_loss = 0.0
        epoch_mel_after_loss = 0.0

        batch_per_epoch = len(self.valid_data_loader)
        log_every_steps = min(batch_per_epoch, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(self.valid_data_loader):

            uttids = batch["uttid"]
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
            mel_after, mel_before, f0_predict, energy_predict, duration_predict \
                = self.model(phoneme_ids, spk_id, duration_gt, f0_gt, energy_gt, mel_length, f0_length, energy_length)

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
            # total_loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()

            epoch_total_loss += total_loss.item()
            epoch_f0_loss += f0_loss.item()
            epoch_energy_loss += energy_loss.item()
            epoch_dur_loss += duration_loss.item()
            epoch_mel_before_loss += mel_before_loss.item()
            epoch_mel_after_loss += mel_after_loss.item()

            # 展示日志
            if batch_idx % log_every_steps == 0:
                log = (f"{flag}: epoch[{self.epoch}], steps[{batch_idx}/{batch_per_epoch}]: "
                       f"total_loss = {round(total_loss.item(), 3)}, "
                       f"f0_loss = {round(f0_loss.item(), 3)}, "
                       f"energy_loss = {round(energy_loss.item(), 3)}, "
                       f"dur_loss = {round(duration_loss.item(), 3)}, "
                       f"mel_before_loss = {round(mel_before_loss.item(), 3)}, "
                       f"mel_after_loss = {round(mel_after_loss.item(), 3)}.")
                print(log)
                self.write_training_log(log, "a")

                # 保存验证集的 Mel谱，用于对比；不用存太多；
                self.save_mel_images(mel_gt, mel_before, mel_after, self.epoch, uttids)

        # end of epoch
        epoch_total_loss /= batch_per_epoch
        epoch_f0_loss /= batch_per_epoch
        epoch_energy_loss /= batch_per_epoch
        epoch_dur_loss /= batch_per_epoch
        epoch_mel_before_loss /= batch_per_epoch
        epoch_mel_after_loss /= batch_per_epoch
        et = time.time()
        log = (f"{flag} epoch end, {round((et - st)/60, 2)} minutes,"
               f"\n"
               f"epoch[{self.epoch}]: "
               f"total_loss = {round(epoch_total_loss, 3)}, "
               f"f0_loss = {round(epoch_f0_loss, 3)}, "
               f"energy_loss = {round(epoch_energy_loss, 3)}, "
               f"dur_loss = {round(epoch_dur_loss, 3)}, "
               f"mel_before_loss = {round(epoch_mel_before_loss, 3)}, "
               f"mel_after_loss = {round(epoch_mel_after_loss, 3)}.\n")
        print(log)
        self.write_training_log(log, "a")

    def gen_mel_spec(self, dir_name="gen_mel_spec"):
        """ 训练一个 epoch """

        # 尝试加载预训练模型
        last_epoch = -1
        last_epoch_may, model_may = self.load_ckpt_auto()
        if model_may is not None:
            self.model = model_may
            last_epoch = last_epoch_may

        # Mel谱的存储位置：
        mel_save_dir = os.path.join(self.ckpt_path, dir_name + '-epoch-{:04d}'.format(last_epoch))
        if not os.path.exists(mel_save_dir):
            os.mkdir(mel_save_dir)

        self.model.eval()

        for batch in tqdm.tqdm(self.train_data_loader):

            uttids = batch["uttid"]
            phoneme_ids = batch["phoneme_ids"].to(self.device)
            spk_id = batch["spk_id"].to(self.device)
            duration_gt = batch["duration"].to(self.device)
            mel_gt = batch["mel"].to(self.device)
            f0_gt = batch["f0"].to(self.device)
            energy_gt = batch["energy"].to(self.device)
            mel_length = batch["mel_length"].to(self.device)
            f0_length = batch["f0_length"].to(self.device)
            energy_length = batch["energy_length"].to(self.device)

            # 前向推理
            mel_after, mel_before, f0_predict, energy_predict, duration_predict \
                = self.model(phoneme_ids, spk_id, duration_gt, f0_gt, energy_gt, mel_length, f0_length, energy_length)

            # 存储Mel谱
            mel_after = mel_after.detach().squeeze(0).cpu().numpy()  # -> [channel=80， time]

            for mel_after_i, uttids_i in zip(mel_after, uttids):
                mel_save_path = os.path.join(mel_save_dir, str(uttids_i) + ".npy")
                np.save(file=mel_save_path, arr=mel_after_i)
