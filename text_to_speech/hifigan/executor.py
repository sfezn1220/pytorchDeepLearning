""" HiFiGAN声码器的训练过程； """

import os
import shutil
import copy
import torch
import torch.nn as nn
import time
import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from bin.base_executor import BaseExecutor
from text_to_speech.fastspeech2.fastspeech_dataset import get_tts_dataloader
from text_to_speech.hifigan.hifigan import HiFiGAN
from text_to_speech.hifigan.multi_stft_loss import MultiSTFTLoss

from text_to_speech.loss import calculate_1d_loss, calculate_2d_loss


class HiFiGANExecutor(BaseExecutor):

    def __init__(self, conf_file: str, name: str = "hifigan"):
        super().__init__(conf_file, name)

        print(f"self.device = {self.device}")
        print(f"self.name = {self.name}")

        # 设置 batch_size
        self.trainer_conf["batch_size"] = self.trainer_conf["batch_size_hifigan"]

        # 读取 configs.yaml
        train_data_conf = copy.deepcopy(self.trainer_conf)
        valid_data_conf = copy.deepcopy(self.trainer_conf)
        # valid_data_conf['shuffle'] = False

        self.sample_rate = train_data_conf['sample_rate']

        # 数据集：
        self.train_data_loader = get_tts_dataloader(
            data_path=train_data_conf["train_data"],
            data_conf=train_data_conf,
            model_type="vocoder",
            data_type="train",
        )
        self.valid_data_loader = get_tts_dataloader(
            data_path=valid_data_conf["valid_data"],
            data_conf=valid_data_conf,
            model_type="vocoder",
            data_type="valid",
        )

        self.max_steps = self.max_epochs * len(self.train_data_loader)

        # 模型
        self.model = HiFiGAN(self.trainer_conf, device=self.device).to(self.device)
        self.pretrain_file = self.trainer_conf["pretrain_hifigan_file"]
        # print(model)

        # 第多少个epoch才开始迭代判别器
        self.start_discriminator_epoch = int(self.trainer_conf["start_discriminator_epoch"])

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.trainer_conf["lr"]))
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps,
            eta_min=float(self.trainer_conf['final_lr']),
            last_epoch=-1,
        )

        # stft loss
        self.stft_loss = MultiSTFTLoss()

        # 合成的语音数据的路径
        self.gen_audios_dir_name = self.name + "predict_epoch"

    def save_gen_audios(self, audio_gen, uttids):
        """ 保存合成音频； """
        if len(audio_gen.shape) == 3:
            audio_gen = audio_gen.clone().squeeze(1)  # [batch, 1, time] -> [batch, time]

        audio_gen = audio_gen.detach().cpu().numpy()

        # 合成语音保存在这里
        save_dir = os.path.join(self.ckpt_path, self.gen_audios_dir_name + '-{:04d}'.format(self.cur_epoch))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for audio_gen_i, uttids_i in zip(audio_gen, uttids):
            sf.write(
                file=os.path.join(save_dir, str(uttids_i) + ".wav"),
                data=audio_gen_i,
                samplerate=self.sample_rate,
            )

        return

    def cal_loss(self, audio_gt, audio_gen, features_gt, features_gen):
        """ 计算 HiFiGAN 的损失函数； """
        # sc_loss, mag_loss of generator
        sc_loss, mag_loss = self.stft_loss.cal_loss(audio_gt, audio_gen)

        adv_loss = torch.tensor([0.0]).to(self.device)
        fm_loss = torch.tensor([0.0]).to(self.device)
        real_loss = torch.tensor([0.0]).to(self.device)
        fake_loss = torch.tensor([0.0]).to(self.device)

        # 最开始不更新判别器，后面再更新；
        if self.cur_epoch >= self.start_discriminator_epoch:
            # adv_loss of generator
            cnt = 0
            for i in range(len(features_gen)):
                adv_loss += calculate_2d_loss(
                    torch.ones_like(features_gen[i][-1], dtype=features_gen[i][-1].dtype),
                    features_gen[i][-1],
                    "MSE"
                )
                cnt += 1
            adv_loss /= cnt

            # fm_loss of generator
            cnt = 0
            for i in range(len(features_gen)):
                for j in range(len(features_gen[i])):
                    fm_loss += calculate_2d_loss(
                        features_gt[i][j],
                        features_gen[i][j],
                        "MAE"
                    )
                    cnt += 1
            fm_loss /= cnt

            # real_loss of discriminator
            cnt = 0
            for i in range(len(features_gt)):
                real_loss += calculate_2d_loss(
                    torch.ones_like(features_gt[i][-1], dtype=features_gt[i][-1].dtype),
                    features_gt[i][-1],
                    "MSE"
                )
                cnt += 1
            real_loss /= cnt

            # fake_loss of discriminator
            cnt = 0
            for i in range(len(features_gen)):
                fake_loss += calculate_2d_loss(
                    torch.zeros_like(features_gen[i][-1], dtype=features_gen[i][-1].dtype),
                    features_gen[i][-1],
                    "MSE"
                )
                cnt += 1
            fake_loss /= cnt

        total_loss = 0.5 * sc_loss + 0.5 * mag_loss + 4.0 * adv_loss + 10.0 * fm_loss + real_loss + fake_loss

        return total_loss, sc_loss, mag_loss, adv_loss, fm_loss, real_loss, fake_loss

    def run(self):
        super().run()

    def train_one_epoch(self):
        """ 训练一个 epoch """

        self.model.train()
        flag = "train"
        data_loader = self.train_data_loader

        epoch_total_loss = torch.tensor([0.0]).to(self.device)
        epoch_sc_loss = torch.tensor([0.0]).to(self.device)
        epoch_mag_loss = torch.tensor([0.0]).to(self.device)
        epoch_adv_loss = torch.tensor([0.0]).to(self.device)
        epoch_fm_loss = torch.tensor([0.0]).to(self.device)
        epoch_real_loss = torch.tensor([0.0]).to(self.device)
        epoch_fake_loss = torch.tensor([0.0]).to(self.device)

        batch_per_epoch = len(data_loader)
        log_every_steps = min(batch_per_epoch, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(data_loader):

            global_steps = self.cur_epoch * batch_per_epoch + batch_idx

            uttids = batch["uttid"]
            spk_id = batch["spk_id"].to(self.device)
            mel_gt = batch["mel"].to(self.device)
            audio_gt = batch['audio'].to(self.device)

            # 前向计算
            audio_gen, features_gen = self.model(mel_gt)
            features_gt = self.model.forward_discriminator(audio_gt)

            # loss
            total_loss, sc_loss, mag_loss, adv_loss, fm_loss, real_loss, fake_loss = self.cal_loss(audio_gt, audio_gen, features_gt, features_gen)

            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # end of step
            self.lr_scheduler.step()

            epoch_total_loss += total_loss.item()
            epoch_sc_loss += sc_loss.item()
            epoch_mag_loss += mag_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_fm_loss += fm_loss.item()
            epoch_real_loss += real_loss.item()
            epoch_fake_loss += fake_loss.item()

            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_total_loss", total_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_sc_loss", sc_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_mag_loss", mag_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_adv_loss", adv_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_fm_loss", fm_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_real_loss", real_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_fake_loss", fake_loss.item(), global_step=global_steps)

            self.tensorboard_writer.add_scalar(f"learning_rate/learning_rate", self.lr_scheduler.get_last_lr(),
                                               global_step=global_steps)

            # 展示日志
            if batch_idx % log_every_steps == 0:
                log = (f"{flag}: epoch[{self.cur_epoch}], steps[{batch_idx}/{batch_per_epoch}]: "
                       f"total_loss = {round(total_loss.item(), 3)}, "
                       f"sc_loss = {round(sc_loss.item(), 3)}, "
                       f"mag_loss = {round(mag_loss.item(), 3)}, "
                       f"adv_loss = {round(adv_loss.item(), 3)}, "
                       f"fm_loss = {round(fm_loss.item(), 3)}, "
                       f"real_loss = {round(real_loss.item(), 3)}, "
                       f"fake_loss = {round(fake_loss.item(), 3)}, "
                       f"")
                print(log)
                self.write_training_log(log, "a")

        # end of epoch
        epoch_total_loss /= batch_per_epoch
        epoch_sc_loss /= batch_per_epoch
        epoch_mag_loss /= batch_per_epoch
        epoch_adv_loss /= batch_per_epoch
        epoch_fm_loss /= batch_per_epoch
        epoch_real_loss /= batch_per_epoch
        epoch_fake_loss /= batch_per_epoch
        et = time.time()
        log = (f"{flag} epoch end, {round((et - st) / 60, 2)} minutes,"
               f"\n"
               f"epoch[{self.cur_epoch}]: "
               f"total_loss = {round(epoch_total_loss.item(), 3)}, "
               f"sc_loss = {round(epoch_sc_loss.item(), 3)}, "
               f"mag_loss = {round(epoch_mag_loss.item(), 3)}, "        
               f"adv_loss = {round(epoch_adv_loss.item(), 3)}, "
               f"fm_loss = {round(epoch_fm_loss.item(), 3)}, "
               f"real_loss = {round(epoch_real_loss.item(), 3)}, "
               f"fake_loss = {round(epoch_fake_loss.item(), 3)}, "
               f"\n")
        print(log)
        self.write_training_log(log, "a")

    def valid_one_epoch(self):
        """ 训练一个 epoch """

        self.model.eval()
        flag = "valid"
        data_loader = self.valid_data_loader

        epoch_total_loss = torch.tensor([0.0]).to(self.device)
        epoch_sc_loss = torch.tensor([0.0]).to(self.device)
        epoch_mag_loss = torch.tensor([0.0]).to(self.device)
        epoch_adv_loss = torch.tensor([0.0]).to(self.device)
        epoch_fm_loss = torch.tensor([0.0]).to(self.device)
        epoch_real_loss = torch.tensor([0.0]).to(self.device)
        epoch_fake_loss = torch.tensor([0.0]).to(self.device)

        batch_per_epoch = len(data_loader)
        log_every_steps = min(batch_per_epoch, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(data_loader):

            global_steps = self.cur_epoch * batch_per_epoch + batch_idx

            uttids = batch["uttid"]
            spk_id = batch["spk_id"].to(self.device)
            mel_gt = batch["mel"].to(self.device)
            audio_gt = batch['audio'].to(self.device)

            # 前向计算
            audio_gen, features_gen = self.model(mel_gt)
            features_gt = self.model.forward_discriminator(audio_gt)

            # loss
            total_loss, sc_loss, mag_loss, adv_loss, fm_loss, real_loss, fake_loss = self.cal_loss(audio_gt, audio_gen, features_gt, features_gen)

            # 反向传播
            # total_loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()

            # end of step
            # self.lr_scheduler.step()

            epoch_total_loss += total_loss.item()
            epoch_sc_loss += sc_loss.item()
            epoch_mag_loss += mag_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_fm_loss += fm_loss.item()
            epoch_real_loss += real_loss.item()
            epoch_fake_loss += fake_loss.item()

            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_total_loss", total_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_sc_loss", sc_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_mag_loss", mag_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_adv_loss", adv_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_fm_loss", fm_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_real_loss", real_loss.item(), global_step=global_steps)
            self.tensorboard_writer.add_scalar(f"{flag}/{flag}_fake_loss", fake_loss.item(), global_step=global_steps)

            # self.tensorboard_writer.add_scalar(f"learning_rate/learning_rate", self.lr_scheduler.get_last_lr(),
            #                                    global_step=global_steps)

            # 保存：合成谱
            self.save_gen_audios(audio_gen, uttids)

            # 展示日志
            if batch_idx % log_every_steps == 0:
                log = (f"{flag}: epoch[{self.cur_epoch}], steps[{batch_idx}/{batch_per_epoch}]: "
                       f"total_loss = {round(total_loss.item(), 3)}, "
                       f"sc_loss = {round(sc_loss.item(), 3)}, "
                       f"mag_loss = {round(mag_loss.item(), 3)}, "
                       f"adv_loss = {round(adv_loss.item(), 3)}, "
                       f"fm_loss = {round(fm_loss.item(), 3)}, "
                       f"real_loss = {round(real_loss.item(), 3)}, "
                       f"fake_loss = {round(fake_loss.item(), 3)}, "
                       f"")
                print(log)
                self.write_training_log(log, "a")

        # end of epoch
        epoch_total_loss /= batch_per_epoch
        epoch_sc_loss /= batch_per_epoch
        epoch_mag_loss /= batch_per_epoch
        epoch_adv_loss /= batch_per_epoch
        epoch_fm_loss /= batch_per_epoch
        epoch_real_loss /= batch_per_epoch
        epoch_fake_loss /= batch_per_epoch
        et = time.time()
        log = (f"{flag} epoch end, {round((et - st) / 60, 2)} minutes,"
               f"\n"
               f"epoch[{self.cur_epoch}]: "
               f"total_loss = {round(epoch_total_loss.item(), 3)}, "
               f"sc_loss = {round(epoch_sc_loss.item(), 3)}, "
               f"mag_loss = {round(epoch_mag_loss.item(), 3)}, "        
               f"adv_loss = {round(epoch_adv_loss.item(), 3)}, "
               f"fm_loss = {round(epoch_fm_loss.item(), 3)}, "
               f"real_loss = {round(epoch_real_loss.item(), 3)}, "
               f"fake_loss = {round(epoch_fake_loss.item(), 3)}, "
               f"\n")
        print(log)
        self.write_training_log(log, "a")
