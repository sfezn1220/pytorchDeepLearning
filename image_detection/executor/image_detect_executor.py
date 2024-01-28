"""图像检测任务的训练过程；"""
import numpy as np
import os
import cv2
import shutil
import torch
import torch.nn as nn
import time
import copy
from contextlib import nullcontext

from bin.base_executor import BaseExecutor
from image_detection.dataset import get_image_dataloader

from image_detection.models import YOLOv3


class ImageDetectExecutor(BaseExecutor):

    def __init__(self, conf_file: str, name: str = ""):
        super().__init__(conf_file, name)

        print(f"self.device = {self.device}")
        print(f"self.name = {self.name}")

        # 读取 configs.yaml
        train_data_conf = copy.deepcopy(self.trainer_conf)
        valid_data_conf = copy.deepcopy(self.trainer_conf)

        # 数据集：
        self.train_data_loader = get_image_dataloader(
            data_dir=train_data_conf["train_data"],
            data_conf=train_data_conf,
            data_type="train",
        )
        self.valid_data_loader = get_image_dataloader(
            data_dir=valid_data_conf["valid_data"],
            data_conf=valid_data_conf,
            data_type="valid",
        )
        self.test_data_loader = get_image_dataloader(
            data_dir=valid_data_conf["test_data"],
            data_conf=valid_data_conf,
            data_type="test",
        )

        self.max_steps = self.max_epochs * len(self.train_data_loader)

        # 模型
        conf_basename = os.path.basename(conf_file)
        if conf_basename.startswith("yolov3"):
            self.model = YOLOv3(self.trainer_conf).to(self.device)
            print("Use yolov3 model.")
        else:
            raise ValueError(f"model name ERROR.")
        # print(self.model)

        # 统计参数量
        total_params_cnt = sum(p.numel() for p in self.model.parameters())
        print("模型参数量总计：%.2fM" % (total_params_cnt/1e6))

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.trainer_conf["lr"]))
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps,
            eta_min=float(self.trainer_conf['final_lr']),
            last_epoch=-1,
        )

        # 测试数据的保存路径
        self.test_save_dir = lambda flag, epoch:os.path.join(self.ckpt_path, f"{flag}_epoch-{epoch}")

        # 加载预训练模型
        self.init_pretrain_model()

    def run(self):
        super().run()

    def forward_one_epoch(self, forward_type="train"):
        """ 训练 or 验证一个 epoch； """

        if forward_type.lower() in ["train"]:  # 模型训练
            self.model.train()
            flag = "train"
            tag = nullcontext
            data_loader = self.train_data_loader
        elif forward_type.lower() in ["valid"]:  # 模型验证
            self.model.eval()
            flag = "valid"
            tag = torch.no_grad
            data_loader = self.valid_data_loader
        elif forward_type.lower() in ["test"]:  # 模型测试
            self.model.eval()
            flag = "test"
            tag = torch.no_grad
            data_loader = self.test_data_loader
        else:
            raise ValueError(f'forward_type must in ["train", "valid", "test"].')

        with tag():
            correct_ids = 0
            total_ids = 0
            epoch_total_loss = torch.tensor([0.0]).to(self.device)

            batch_per_epoch = len(data_loader)
            log_every_steps = min(batch_per_epoch, self.log_every_steps)

            st = time.time()
            for batch_idx, batch in enumerate(data_loader):

                global_steps = self.cur_epoch * batch_per_epoch + batch_idx

                images = batch["image"].to(self.device)

                # 前向计算
                pred = self.model(images)
                loss = self.criterion(pred)

                # 反向传播
                if forward_type.lower() in ["train"]:  # 模型训练
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                # 计算 accuracy

                if forward_type.lower() in ["test"]:  # 模型测试
                    pass

                if forward_type.lower() in ["train", "valid"]:  # 模型训练、验证
                    self.tensorboard_writer.add_scalar(f"{flag}/{flag}_loss", loss.item(), global_steps)

                # 记录学习率
                if forward_type.lower() in ["train"]:
                    self.tensorboard_writer.add_scalar(f"learning_rate/learning_rate", self.lr_scheduler.get_last_lr(),
                                                       global_step=global_steps)

                # 展示日志
                if batch_idx % log_every_steps == 0:
                    log = (f"{flag}: epoch[{self.cur_epoch}], steps[{batch_idx}/{batch_per_epoch}]: "
                           f"total_loss = {round(loss.item(), 3)} ."
                           f"")
                    print(log)
                    self.write_training_log(log, "a")

        # end of epoch
        train_accuracy = 100. * correct_ids / total_ids
        epoch_total_loss /= batch_per_epoch
        et = time.time()
        log = (f"{flag} epoch end, {round((et - st)/60, 2)} minutes,"
               f"\n"
               f"epoch[{self.cur_epoch}]: "
               f"total_loss = {round(epoch_total_loss.item(), 3)}, "
               f"total_accuracy = {round(train_accuracy, 2)}%. "
               f"\n")
        print(log)
        self.write_training_log(log, "a")
