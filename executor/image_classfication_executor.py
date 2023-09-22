"""图像分类任务的训练过程；"""

import os
import shutil
import torch


class Executor:

    def __init__(self, trainer_conf: dict, criterion, optimizer, device: str = "gpu"):
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化函数
        self.device = device  # gpu or cpu

        self.max_epochs = trainer_conf["epochs"]
        self.ckpt_path = trainer_conf["ckpt_path"]
        self.max_ckpt_save = trainer_conf["max_ckpt_save"]

    def save_ckpt(self, model, epoch):
        """存储ckpt，并定期删除多余的；"""
        # 模型权重存储在这里
        model_path = os.path.join(self.ckpt_path, 'model_epoch-{:04d}.pth'.format(epoch))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, model_path
        )
        # 优化器参数存储在这里
        states_path = os.path.join(self.ckpt_path, 'states_epoch-{:04d}.pth'.format(epoch))
        torch.save(
            {
                'epoch': epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, states_path
        )
        # 删除过多的ckpt文件
        ckpt_list = []
        for file in os.listdir(self.ckpt_path):
            if not file.startswith("model_epoch-"):
                continue
            epoch = int(file.lstrip("model_epoch-").rstrip(".pth"))
            full_path_1 = os.path.join(self.ckpt_path, file)
            full_path_2 = os.path.join(self.ckpt_path, file.replace("model_epoch-", "states_epoch-"))
            ckpt_list.append([epoch, full_path_1, full_path_2])

        ckpt_list.sort(reverse=True)

        for _, path_1, path_2 in ckpt_list[self.max_ckpt_save:]:
            os.remove(path_1)
            os.remove(path_2)

    def load_ckpt_auto(self, model):
        """训练开始之前，找找有没有最近的ckpt，自动加载；"""
        # TODO

    def run(self, model, train_data_loader, valid_data_loader):
        # 正式开始训练
        for epoch in range(self.max_epochs):

            # 训练一个 epoch
            self.run_one_epoch("train", model, train_data_loader, epoch)

            # save ckpt
            self.save_ckpt(model, epoch)

            # eval
            self.run_one_epoch("valid", model, valid_data_loader, epoch)

    def run_one_epoch(self, train_or_valid, model, data_loader, epoch):
        """ 训练一个 epoch """

        if train_or_valid == "train":
            model.train()
        else:
            model.eval()

        correct_ids = 0
        total_ids = 0
        train_loss = 0.0

        for batch_idx, batch in enumerate(data_loader):

            images = batch["image"].to(self.device)
            labels = batch["label_one_hot"].to(self.device)
            labels_id = batch["label_id"].to(self.device)

            # 前向计算
            pred = model(images)
            loss = self.criterion(pred, labels)

            # 反向传播
            if train_or_valid == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 计算 accuracy
            x, pred_id = pred.max(1)
            correct_ids += pred_id.eq(labels_id).sum().item()
            total_ids += labels.size(0)
            train_loss += loss.item()

            # 展示日志
            if batch_idx % 1 == 0:
                print(f"{train_or_valid}: epoch[{epoch}], steps[{batch_idx}]: loss = {loss.item()}")

        # end of epoch
        train_accuracy = 100. * correct_ids / total_ids
        train_loss = 100. * train_loss / total_ids
        print(f"epoch end"
              f"\n"
              f"{train_or_valid}: epoch[{epoch}]: total_loss = {train_loss}, total_accuracy = {train_accuracy}\n")
