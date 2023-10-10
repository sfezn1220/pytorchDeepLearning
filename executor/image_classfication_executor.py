"""图像分类任务的训练过程；"""

import os
import shutil
import torch
import time


class Executor:

    def __init__(self, trainer_conf: dict, criterion, optimizer, device: str = "gpu"):
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化函数
        self.device = device  # gpu or cpu

        self.max_epochs = trainer_conf["epochs"]
        self.ckpt_path = trainer_conf["ckpt_path"]
        self.max_ckpt_save = trainer_conf["max_ckpt_save"]

        self.log_every_steps = trainer_conf["log_every_steps"]  # 每多少个step展示一次日志

    def write_training_log(self, logs: str, mode: str = "a"):
        """记录下日志"""
        log_file = os.path.join(self.ckpt_path, "training.log")
        with open(log_file, mode, encoding='utf-8') as w1:
            w1.write(logs + "\n")

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
            if "epoch-0049.pth" in file:  # 特例，单独保存下这个ckpt、不删除
                continue
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
        # 找到最近的ckpt
        ckpt_list = []
        for file in os.listdir(self.ckpt_path):
            if not file.startswith("model_epoch-"):
                continue
            epoch = int(file.lstrip("model_epoch-").rstrip(".pth"))
            full_path_1 = os.path.join(self.ckpt_path, file)
            full_path_2 = os.path.join(self.ckpt_path, file.replace("model_epoch-", "states_epoch-"))
            ckpt_list.append([epoch, full_path_1, full_path_2])

        ckpt_list.sort(key=lambda x: x[0], reverse=True)

        if len(ckpt_list) > 0:
            last_ckpt_path = ckpt_list[0][1]
            last_stat_path = ckpt_list[0][2]

            ckpt_dict = torch.load(last_ckpt_path)
            stat_dict = torch.load(last_stat_path)

            last_epoch = int(ckpt_dict["epoch"])
            self.optimizer.load_state_dict(stat_dict['optimizer_state_dict'])
            model.load_state_dict(ckpt_dict["model_state_dict"])

            print(f"load ckpt: {os.path.basename(last_ckpt_path)}")
            return last_epoch, model

        else:
            return -1, None

    def run(self, model, train_data_loader, valid_data_loader):

        # 尝试加载与训练模型
        os.makedirs(self.ckpt_path, exist_ok=True)

        last_epoch = -1
        last_epoch_may, model_may = self.load_ckpt_auto(model)
        if model_may is not None:
            model = model_may
            last_epoch = last_epoch_may

        # 写入日志
        if last_epoch == -1:
            self.write_training_log("start training...", "w")

        # 正式开始训练
        for epoch in range(last_epoch+1, self.max_epochs):

            # 训练一个 epoch
            train_data_loader.dataset.set_epoch(epoch)
            self.train_one_epoch(model, train_data_loader, epoch)

            # save ckpt
            self.save_ckpt(model, epoch)

            # eval
            valid_data_loader.dataset.set_epoch(epoch)
            self.valid_one_epoch(model, valid_data_loader, epoch)

    def train_one_epoch(self, model, data_loader, epoch):
        """ 训练一个 epoch """

        model.train()

        correct_ids = 0
        total_ids = 0
        train_loss = 0.0

        batch_per_epoch = len(data_loader)
        log_every_steps = min(batch_per_epoch-1, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(data_loader):

            images = batch["image"].to(self.device)
            labels = batch["label_one_hot"].to(self.device)
            labels_id = batch["label_id"].to(self.device)

            # 前向计算
            pred = model(images)
            loss = self.criterion(pred, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 计算 accuracy
            x, pred_id = pred.max(1)
            correct_ids += pred_id.eq(labels_id).sum().item()
            total_ids += labels.size(0)
            train_loss += loss.item()

            # 展示日志
            if batch_idx % log_every_steps == 0:
                log = f"train: epoch[{epoch}], steps[{batch_idx}/{batch_per_epoch}]: loss = {round(loss.item(), 2)}"
                print(log)
                self.write_training_log(log, "a")

        # end of epoch
        train_accuracy = 100. * correct_ids / total_ids
        train_loss = train_loss / batch_per_epoch
        et = time.time()
        log = (f"epoch end, {round((et - st)/60, 2)} minutes,"
               f"\n"
               f"train: epoch[{epoch}]: "
               f"total_loss = {round(train_loss, 2)}, total_accuracy = {round(train_accuracy, 2)}%\n")
        print(log)
        self.write_training_log(log, "a")

    def valid_one_epoch(self, model, data_loader, epoch):
        """ 验证 """

        model.eval()

        correct_ids = 0
        total_ids = 0
        train_loss = 0.0

        batch_per_epoch = len(data_loader)
        log_every_steps = min(batch_per_epoch-1, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(data_loader):

            images = batch["image"].to(self.device)
            labels = batch["label_one_hot"].to(self.device)
            labels_id = batch["label_id"].to(self.device)

            # 前向计算
            pred = model(images)
            loss = self.criterion(pred, labels)

            # 计算 accuracy
            x, pred_id = pred.max(1)
            correct_ids += pred_id.eq(labels_id).sum().item()
            total_ids += labels.size(0)
            train_loss += loss.item()

            # 展示日志
            if batch_idx % log_every_steps == 0:
                log = f"valid: epoch[{epoch}], steps[{batch_idx}/{batch_per_epoch}]: loss = {round(loss.item(), 2)}"
                print(log)
                self.write_training_log(log, "a")

        # end of epoch
        train_accuracy = 100. * correct_ids / total_ids
        train_loss = train_loss / batch_per_epoch
        et = time.time()
        log = (f"epoch end, {round((et - st) / 60, 2)} minutes,"
               f"\n"
               f"valid: epoch[{epoch}]: "
               f"total_loss = {round(train_loss, 2)}, total_accuracy = {round(train_accuracy, 2)}%\n")
        print(log)
        self.write_training_log(log, "a")

    def test(self, model, data_loader):
        """测试"""
        # 尝试加载预训练模型
        last_epoch, model_may = self.load_ckpt_auto(model)
        if model_may is not None:
            model = model_may
        else:
            raise ValueError(f"No checkpoint find. Exit.")

        model.eval()

        correct_ids = 0
        total_ids = 0

        batch_per_epoch = len(data_loader)
        log_every_steps = min(batch_per_epoch-1, self.log_every_steps)

        st = time.time()
        for batch_idx, batch in enumerate(data_loader):

            images = batch["image"].to(self.device)
            labels = batch["label_one_hot"].to(self.device)
            labels_id = batch["label_id"].to(self.device)

            # 前向计算
            pred = model(images)

            # 计算 accuracy
            x, pred_id = pred.max(1)
            correct_ids += pred_id.eq(labels_id).sum().item()
            total_ids += labels.size(0)

            # 展示日志
            if batch_idx % log_every_steps == 0:
                log = f"test: steps[{batch_idx}/{batch_per_epoch}]: done"
                print(log)

            # 展示结果
            # self.write_test_result(batch, pred_id, last_epoch)  # TODO
            # 对于每条测试结果：
            for i in range(len(batch)):
                true_i = batch["label_id"][i]
                pred_i = pred_id[i]
                # if true_i == pred_i:
                #     print(f"{batch['path'][i]}")
                #     print(f"true[{i}] = {true_i}, pred[{i}] = {pred_i}\n")

        # end of epoch
        train_accuracy = 100. * correct_ids / total_ids
        et = time.time()
        log = (f"test cost {round((et - st) / 60, 2)} minutes,"
               f"\n"
               f"total_accuracy = {round(train_accuracy, 2)}%\n")
        print(log)

    def write_test_result(self, batch, pred_id, epoch):
        """保存每一个测试用例的测试结果；"""

        # 测试结果的保存路径
        test_save_dir = os.path.join(self.ckpt_path, f"test_epoch-{epoch}")
        if not os.path.exists(test_save_dir):
            os.makedirs(test_save_dir)

        # TODO
