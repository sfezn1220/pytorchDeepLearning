""" 定义基础的训练过程；与任务无关； """

import os
import yaml
import shutil
import torch
import logging


class BaseExecutor:

    def __init__(self, conf_file: str, name: str = ""):
        """ 定义基础的训练过程；与任务无关； """

        # 确定配置文件
        with open(conf_file, 'r', encoding='utf-8') as r1:
            self.trainer_conf = yaml.load(r1, Loader=yaml.FullLoader)

        # GPU or CPU
        gpu = str(self.trainer_conf["gpu"])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        if gpu == "-1":
            self.device = "cpu"
            logging.info(f"Use device: CPU.")
        elif gpu == "0":
            self.device = "cuda"
            logging.info(f"Use device: GPU {gpu}.")
        else:
            raise ValueError(f"\"--gpu\" must in [-1, 0], while input is {gpu}")

        # Set random seed
        torch.manual_seed(777)

        self.model = None
        self.train_data_loader = None
        self.valid_data_loader = None

        self.criterion = None  # 损失函数
        self.optimizer = None  # 优化函数

        self.max_epochs = self.trainer_conf["epochs"]
        self.ckpt_path = self.trainer_conf["ckpt_path"]
        self.max_ckpt_save = self.trainer_conf["max_ckpt_save"]

        self.epoch = -1

        self.log_every_steps = self.trainer_conf["log_every_steps"]  # 每多少个step展示一次日志

        self.name = name + "-" if len(name) > 0 else ""

    def write_training_log(self, logs: str, mode: str = "a"):
        """ 记录下日志 """
        log_file = os.path.join(self.ckpt_path, self.name + "training.log")

        with open(log_file, mode, encoding='utf-8') as w1:
            w1.write(logs + "\n")

    def save_ckpt(self):
        """ 存储ckpt，并定期删除多余的； """
        print(f"saving checkpoint of epoch {self.epoch}")

        # 模型权重存储在这里
        model_path = os.path.join(self.ckpt_path, self.name + 'model_epoch-{:04d}.pth'.format(self.epoch))
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
            }, model_path
        )
        # 优化器参数存储在这里
        states_path = os.path.join(self.ckpt_path, self.name + 'states_epoch-{:04d}.pth'.format(self.epoch))
        torch.save(
            {
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, states_path
        )

        # 找出过多的ckpt文件
        ckpt_dic = {}
        for file in os.listdir(self.ckpt_path):
            if file.endswith(".log"):
                continue
            if not file.startswith(self.name):
                continue
            epoch = int(file.split(".")[0].split("-")[-1])
            if epoch not in ckpt_dic:
                ckpt_dic[epoch] = []
            ckpt_dic[epoch].append(file)
        # 排序
        ckpt_list = []
        for epoch in ckpt_dic:
            ckpt_list.append([epoch] + ckpt_dic[epoch])
        ckpt_list.sort(reverse=True)

        # 正式删除过多的ckpt文件
        for item in ckpt_list[self.max_ckpt_save:]:
            for file in item[1:]:
                file = str(file)
                full_path = os.path.join(self.ckpt_path, file)  # 绝对路径，可能是多余的权重文件、中间结果等；
                if os.path.exists(full_path):
                    if os.path.isfile(full_path):  # 如果是文件，直接删除
                        os.remove(full_path)
                    elif os.path.isdir(full_path):  # 如果是文件夹，先删除里面的文件，再删除文件夹本身；
                        for f in os.listdir(full_path):
                            os.remove(os.path.join(full_path, f))
                        shutil.rmtree(full_path)
                    else:
                        print(f"Skip cannot-deleting file: {full_path}")
                else:
                    print(f"Skip deleting non-existing file: {full_path}")

    def load_ckpt_auto(self):
        """训练开始之前，找找有没有最近的ckpt，自动加载；"""
        os.makedirs(self.ckpt_path, exist_ok=True)

        # 找到最近的ckpt
        ckpt_list = []
        for file in os.listdir(self.ckpt_path):
            if not file.startswith(self.name + "model_epoch-"):
                continue
            epoch = int(file.lstrip(self.name + "model_epoch-").rstrip(".pth"))
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
            self.model.load_state_dict(ckpt_dict["model_state_dict"])

            print(f"load ckpt: {os.path.basename(last_ckpt_path)}")
            return last_epoch, self.model

        else:
            return -1, None

    def run(self):

        # 尝试加载预训练模型
        last_epoch = -1
        last_epoch_may, model_may = self.load_ckpt_auto()
        if model_may is not None:
            self.model = model_may
            last_epoch = last_epoch_may

        # 写入日志
        if last_epoch == -1:
            self.write_training_log("start training...", "w")

        # 正式开始训练
        for epoch in range(last_epoch+1, self.max_epochs):
            self.epoch = epoch

            # 训练一个 epoch
            self.train_data_loader.dataset.set_epoch(self.epoch)
            self.train_one_epoch()

            # save ckpt
            self.save_ckpt()

            # eval
            self.valid_data_loader.dataset.set_epoch(self.epoch)
            self.valid_one_epoch()

    def train_one_epoch(self):
        """ 训练一个 epoch；根据任务定义 """
        pass

    def valid_one_epoch(self):
        """ 验证一个 epoch；根据任务定义 """
        pass

    def test(self):
        """ 测试；根据任务定义 """
        pass
