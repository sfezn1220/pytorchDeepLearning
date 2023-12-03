""" 定义基础的训练过程；与任务无关； """

import os
import yaml
import shutil
import torch
import logging

from tensorboardX import SummaryWriter


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

        # 初始化：模型、数据集
        self.model = None
        self.train_data_loader = None
        self.valid_data_loader = None
        self.pretrain_file = None

        # 初始化：损失函数、优化函数、学习率计划、tensorboard
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tensorboard_writer = None

        # 初始化：保存checkpoint的路径、最多同时保存多少个checkpoint
        self.ckpt_path = self.trainer_conf["ckpt_path"]
        self.max_ckpt_save = int(self.trainer_conf["max_ckpt_save"])

        # 初始化：最大epochs数量、最大steps数量、当前epochs（-1表示随便定个初始值）
        self.max_epochs = int(self.trainer_conf["epochs"])
        self.max_steps = -1  # 初始化：随便定个值；
        self.cur_epoch = -1

        # 初始化：每多少个step展示一次日志
        self.log_every_steps = int(self.trainer_conf["log_every_steps"])

        # 初始化：模型的名称、日志名称、
        self.name = name + "-" if len(name) > 0 else ""
        self.log_file_name = self.name + "training.log"
        self.model_name_prefix = self.name + "model_epoch"
        self.states_name_prefix = self.name + "states_epoch"
        self.tensorboard_dir_name = self.name + "logs"

    def write_training_log(self, logs: str, mode: str = "a"):
        """ 记录下日志 """
        assert mode in ["w", "a"]

        log_file = os.path.join(self.ckpt_path, self.log_file_name)
        with open(log_file, mode, encoding='utf-8') as w1:
            w1.write(logs + "\n")

    def save_ckpt(self):
        """ 存储ckpt，并定期删除多余的； """
        print(f"\n"
              f"saving checkpoint of epoch {self.cur_epoch}")

        # 模型权重存储在这里
        model_path = os.path.join(self.ckpt_path, self.model_name_prefix + '-{:04d}.pth'.format(self.cur_epoch))
        torch.save(
            {
                'epoch': self.cur_epoch,
                'model_state_dict': self.model.state_dict(),
            }, model_path
        )
        # 优化器参数存储在这里
        states_path = os.path.join(self.ckpt_path, self.states_name_prefix + '-{:04d}.pth'.format(self.cur_epoch))
        torch.save(
            {
                'epoch': self.cur_epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            }, states_path
        )
        # 自动删除多余的checkpoint
        self.delete_extra_ckpt_auto()

    def delete_extra_ckpt_auto(self):
        """ 在每次保存checkpoint之后，自动删除多余的checkpoint； """
        # 找出过多的ckpt文件
        ckpt_dic = {}
        for file in os.listdir(self.ckpt_path):
            if not file.startswith(self.name):  # 只寻找：前缀和模型名称相同的文件
                continue
            try:
                epoch = int(file.split(".")[0].split("-")[-1])
            except:
                continue
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
            # 逐个删除每个过多的epoch的ckpt文件
            for file_name in item[1:]:
                file_name = str(file_name)
                full_path = os.path.join(self.ckpt_path, file_name)  # 绝对路径，可能是多余的权重文件、中间结果等；
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

        # 先尝试找到最近的ckpt
        ckpt_list = []
        for file in os.listdir(self.ckpt_path):
            if not file.startswith(self.model_name_prefix):
                continue
            epoch = int(file.split(".")[0].split("-")[-1])
            # 模型文件的绝对路径
            model_path = os.path.join(self.ckpt_path, file)
            # 优化前参数文件的绝对路径
            stats_path = os.path.join(self.ckpt_path, file.replace(self.model_name_prefix, self.states_name_prefix))
            if not os.path.exists(stats_path):
                print(f"Can not find stats_path of epoch {str(epoch)}")
                continue
            ckpt_list.append([epoch, model_path, stats_path])

        ckpt_list.sort(key=lambda x: x[0], reverse=True)

        # 如果找到了以往的文件，取最近的
        if len(ckpt_list) > 0:
            last_model_path = ckpt_list[0][1]
            last_stats_path = ckpt_list[0][2]

            model_dict = torch.load(last_model_path)
            stats_dict = torch.load(last_stats_path)

            last_epoch = int(model_dict["epoch"])
            self.model.load_state_dict(model_dict["model_state_dict"])
            self.optimizer.load_state_dict(stats_dict['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(stats_dict['lr_scheduler_state_dict'])

            print(f"load ckpt: {os.path.basename(last_model_path)}")
            return last_epoch

        # 再看看有没有预训练模型：（self.cur_epoch < 0 表示 epoch 0 开始之前）
        elif os.path.exists(self.pretrain_file):
            model_dict = torch.load(self.pretrain_file)
            self.model.load_state_dict(model_dict["model_state_dict"])
            print(f"load pretrain: {os.path.basename(self.pretrain_file)}")
            return -1

        else:
            return -1

    def init_summary_writer(self):
        """ 在每个epoch的开始，设置tensorboard； """
        # 先检查：如果是从头训练，就删除以前的 tensorboard 文件
        if self.cur_epoch == 0:
            if os.path.exists(os.path.join(self.ckpt_path, self.tensorboard_dir_name)):
                for file in os.listdir(os.path.join(self.ckpt_path, self.tensorboard_dir_name)):
                    try:
                        os.remove(os.path.join(os.path.join(self.ckpt_path, self.tensorboard_dir_name, file)))
                    except:
                        pass
        # 设置 tensorboard，每个epoch保存一次
        self.tensorboard_writer = SummaryWriter(os.path.join(self.ckpt_path, self.tensorboard_dir_name))

    def close_summary_writer(self):
        """ 在每个epoch的最后，保存一个epoch的tensorboard； """
        self.tensorboard_writer.close()

    def run(self):

        # 尝试加载预训练模型
        last_epoch = self.load_ckpt_auto()

        # 写入日志
        if last_epoch < 0:  # 从头开始训练
            self.write_training_log("start training...\n", "w")

        # 正式开始训练
        for epoch in range(last_epoch+1, self.max_epochs):
            self.cur_epoch = epoch

            # 设置summary writer
            self.init_summary_writer()

            # 训练一个 epoch
            self.train_data_loader.dataset.set_epoch(self.cur_epoch)
            self.forward_one_epoch(training=True)

            # save ckpt
            self.save_ckpt()

            # eval
            self.valid_data_loader.dataset.set_epoch(self.cur_epoch)
            self.forward_one_epoch(training=False)

            # 保存当前epoch的tensorboard
            self.close_summary_writer()

    def forward_one_epoch(self, training: bool = True):
        """ 训练 or 验证一个 epoch；根据任务定义 """
        pass

    def test(self):
        """ 测试；根据任务定义 """
        pass
