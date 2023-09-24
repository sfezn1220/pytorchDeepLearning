""" 定义：图像数据集；"""
import tqdm
import yaml
import random
import torch
import cv2
from torch.utils.data import IterableDataset
from utils import read_json_lists
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


class ImageDataList(IterableDataset):
    def __init__(self, data_list_file: str, conf: dict = {}):
        """
        定义数据集：输入数据数据的格式；
        :param data_list_file: 训练集/验证集的label文件路径；
        :param conf: 数据集的参数；
        :return:
        """
        super(ImageDataList).__init__()
        self.shuffle = conf['shuffle']
        self.input_shape = conf['input_shape']
        self.n_classes = conf['n_classes']
        self.data_list = self.get_images_and_labels(data_list_file)  # 输入数据集，list 格式

        self.epoch = -1  # 每个epoch的随机打乱的种子

        self.load_image_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_shape),
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.RandomVerticalFlip(p=0.5),    # 竖直翻转
            transforms.Pad((                         # 随机 pad
                random.randint(1, 40),
                random.randint(1, 40),
                random.randint(1, 40),
                random.randint(1, 40),
            )),
            transforms.RandomRotation([-90, 90]),

            transforms.Resize(self.input_shape),
            transforms.ToTensor()
        ])

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(f"dataset: set epoch = {epoch}")

    def get_images_and_labels(self, data_list_file):
        """ 最开始的 读取 label 文件的操作；"""
        data_list = []
        for data in read_json_lists(data_list_file):
            data["label_id"] = torch.tensor(int(data["label_id"]))
            data["label_one_hot"] = F.one_hot(data["label_id"], self.n_classes).float()
            data_list.append(data)
        return data_list

    def __iter__(self):
        if self.shuffle is True:
            random.Random(self.epoch).shuffle(self.data_list)  # 按照epoch设置random的随机种子，保证可复现
        for data in self.data_list:
            img = cv2.imread(data["path"])
            img = self.load_image_tensor(img)  # 尺寸变为 224 * 224、转换成 tensor；
            data["image"] = img
            yield data

    def __len__(self):
        return len(self.data_list)


def get_image_dataloader(
        data_path: str,
        data_conf: dict,
        num_workers: int = 0,
):
    """
    生成 train_dataloader、valid_dataloader、test_dataloader；
    :param data_path: label文件路径；
    :param data_conf: 数据集的参数；
    :param num_workers: 默认为 0；
    :return:
    """
    dataset = ImageDataList(
        data_list_file=data_path,
        conf=data_conf,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=data_conf["batch_size"],
        num_workers=num_workers,
    )

    print(f"steps_per_epoch = {len(data_loader)}")

    return data_loader


if __name__ == "__main__":
    # config 文件
    conf_file = "..\configs\\vgg_base.yaml"
    with open(conf_file, 'r', encoding='utf-8') as r1:
        configs = yaml.load(r1, Loader=yaml.FullLoader)

    # 测试 dataloader 的功能
    train_data_loader = get_image_dataloader(
        data_path=configs["train_data"],
        data_conf=configs,
    )

    # 测试是 shuffle 功能：
    for epoch in range(5):
        for batch_idx, batch in enumerate(train_data_loader):
            # if batch_idx == 0:
            print(f"batch[{batch_idx}]")
            print(f"ids[{len(batch['label_id'])}] = {batch['label_id']}")
            print(f"shape = {batch['image'].shape}")
            print()
