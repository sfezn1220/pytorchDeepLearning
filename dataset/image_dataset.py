""" 定义：图像数据集；"""
import tqdm
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

        self.load_image_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_shape),
            transforms.ToTensor()
        ])

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
            random.Random(777).shuffle(self.data_list)  # 按照epoch设置random的随机种子，保证可复现
        for data in self.data_list:
            img = cv2.imread(data["path"])
            img = self.load_image_tensor(img)  # 尺寸变为 224 * 224、转换成 tensor；
            data["image"] = img
            yield data

    def __len__(self):
        return len(self.data_list)


def get_image_dataloader(
        train_data: str,
        valid_data: str,
        train_conf: dict,
        valid_conf: dict,
        num_workers: int = 0,
):
    """
    生成 train_dataloader、valid_dataloader；
    :param train_data: 训练集的label文件路径；
    :param valid_data: 验证集的label文件路径；
    :param train_conf: 训练集的参数；
    :param valid_conf: 验证集的参数；
    :param num_workers: 默认为 0；
    :return:
    """
    train_dataset = ImageDataList(
        data_list_file=train_data,
        conf=train_conf,
    )
    valid_dataset = ImageDataList(
        data_list_file=valid_data,
        conf=valid_conf,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_conf["batch_size"],
        num_workers=num_workers,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=train_conf["batch_size"],
        num_workers=num_workers,
    )

    print(f"steps_per_epoch = {len(train_data_loader)}")

    return train_data_loader, valid_data_loader


if __name__ == "__main__":
    data_list_file = "G:\\Images\\3.labeled_json_0921.train.txt"  # 测试数据，每行就是一条数据
    conf = {
        "shuffle": True,
        "lr": 1e-3,
        "epochs": 1000,
        "batch_size": 64,
        "n_classes": 160,
    }

    # 测试 dataloader 的功能
    train_data_loader, valid_data_loader = get_image_dataloader(
        train_data=data_list_file,
        valid_data=data_list_file,
        train_conf=conf,
        valid_conf=conf,
    )

    # 测试是 shuffle 功能：
    for epoch in range(5):
        for batch_idx, batch in enumerate(train_data_loader):
            # if batch_idx == 0:
            print(f"batch[{batch_idx}]")
            print(f"ids[{len(batch['label_id'])}] = {batch['label_id']}")
            print(f"shape = {batch['image'].shape}")
            print()
