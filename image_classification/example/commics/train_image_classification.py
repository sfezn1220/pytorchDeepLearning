""" 训练、测试的主函数； """

import os
import torch
import copy
import yaml
import logging
import torch.nn as nn

from image_classification.executor import ImageClassificationExecutor


def main():
    """ 训练 FastSpeech2 声学模型 """
    # 最多使用90%的显存；需要设置一下，要不显存使用过多，会强制重启windows
    # torch.cuda.set_per_process_memory_fraction(0.95, 0)

    trainer = ImageClassificationExecutor(
        # conf_file=f"./configs/resnet/resnet152_ft2.yaml",
        conf_file=f"./configs/darknet/darknet_base.yaml",
    )

    trainer.run()

    trainer.forward_one_epoch(forward_type="test")

    return


if __name__ == "__main__":
    main()
