""" 训练 FastSpeech2 声学模型 """

import torch

from text_to_speech.fastspeech2.train_fastspeech import train


def main():
    """ 训练 FastSpeech2 声学模型 """
    # 最多使用90%的显存；需要设置一下，要不显存使用过多，会强制重启windows
    torch.cuda.set_per_process_memory_fraction(0.93, 0)

    train(conf_file=f"./configs/fs+hifi/demo.yaml")

    return


if __name__ == "__main__":
    main()
