""" 联合测试：声学模型 & 声码器； """

import torch

from text_to_speech import JointTTS


def main():
    """ 训练 FastSpeech2 声学模型 """
    # 最多使用90%的显存；需要设置一下，要不显存使用过多，会强制重启windows
    torch.cuda.set_per_process_memory_fraction(0.93, 0)

    trainer = JointTTS(
        conf_file=f"./configs/fs+hifi/demo.yaml",
        acoustic_model="fastspeech2",
        vocoder="hifigan",
    )

    trainer.test()

    return


if __name__ == "__main__":
    main()
