""" 训练 FastSpeech2 声学模型 """

import torch

from text_to_speech.fastspeech2.executor import FastSpeechExecutor


def main():
    """ 训练 FastSpeech2 声学模型 """
    # 最多使用90%的显存；需要设置一下，要不显存使用过多，会强制重启windows
    torch.cuda.set_per_process_memory_fraction(0.95, 0)

    trainer = FastSpeechExecutor(
        conf_file=f"./configs/fs+hifi/base-2.yaml",
    )

    # trainer.train_data_loader.dataset.save_features()
    # trainer.valid_data_loader.dataset.save_features()

    # trainer.run()

    trainer.forward_one_epoch(forward_type="gen_spec_train")
    trainer.forward_one_epoch(forward_type="gen_spec_valid")

    return


if __name__ == "__main__":
    main()
