""" 训练、测试的主函数； """

from image_detection.executor import ImageDetectExecutor


def main():
    """ 训练 FastSpeech2 声学模型 """
    # 最多使用90%的显存；需要设置一下，要不显存使用过多，会强制重启windows
    # torch.cuda.set_per_process_memory_fraction(0.95, 0)

    trainer = ImageDetectExecutor(
        conf_file=f"./configs/yolov3_demo.yaml",
    )

    trainer.train_data_loader.dataset.save_images()
    trainer.valid_data_loader.dataset.save_images()
    trainer.test_data_loader.dataset.save_images()

    trainer.run()

    trainer.forward_one_epoch(forward_type="test")

    return


if __name__ == "__main__":
    main()
