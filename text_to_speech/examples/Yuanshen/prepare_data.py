""" 预处理数据的主函数； """

import os
from text_to_speech.utils.preprocess_main import main_jieba_cut
from text_to_speech.tools import copy_label_files, split_train_test

if __name__ == "__main__":
    """ 预处理：所有数据 """

    tgt_dir = "G:\\tts_train_data\\20231201_16K_Yuanshen34+Aishell3useful84_no-loudnorm"

    start_stage = 5
    stop_stage = 5

    # stage 2: 复制数据到统一的文件夹
    combine_label_file = os.path.join(tgt_dir, "4.Yuanshen+aishell3-useful84-no-loud-norm.txt")
    if start_stage <= 2 <= stop_stage:
        copy_label_files(
            ori_label_file=[
                "G:\\Yuanshen\\3.no-loudnorm_16K_version-3.0_label.txt",
                "G:\\aishell3\\2.no-loudnorm_16K_version-2.0_label.txt",
            ],
            tgt_dir=tgt_dir,
            tgt_label_file=combine_label_file,
        )

    # stage 3: 生成 .lab 文件
    if start_stage <= 3 <= stop_stage:
        main_jieba_cut(
            label_file=combine_label_file,
            res_dir=tgt_dir,
        )

    # stage 4: MFA训练

    # stage 5: 划分训练集、测试集
    if start_stage <= 5 <= stop_stage:
        split_train_test(
            ori_label_file=combine_label_file,
        )
