""" 对 aishell3 语音数据进行预处理；"""

import os
import tqdm
import shutil
from urls import aishell3_useful_spks
from text_to_speech.utils.preprocess_main import main_make_label, check_punctuation, main_jieba_cut


def basename_split_func(bn: str) -> list[str, str, str]:
    """ 根据场景，将文件名分割成三份：spk、uttid、text """

    spk, uttid, text = bn.split("-", 2)
    return [spk, uttid, text]


def get_uttid2text_dic(dic_path) -> dict:
    """ 读取 aishell3 的标注文件，生成 dict，将 uttid 映射成 文本；"""

    res = {}
    with open(dic_path, "r", encoding="utf-8") as r1:
        for i, line in enumerate(r1.readlines()):
            if line.startswith("#") or len(line.strip()) < 1:
                continue
            try:
                uttid, pinyin, text = line.strip().split("|", 2)
            except:
                print(f"skip error line: {line.strip()}")

            # 将 aishell3 的 两级停顿 %、$ 进行修改；
            text = str(text).replace(" ", "").replace("%", "").replace("$", "。").replace("-", "")

            res[uttid] = text

    return res


def main_prepare_aishell(ori_data_dir, new_data_dir, useful_spks: list):
    """ 将原始 aishell3 数据进行重命名 """
    assert os.path.exists(ori_data_dir), f"aishell3 原始数据路径不存在；"

    # 从 uttid 到 文本 的映射
    uttid2text_dic = get_uttid2text_dic(
        os.path.join(ori_data_dir, "train", "label_train-set.txt")
    )

    # 遍历每个说话人：
    cnt = 0
    for spk in tqdm.tqdm(os.listdir(os.path.join(ori_data_dir, "train", "wav"))):

        ori_dir = os.path.join(ori_data_dir, "train", "wav", spk)
        if not os.path.isdir(ori_dir):
            continue

        new_spk = "aishell3_" + spk  # 新的音色名称
        if len(useful_spks) > 0 and new_spk not in useful_spks:
            continue

        # 遍历每条语音
        for file in os.listdir(ori_dir):
            ori_full_path = os.path.join(ori_dir, file)

            if not os.path.isfile(ori_full_path) or not str(file).endswith(".wav"):
                continue

            uttid = str(file).replace(".wav", "")
            text = uttid2text_dic[uttid]

            # 复制到新路径
            new_dir = os.path.join(new_data_dir, new_spk)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            new_basename = new_spk + "-" + uttid + "-" + text + ".wav"
            new_full_path = os.path.join(new_dir, new_basename)

            shutil.copy(ori_full_path, new_full_path)

            cnt += 1
            # print(f"copy and rename {cnt} data...", end="\r", flush=True)

    print(f"totally copy and rename {cnt} data.    ")

    return


def select_spks(ori_label_file, new_label_file, selected_spks):
    """ aishell3 的数据比较多，因此 选出一部分 音色，只用这些音色进行训练； """

    assert ori_label_file != new_label_file, f"输入和输出的label文件名不能一样；"

    cnt = 0
    with open(ori_label_file, "r", encoding="utf-8") as r1, \
            open(new_label_file, "w", encoding="utf-8") as w1:
        w1.write(" ".join(["spk", "duration", "text", "path"]) + "\n")

        for line in r1.readlines():
            if line.startswith("spk"):
                continue

            try:
                spk, duration, text, new_full_path = str(line).strip().split(" ")
            except:
                pass

            if spk not in selected_spks:
                continue

            w1.write(" ".join([spk, duration, text, new_full_path]) + "\n")

            cnt += 1
            print(f"write {cnt} data...", end="\r", flush=True)
    print(f"totally write {cnt} data.  ")

    return


if __name__ == "__main__":
    root_dir = "G:\\aishell3"

    start_stage = 2
    stop_stage = 2

    # stage 1: 将原始 aishell3 数据进行重命名
    if start_stage <= 1 and stop_stage >= 1:
        main_prepare_aishell(
            ori_data_dir=os.path.join(root_dir, "0.download_ori"),
            new_data_dir=os.path.join(root_dir, "1.rename"),
            useful_spks=aishell3_useful_spks,
        )

    # stage 2: 重命名、制作 label.txt、响度归一化
    if start_stage <= 2 and stop_stage >= 2:
        main_make_label(
            ori_data_dir=os.path.join(root_dir, "1.rename"),
            new_data_dir=os.path.join(root_dir, "2.no-loudnorm_16K_version-2.0"),
            basename_split_func=basename_split_func,
            loudness_norm_or_not=False,
            sample_rate=16000,
        )
        check_punctuation(
            label_file=os.path.join(root_dir, "2.no-loudnorm_16K_version-2.0" + "_label.txt"),
            res_file=os.path.join(root_dir, "2.punctuation_check_version-2.0.txt"),
        )

    # stage 3: 分词、生成拼音
    if start_stage <= 3 and stop_stage >= 3:
        main_jieba_cut(
            label_file=os.path.join(root_dir, "2.no-loudnorm_16K_version-2.0" + "_label.txt"),
            res_file=os.path.join(root_dir, "3.jiaba_cut_16K_version-2.0_label.txt"),
        )