""" 读取一个 label 文件，将里面的语音数据，复制到新路径下，并生成新的 label 文件； """

import os
import tqdm
import shutil


def copy_label_files(ori_label_file, tgt_dir: str, tgt_label_file: str):
    """ 读取一个 label 文件，将里面的语音数据，复制到新路径下，并生成新的 label 文件； """

    # 原始 label 文件：
    if isinstance(ori_label_file, str):
        ori_label_file = [ori_label_file]
    elif isinstance(ori_label_file, list):
        pass
    else:
        raise ValueError(f"input ori_label_file must be List or Str.")

    # wav 数据复制到这里（每个说话人单独一个文件夹）
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir, exist_ok=True)

    # 顺便统计每个音色的数据量
    tgt_count_file = os.path.join(tgt_dir, "spk_dur_cnt.txt")
    spk_cnt_dic = {}
    spk_dur_dic = {}

    # 正式运行：
    with open(tgt_label_file, 'w', encoding='utf-8') as w1:
        w1.write(" ".join(["spk", "duration", "text", "path"]) + "\n")

        for ori_label_file_i in ori_label_file:
            print(f"copying data of file {ori_label_file_i} ...")

            with open(ori_label_file_i, 'r', encoding='utf-8') as r1:
                lines = r1.readlines()

                for line in tqdm.tqdm(lines):
                    if line.startswith("spk"):  # 去掉首行
                        continue

                    try:
                        spk, dur, text, wav_ori_path = line.strip().split(" ", 3)
                        basename = os.path.basename(wav_ori_path.replace(".wav", ""))
                        lab_ori_path = os.path.join(os.path.dirname(wav_ori_path), basename + ".lab")
                    except:
                        print(f"ignore split error line: {line.strip()}")
                        continue

                    # 确定目标路径
                    tgt_spk_dir = os.path.join(tgt_dir, spk)
                    if not os.path.exists(tgt_spk_dir):
                        os.makedirs(tgt_spk_dir)

                    tgt_wav_path = os.path.join(tgt_spk_dir, basename + ".wav")
                    tgt_lab_path = os.path.join(tgt_spk_dir, basename + ".lab")

                    # 正式复制
                    shutil.copy(wav_ori_path, tgt_wav_path)
                    if os.path.exists(lab_ori_path):
                        shutil.copy(lab_ori_path, tgt_lab_path)

                    # 写入新的 label 文件
                    w1.write(" ".join([spk, dur, text, tgt_wav_path]) + "\n")

                    # 顺便统计每个音色的时长、条数
                    spk_cnt_dic[spk] = spk_cnt_dic.get(spk, 0) + 1
                    spk_dur_dic[spk] = spk_dur_dic.get(spk, 0.0) + float(dur)

    # 展示每个音色的时长、条数
    with open(tgt_count_file, 'w', encoding='utf-8') as w1:
        w1.write("\t".join(["spk", "cnt", "duration"]) + "\n")
        for spk in spk_cnt_dic:
            w1.write("\t".join([spk, str(spk_cnt_dic[spk]), str(round(spk_dur_dic[spk], 2))]) + "\n")

    return


if __name__ == "__main__":
    copy_label_files(
        ori_label_file="",
        tgt_dir="",
        tgt_label_file="",
    )
