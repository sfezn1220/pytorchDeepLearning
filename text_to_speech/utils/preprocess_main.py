"""
    所有数据的预处理的公共脚本；
    包含功能：
        main_make_label：
            读取一个文件夹内，每个说话人的所有数据，重命名、响度归一化、提取单声道、重采样至 22.05K、制作 label 文件；
        check_punctuation：
            读取 label 文件，统计标点符号有哪些；
        main_jieba_cut：
            读取 label 文件，进行 jieba 分词，生成包含 音素的 label 文件；
"""

import os
import tqdm
from .audio_process import file2dur, file2sample_rate, loudness_norm, channels_to_mono, resample, copy_and_rename  # 语音处理相关
from text_to_speech.text_precess import TextFrontEnd  # 处理文字相关
from .dict_process import dic_sort  # 处理字典相关


def main_make_label(
        ori_data_dir: str,
        new_data_dir: str,
        basename_split_func=None,
        sample_rate=22050,
        loudness_norm_or_not=True,
):
    """根据下载好的语音数据，重命名、响度归一化、制作 label.txt"""

    max_sample_rate = 0
    min_sample_rate = 1e10

    cnt = 0
    with open(new_data_dir + "_label.txt", "w", encoding="utf-8") as w1:
        w1.write(" ".join(["spk", "duration", "text", "path"]) + "\n")

        # 每个音色
        for spk in tqdm.tqdm(os.listdir(ori_data_dir)):
            ori_full_dir = os.path.join(ori_data_dir, spk)
            new_full_dir = os.path.join(new_data_dir, spk)
            if not os.path.exists(new_full_dir):
                os.makedirs(new_full_dir)

            if not os.path.isdir(ori_full_dir):  # 非文件夹
                continue

            # 当前音色的每条数据
            for file in os.listdir(ori_full_dir):

                if not file.endswith(".mp3") and not file.endswith(".wav") and not file.endswith(".ogg"):
                    continue

                ori_basename = file.replace(".mp3", "").replace(".wav", "").replace(".ogg", "")
                ori_full_path = os.path.join(ori_full_dir, file)

                _, duration = file2dur(ori_full_path)

                try:
                    if basename_split_func is not None:
                        _, uttid, text = basename_split_func(ori_basename)
                    else:
                        _, uttid, text = ori_basename.split("-", 2)

                    text = str(text).replace("_01", "") \
                        .replace("......", "。").replace("...", "，") \
                        .replace("……", "，").replace("…", "，") \
                        .replace(",", "，").replace(".", "。") \
                        .replace("?", "？").replace("!", "！").replace(" ", "")  # 修饰 文本
                except:
                    print(f"ignore error basename: {ori_basename}")
                    continue

                new_uttid = uttid + "_" + str(cnt).rjust(6, "0")
                new_basename = "-".join([spk, new_uttid])
                new_full_path = os.path.join(new_full_dir, new_basename + ".wav")

                # 复制到新路径下
                # shutil.copy(ori_full_path, new_full_path)
                if copy_and_rename(ori_full_path, new_full_path) is False:
                    continue

                # 响度归一化
                if loudness_norm_or_not is True:
                    if loudness_norm(new_full_path, new_full_path) is False:
                        continue

                # 双声道 -> 单声道
                if channels_to_mono(new_full_path) is False:
                    continue

                # 重采样至 22.05K
                if resample(new_full_path, sample_rate=sample_rate) is False:
                    continue

                max_sample_rate = max(max_sample_rate, file2sample_rate(new_full_path))
                min_sample_rate = min(min_sample_rate, file2sample_rate(new_full_path))

                w1.write(" ".join([spk, duration, text, new_full_path]) + "\n")
                cnt += 1
                # print(f"write {cnt} data...", end="\r", flush=True)

        print(f"totally write {cnt} data.  ")

        print(f"max_sample_rate = {max_sample_rate}")
        print(f"min_sample_rate = {min_sample_rate}")

    return


def check_punctuation(label_file, res_file):
    """读取 label 文件，统计标点符号有哪些"""
    textFrontEnd = TextFrontEnd()  # 文本前端模型

    punctuation_dic = {}
    with open(label_file, "r", encoding="utf-8") as r1:
        for i, line in enumerate(r1.readlines()):
            if line.startswith("spk"):  # 去掉首行
                continue

            try:
                spk, dur, text, path = line.strip().split(" ", 3)
            except:
                print(f"ignore split error line: {line.strip()}")
                continue

            for char in text:
                if textFrontEnd.is_alpha(char) is True:
                    # print(f"Ignore alpha: {char}\ntext: {text}\n")
                    continue
                if textFrontEnd.is_chinese(char) is False:  # 不是汉字
                    punctuation_dic[char] = punctuation_dic.get(char, 0) + 1
                    if char not in [",", "，", ".", "。", "?", "？", "!", "！", "…", "「", "」", "『", "』", ":", "：", "、", "~", "~", "～", ";", "；", "—", "·", "《", "》", "♪", "$"]:
                        print(f"punctuation: {char}\ntext: {text}\n")

            print(f"read {i+1} line data...", end="\r", flush=True)

    print(f"totally read {i+1} line data.")

    dic_sort(punctuation_dic, reverse=True)  # 降序排序、展示

    return


def main_jieba_cut(label_file, res_dir, spk_map_file="", lexicon_file=""):
    """读取 label 文件，进行 jieba 分词"""
    textFrontEnd = TextFrontEnd()  # 文本前端模型

    if isinstance(label_file, str):
        label_file = [label_file]
    elif isinstance(label_file, list):
        pass
    else:
        raise ValueError(f"input ori_label_file must be List or Str.")

    all_words = []  # 统计所有出现的词语，以生成MFA所需的.lab文件
    if len(lexicon_file) < 1:
        lexicon_file = os.path.join(res_dir, "lexicon.txt")
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    all_spk_list = []  # 统计所有音色
    if len(spk_map_file) < 1:
        spk_map_file = os.path.join(res_dir, "spk_map.txt")

    max_phonemes = 0  # 统计最大的文本长度

    min_dur = 1e10
    max_dur = 0.0
    total_dur_sec = 0.0
    with open(spk_map_file, "w", encoding="utf-8") as w2,\
            open(lexicon_file, "w", encoding="utf-8") as w3:

        for label_file_i in label_file:
            with open(label_file_i, "r", encoding="utf-8") as r1:

                all_lines = r1.readlines()

                for line in tqdm.tqdm(all_lines):
                    if line.startswith("spk"):  # 去掉首行
                        continue

                    try:
                        spk, dur, ori_text, path = line.strip().split(" ", 3)
                        min_dur = min(min_dur, float(dur))
                        max_dur = max(max_dur, float(dur))
                        total_dur_sec += float(dur)
                    except:
                        print(f"ignore split error line: {line.strip()}")
                        continue

                    jieba_word_list, phoneme_list = textFrontEnd.text_processor(ori_text)

                    max_phonemes = max(max_phonemes, len(phoneme_list))

                    if spk not in all_spk_list:  # 统计所有说话人、生成 spk_map
                        all_spk_list.append(spk)

                    with open(str(path).replace(".wav", ".lab"), "w", encoding="utf-8") as w_lab:
                        for j, [word, phonemes] in enumerate(jieba_word_list):
                            if j != 0:
                                w_lab.write(" ")
                            w_lab.write(word)

                    for word, phonemes in jieba_word_list:  # 统计所有词语
                        if len(phonemes) > 1 and [word, phonemes] not in all_words:
                            all_words.append([word, phonemes])

                    # print(f"read {i+1} line data...", end="\r", flush=True)

        # 写入 spk_map
        all_spk_list.sort()
        for j, spk in enumerate(all_spk_list):
            w2.write(" ".join([spk, str(j)]) + "\n")

        # 写入：发音词典
        all_words.sort(key=lambda x:x[0])
        for word, phonemes in all_words:
            w3.write(" ".join([word, phonemes]) + "\n")

    print(f"totally read {len(all_lines)} line data.")

    print()
    print(f"max_phonemes = {max_phonemes} tokens.")

    print()
    print(f"min_dur = {min_dur} seconds.")
    print(f"max_dur = {max_dur} seconds.")

    print()
    print(f"total_dur = {round(total_dur_sec / 60, 2)} minutes, {round(total_dur_sec / 3600, 2)} hours.")

    return
