"""从网页链接的源码中，提取并下载语音数据"""
import os
import sys
import time
import requests
import librosa
import tqdm
import logging
from urls import Yuanshen_spk_url_list
from text_to_speech.utils.preprocess_main import main_make_label, check_punctuation, main_jieba_cut


# 示例
# url = "https://wiki.biligame.com/ys/%E7%94%98%E9%9B%A8%E8%AF%AD%E9%9F%B3"
# res = requests.get(url).text
# print(res)


class SpeechData:
    def __init__(self):
        self.scene = None  # 场景
        self.audios = []
        self.texts = []
        # self.audio_ch, self.audio_jp, self.audio_en, self.audio_ko = None, None, None, None  # 四种语言的语音
        # self.text_ch, self.text_jp, self.text_en, self.text_ko = None, None, None, None  # 四种语言的文本


def get_data_from_url_1(url, spk=None):
    """
        从网页源码中，提取并下载语音数据；（版本一）
        flags:
            "bikit-audio"
            "voice_text_chs_m vt_active_m"
            "voice_text_jp_m"
            "voice_text_en_m"
            "voice_text_kr_m"
    """
    res = []
    find_data = False
    try:
        url_texts = requests.get(url).text.split("\n")
    except:
        print(f"Ignore error url: {url}")
        return []

    for line in url_texts:
        # 每条语音的开始
        if line.startswith("<tbody>"):
            speech_data = SpeechData()
            find_data = True

        # 每条语音的结束
        elif "</tbody>" in line:
            find_data = False
            if len(speech_data.audios) > 0 and len(speech_data.texts) > 0:
                if len(speech_data.audios[0]) > 0 and len(speech_data.texts[0]) > 0:
                    res.append([spk, speech_data.audios[0], speech_data.texts[0]])  # 不再需要 场景

        # 每条语音在一行
        elif "bikit-audio" in line and find_data is True:
            line_split = line.strip().split("\"")
            wav_url = ""
            for part in line_split:
                if part.startswith("https"):
                    if part.endswith(".mp3") or part.endswith(".ogg") or part.endswith(".wav"):
                        wav_url = part.replace("&#58;", ":")
            speech_data.audios.append(wav_url)

        # 所有文本都在同一行
        elif "voice_text_chs_m vt_active_m" in line and find_data is True:
            texts = []
            i = 0
            while i < len(line.strip()):
                if line[i] == ">" and i+1 < len(line.strip()) and line[i+1] != "<":
                    text = ""
                    i += 1
                    while i < len(line.strip()) and line[i] != "<":
                        text += line[i]
                        i += 1
                    texts.append(text)
                else:
                    i += 1
            speech_data.texts = texts
            
    return res


def download_urls_1(data_list, save_dir="./data_download_cache"):
    """根据语音数据的链接，下载语音数据；（版本一）"""

    spk_cnt = {}
    spk_dur = {}
    for spk, audio_url, text in data_list:

        text = str(text).replace("-", "$").replace("/", "$").replace("\\", "$")\
            .replace("......", "。").replace("...", "，").replace("…", "，")\
            .replace(",", "，").replace(".", "。")\
            .replace("?", "？").replace("!", "！").replace(" ", "")  # 修饰 文本

        basename = os.path.basename(audio_url).replace(".mp3", "").replace(".ogg", "").replace(".wav", "")  # 原始文件名
        new_basename = spk + "-" \
                       + str(spk_cnt.get(spk, 0)).rjust(4, "0") + "_" + basename\
                       + "-" + text\
                       + ".mp3"  # 新的文件名
        spk_path = os.path.join(save_dir, spk)  # 按每个说话人存储
        if not os.path.exists(spk_path):
            os.makedirs(spk_path)

        download_success = False
        for _ in range(10):  # 尝试10次下载
            try:
                save_path = os.path.join(spk_path, new_basename)  # 新的绝对路径
                with open(save_path, "wb") as f:
                    url_data = requests.get(audio_url).content
                    f.write(url_data)
                download_success = True
                break
            except:
                pass
        if download_success is False:
            print(f"ignore error url: {audio_url}  {spk}  {text}")
            continue

        sample_rate = librosa.get_samplerate(save_path)
        audio_data, sample_rate = librosa.load(save_path, sr=sample_rate)
        # print(f"audio.shape: {audio_data.shape}")
        # audio_data_mono = librosa.to_mono(audio_data)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        duration_str = str(round(duration, 2))
        # print(f"duration: {duration}")

        # print("\t".join([spk, duration_str, text, save_path, scene, "中文"]))

        spk_cnt[spk] = spk_cnt.get(spk, 0) + 1
        spk_dur[spk] = spk_dur.get(spk, 0.0) + duration

        print(f"download {spk_cnt[spk]} data of {spk}...", end="\r", flush=True)

    for spk in spk_cnt:
        print(f"{spk} has {spk_cnt[spk]} pieces and {round(spk_dur[spk]/60, 2)} minutes data.")

    return


def check_url_list(spk_url_list):
    """检查输入的音色和URL是否准确：url_list: [ [ 中文名，拼音，URL ], ...]"""
    spk_list = []
    pinyin_list = []
    url_list = []

    for item in spk_url_list:

        spk, pinyin, url = item

        if len(spk) < 1 or len(pinyin) < 1 or len(url) < 1:
            continue

        if spk in spk_list:
            raise ValueError(f"spk error: {item}.")
        else:
            spk_list.append(spk)

        if pinyin in pinyin_list:
            raise ValueError(f"pinyin error: {item}.")
        else:
            pinyin_list.append(pinyin)

        if url in url_list:
            raise ValueError(f"url error: {item}.")
        else:
            url_list.append(url)

        if not str(url).endswith("%AD%E9%9F%B3"):
            raise ValueError(f"url end error: {item}.")

    print(f"Done! check spk_with_urls done.")
    print(f"totally have {len(spk_list)} speaker.")

    return


def main_download(spk_url_list, save_dir="G:\Yuanshen/download_cache"):
    """获取每个音色的语音url、下载；"""

    for _, spk, url in spk_url_list:

        data_list = get_data_from_url_1(url, spk)

        download_urls_1(data_list, save_dir=save_dir)

    return


if __name__ == "__main__":
    """ 预处理：Yuanshen 数据 """

    data_dir = "G:\Yuanshen"

    start_stage = 2
    stop_stage = 2

    # stage 0: 检查输入
    if start_stage <= 0 and stop_stage >= 0:
        check_url_list(
            Yuanshen_spk_url_list
        )

    # stage 1: 下载数据，从网页
    if start_stage <= 1 and stop_stage >= 1:
        main_download(
            Yuanshen_spk_url_list,
            save_dir="G:\Yuanshen/0.download_ori",
        )

    # stage 2: 重命名、制作 label.txt、响度归一化
    if start_stage <= 2 and stop_stage >= 2:
        main_make_label(
            ori_data_dir=os.path.join(data_dir, "2.labeled"),
            new_data_dir=os.path.join(data_dir, "3.no-loudnorm_16K_version-3.0"),
            loudness_norm_or_not=False,
            sample_rate=16000,
        )
        check_punctuation(
            label_file=os.path.join(data_dir, "3.no-loudnorm_16K_version-3.0" + "_label.txt"),
            res_file=os.path.join(data_dir, "3.punctuation_check_version-3.0.txt"),
        )

    # stage 3: 分词、生成拼音
    if start_stage <= 3 and stop_stage >= 3:
        main_jieba_cut(
            label_file=os.path.join(data_dir, "3.no-loudnorm_16K_version-3.0+aishell3-2.0_label.txt"),
            res_file=os.path.join(data_dir),
        )
