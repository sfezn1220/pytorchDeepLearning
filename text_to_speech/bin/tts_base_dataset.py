""" 定义：TTS基础的数据集；与模型无关； """

import os
import random
from typing import Dict

import torch

from bin.base_dataset import BaseDataList

from text_to_speech.text_precess import TextFrontEnd  # 文本前端模型
from text_to_speech.utils.gen_feature import AudioFeatureExtractor  # 提取语音特征的模块


class TTSBaseDataList(BaseDataList):

    def __init__(self, data_list_file: str, conf: Dict, data_type: str = "train"):
        """
        定义：基础的数据集；与任务无关；
        :param conf: 数据集的参数；以字典的形式转入；
        :param data_type: ["train", "valid"]
        :return: 可迭代的数据集；
        """
        super().__init__(conf, data_type)

        self.sample_rate = conf['sample_rate']  # 采样率，默认 16K Hz，如果输入数据不是这个采样率，就会重采样；
        self.hop_size = conf['hop_size']  # 每多少个点计算一次FFT；需要能被 sample_rate 整除；
        self.fft_size = conf['fft_size']
        self.win_length = conf['win_length']
        self.window = conf['window']
        self.num_mels = conf['num_mels']

        self.mel_f_min = conf['mel_f_min']  # Mel谱频率的最小值
        self.mel_f_max = conf['mel_f_max']  # Mel谱频率的最大值

        self.feature_extractor = AudioFeatureExtractor(conf)  # 提取语音特征的模块

        self.text_processor = TextFrontEnd(phoneme_map_file=conf['phoneme_map'])  # 文本前端模型

        self.input_max_seconds = conf['input_max_seconds']  # 输入音频的最大长度/秒，默认是 8秒
        self.input_max_length = self.input_max_seconds * self.sample_rate // self.hop_size

        self.input_max_tokens = conf['input_max_tokens']  # 输入音素的最大长度/个，默认是 196个字符

        self.initial_maps(conf['spk_map'])  # 初始化：spk_map

        self.data_list = self.get_tts_data(data_list_file)  # 输入数据集，list 格式

    def initial_maps(self, spk_map_file: str, split_symbol: str = " "):
        """初始化： spk_map；"""
        self.spk_map = {}
        with open(spk_map_file, 'r', encoding='utf-8') as r1:
            for line in r1.readlines():
                try:
                    spk, spk_id = line.strip().split(split_symbol)
                except:
                    continue
                self.spk_map[spk] = int(spk_id)
        return

    def get_tts_data(self, data_list_file: str):
        """ 最开始的 读取 label 文件的操作；"""
        data_list = []
        with open(data_list_file, 'r', encoding='utf-8') as r1:
            for line in r1.readlines():
                line = line.strip()
                if line.startswith("spk"):
                    continue
                try:
                    if len(line.split(' ')) == 4:
                        spk, duration, text, path = line.split(' ', 3)
                    elif len(line.split(' ')) == 5:
                        spk, duration, text, pinyin, path = line.split(' ', 4)
                    else:
                        raise ValueError(f"数据集格式不正确；")
                    uttid = os.path.basename(path).replace(".wav", "")
                except:
                    continue
                if float(duration) >= self.input_max_seconds:
                    print(f"skip audio with duration {duration}, {uttid}")
                    continue
                data = {}
                data["spk"] = spk
                data["spk_id"] = self.spk_map[spk]
                data["path"] = path
                data["uttid"] = uttid

                data['phonemes'] = self.text_processor.text_processor(text)[-1]
                data['phoneme_ids'] = self.get_phoneme_ids(text)

                data['mel_type'] = "raw"
                data_list.append(data)

                # if self.use_syn_mel is True:  # 如果使用合成谱，就复制一份数据；
                #     data_copy = copy.deepcopy(data)
                #     data_copy['mel_type'] = "syn"
                #     data_list.append(data_copy)

        print(f"读取到语音数据：{len(data_list)}条。")
        # print(f"注意 use_syn_mel is {self.use_syn_mel}.")
        return data_list

    def get_phoneme_ids(self, text: str):
        """ 文本到音素ID，并padding到统一长度； """
        phoneme_ids = self.text_processor.text2phoneme_ids(text)  # 文本前端模型：文本 -> 音素ID
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.int32)
        # padding
        if len(phoneme_ids) < self.input_max_tokens:
            zeros_pad = torch.zeros([self.input_max_tokens - phoneme_ids.shape[-1]], dtype=torch.int32)
            phoneme_ids = torch.concat([phoneme_ids, zeros_pad], dim=0)
        elif len(phoneme_ids) > self.input_max_tokens:
            raise ValueError(f"超过预设的最大音素长度，需要增大 self.input_max_tokens：\n"
                             f"text{len(text)} = {text}\n"
                             f"phoneme_lens = {len(phoneme_ids)}")
        return phoneme_ids

    def get_features(self, audio_path):
        """读取音频、计算语音特征；"""
        pass

    def __iter__(self):
        """ 模型加载数据集的入口； """
        super().__iter__()
