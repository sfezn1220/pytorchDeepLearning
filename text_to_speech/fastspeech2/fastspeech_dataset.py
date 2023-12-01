""" 定义：FastSpeech2 语音合成任务的数据集；"""

from typing import Dict
import os
import numpy as np
import tqdm
import yaml
import random

import torch
from torch.utils.data import DataLoader

from bin.base_dataset import BaseDataList

from text_to_speech.utils.read_textgrid import read_all_textgrid
from text_to_speech.utils.gen_feature import AudioFeatureExtractor

from text_to_speech.text_precess import TextFrontEnd  # 文本前端模型


class FastSpeechDataList(BaseDataList):
    def __init__(self, data_list_file: str, conf: Dict, model_type: str = "acoustic model", data_type: str = "train"):
        """
        定义数据集：输入数据数据的格式；
        :param data_list_file: 训练集/验证集的label文件路径；
        :param conf: 数据集的参数；
        :param model_type: 用于声学模型 or 声码器：["acoustic model", "vocoder"]
        :param data_type: ["train", "valid"]
        :return:
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

        self.text_processor = TextFrontEnd(phoneme_map_file=conf['phoneme_map'])
        self.feature_extractor = AudioFeatureExtractor(conf)

        self.input_max_seconds = conf['input_max_seconds']  # 输入音频的最大长度/秒，默认是 12秒
        self.input_max_length = self.input_max_seconds * self.sample_rate // self.hop_size

        self.input_max_tokens = conf['input_max_tokens']  # 输入音素的最大长度/个，默认是 256个字符

        self.initial_maps(conf['spk_map'])  # 初始化：spk_map
        self.mfa_dir = conf['mfa_dir']  # MFA对齐结果

        assert model_type.lower() in ["acoustic model", "vocoder"]
        self.model_type = model_type.lower()  # 声学模型 or 声码器

        self.data_list = self.get_tts_data(data_list_file)  # 输入数据集，list 格式
        if self.model_type == "acoustic model":  # 只有声学模型需要MFA对齐结果
            self.get_mfa_results()  # 读取duration信息

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
                    spk, duration, text, pinyin, path = line.split(' ', 4)
                except:
                    continue
                if float(duration) >= self.input_max_seconds:
                    print(f"skip audio with duration {duration}")
                    continue
                if len(str(pinyin).split(',')) >= self.input_max_tokens:
                    print(f"skip audio with {len(str(pinyin).split(','))} phoneme tokens")
                    continue
                data = {}
                data["spk"] = spk
                data["spk_id"] = self.spk_map[spk]
                data["path"] = path
                data["uttid"] = os.path.basename(path).replace(".wav", "")
                data['pinyin'] = pinyin

                data['phoneme_ids'] = self.get_phoneme_ids(text)  # 调用文本前端模型：文本转音素ID
                data_list.append(data)
        print(f"读取到语音数据：{len(data_list)}条。")
        return data_list

    def get_mfa_results(self):
        """从MFA对齐结果中，读取duration信息；"""
        # 读取MFA对齐结果中的所有的textgrid文件
        all_dur_list = read_all_textgrid(self.mfa_dir)

        print(f"generator durations for MFA results...")

        new_data_list = []
        for data in tqdm.tqdm(self.data_list):
            uttid = data['uttid']
            phonemes = str(data['pinyin']).split(",")
            if uttid not in all_dur_list:
                print(f"跳过MFA失败的语音：{uttid}")
                continue
            phoneme_list, dur_list = all_dur_list[uttid]
            if len(dur_list) != len(phonemes):
                print(f"跳过MFA后长度不一致的数据：{uttid}, dur_list[{len(dur_list)}] vs phonemes[{len(phonemes)}]")
                continue
            if phoneme_list != phonemes:
                print(f"跳过MFA后音素不一致的数据：{uttid}")
                continue
            duration_frames = [round(float(d) * self.sample_rate / self.hop_size) for d in dur_list]
            duration_frames = torch.tensor(duration_frames, dtype=torch.float32)
            # padding
            if len(duration_frames) < self.input_max_tokens:
                zeros_pad = torch.zeros([self.input_max_tokens - duration_frames.shape[-1]], dtype=torch.float32)
                duration_frames = torch.concat([duration_frames, zeros_pad], dim=0)

            data['duration'] = duration_frames
            new_data_list.append(data)

        del all_dur_list

        self.data_list = new_data_list
        print(f"经过MFA对齐后，语音数据还有：{len(new_data_list)}条。")
        return

    def get_features(self, audio_path):
        """读取音频、计算Mel谱；"""

        audio, mel, f0, energy = self.feature_extractor.forward_mel_f0_energy(audio_path)

        # get mel
        mel = torch.tensor(mel, dtype=torch.float32)
        mel_length = mel.shape[0]  # MEL谱长度
        # get mask
        mel_mask = torch.ones_like(mel, dtype=torch.float32)
        # padding
        if mel.shape[0] < self.input_max_length:
            zeros_pad = torch.zeros([self.input_max_length - mel.shape[0], 80], dtype=torch.float32)
            mel = torch.concat([mel, zeros_pad], dim=0)
            mask_zeros = torch.zeros_like(zeros_pad, dtype=torch.float32)
            mel_mask = torch.concat([mel_mask, mask_zeros], dim=0)

        # get F0
        f0 = np.log(np.maximum(f0, 1e-10))
        f0 = torch.tensor(f0, dtype=torch.float32)
        f0_length = len(f0)
        # padding
        if len(f0) < self.input_max_length:
            zeros_pad = torch.zeros([self.input_max_length - len(f0)], dtype=torch.float32)
            f0 = torch.concat([f0, zeros_pad], dim=0)
            f0 *= mel_mask[:, 0]

        # get energy
        energy = np.log(np.maximum(energy, 1e-10))
        energy = torch.tensor(energy, dtype=torch.float32)
        energy_length = len(energy)
        # padding
        if len(energy) < self.input_max_length:
            zeros_pad = torch.zeros([self.input_max_length - len(energy)], dtype=torch.float32)
            energy = torch.concat([energy, zeros_pad], dim=0)
            energy *= mel_mask[:, 0]

        # padding audio
        audio = torch.tensor(audio, dtype=torch.float32)
        seconds = len(audio) / self.sample_rate  # 语音时长，秒
        if len(audio) < self.input_max_length * self.hop_size:
            zeros_pad = torch.zeros([self.input_max_length * self.hop_size - len(audio)], dtype=torch.float32)
            audio = torch.concat([audio, zeros_pad], dim=0)

        assert len(energy) == len(f0) == mel.shape[0] == len(audio) // self.hop_size

        return mel, f0, energy, audio, mel_mask, seconds, mel_length, f0_length, energy_length

    def get_phoneme_ids(self, text: str):
        """ 文本到音素ID，并padding到统一长度；"""
        phoneme_ids = self.text_processor.text2phoneme_ids(text)  # 文本前端模型：文本 -> 音素ID
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.int32)
        # padding
        if len(phoneme_ids) < self.input_max_tokens:
            zeros_pad = torch.zeros([self.input_max_tokens - phoneme_ids.shape[-1]], dtype=torch.int32)
            phoneme_ids = torch.concat([phoneme_ids, zeros_pad], dim=0)
        return phoneme_ids

    def __iter__(self):
        """ 模型加载数据集的入口； """
        if self.shuffle is True:
            random.Random(self.epoch).shuffle(self.data_list)  # 按照epoch设置random的随机种子，保证可复现

        for data in self.data_list:
            (data['mel'], data['f0'], data['energy'], data['audio'], data['mel_mask'], data['seconds'],
             data['mel_length'], data['f0_length'], data['energy_length']) = self.get_features(data['path'])
            yield data


def get_tts_dataloader(
        data_path: str,
        data_conf: dict,
        num_workers: int = 0,
        model_type: str = "acoustic model",
        data_type: str = "train",
):
    """
    生成 train_dataloader、valid_dataloader、test_dataloader；
    :param data_path: label文件路径；
    :param data_conf: 数据集的参数；
    :param num_workers: 默认为 0；
    :param model_type: 用于声学模型 or 声码器：["acoustic model", "vocoder"]
    :param data_type: ["train", "valid"]
    :return:
    """
    dataset = FastSpeechDataList(
        data_list_file=data_path,
        conf=data_conf,
        model_type=model_type,
        data_type=data_type,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=data_conf["batch_size"],
        num_workers=num_workers,
    )

    print(f"{data_type} steps_per_epoch = {len(data_loader)}")

    return data_loader


if __name__ == "__main__":
    # config 文件
    conf_file = "../examples/Yuanshen/configs/fs+hifi/demo.yaml"
    with open(conf_file, 'r', encoding='utf-8') as r1:
        configs = yaml.load(r1, Loader=yaml.FullLoader)

    configs['batch_size'] = 16
    # configs["train_data"] = configs["valid_data"]

    # 测试 dataloader 的功能
    train_data_loader = get_tts_dataloader(data_path=configs["train_data"], data_conf=configs)

    # 统计音素的最大长度
    max_phoneme_ids_length = 0

    for epoch in range(1):
        for batch_idx, batch in enumerate(train_data_loader):
            # if batch_idx == 0:
            print(f"batch[{batch_idx}]")
            print(f"spk[{len(batch['spk'])}] = {batch['spk']}")
            print(f"spk_id[{len(batch['spk_id'])}] = {batch['spk_id']}")
            print(f"uttid[{len(batch['uttid'])}] = {batch['uttid']}")
            print(f"phoneme_ids.shape = {batch['phoneme_ids'].shape}")
            print(f"mel.shape = {batch['mel'].shape}")
            print(f"f0.shape = {batch['f0'].shape}")
            print(f"energy.shape = {batch['energy'].shape}")
            print(f"audio.shape = {batch['audio'].shape}")
            print(f"mel_mask.shape = {batch['mel_mask'].shape}")
            print(f"duration.shape = {batch['duration'].shape}, sum={torch.sum(batch['duration'], dim=-1)}")
            print(f"seconds = {batch['seconds']}")
            print(f"mel_length = {batch['mel_length']}")
            print()

            max_phoneme_ids_length = max(max_phoneme_ids_length, batch['phoneme_ids'].shape[1])

    print(f"max_phoneme_ids_length = {max_phoneme_ids_length}")