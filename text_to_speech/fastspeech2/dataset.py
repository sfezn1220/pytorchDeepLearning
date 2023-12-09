""" 定义：FastSpeech2 语音合成任务的数据集；"""
import copy
from typing import Dict
import os
import numpy as np
import tqdm
import yaml
import random

import torch
from torch.utils.data import DataLoader

from text_to_speech.bin.tts_base_dataset import TTSBaseDataList

from text_to_speech.utils.read_textgrid import read_all_textgrid, read_spk_uttid_textgrid
from text_to_speech.utils import str2bool  # 字符串转布尔值


class FastSpeechDataList(TTSBaseDataList):
    def __init__(self, data_list_file: str, conf: Dict, model_type: str = "acoustic model", data_type: str = "train"):
        """
        定义数据集：输入数据数据的格式；
        :param data_list_file: 训练集/验证集的label文件路径；
        :param conf: 数据集的参数；
        :param model_type: 用于声学模型 or 声码器：["acoustic model", "vocoder"]
        :param data_type: ["train", "valid"]
        :return:
        """
        super().__init__(data_list_file, conf, data_type)

        self.mfa_dir = conf['mfa_dir']  # MFA对齐结果

        assert model_type.lower() in ["acoustic model", "vocoder"]
        self.model_type = model_type.lower()  # 声学模型 or 声码器

        self.use_syn_mel = conf['use_syn_mel'] if self.model_type == "vocoder" else False   # 是否使用合成Mel谱
        self.syn_mel_path = conf['syn_mel_path']  # 合成Mel谱的路径

        if self.model_type == "acoustic model":  # 只有声学模型需要MFA对齐结果
            self.get_mfa_results()  # 读取duration信息

    def get_mfa_results(self):
        """从MFA对齐结果中，读取duration信息；"""
        # 先收集uttid：
        spk_uttid_list = []
        for data in tqdm.tqdm(self.data_list):
            spk = data["spk"]
            uttid = data['uttid']
            spk_uttid_list.append([spk, uttid])

        # 读取MFA对齐结果中的所有的textgrid文件
        # all_dur_list = read_all_textgrid(self.mfa_dir)
        all_dur_list = read_spk_uttid_textgrid(self.mfa_dir, spk_uttid_list=spk_uttid_list)

        print(f"generator durations for MFA results...")

        new_data_list = []
        for data in tqdm.tqdm(self.data_list):
            uttid = data['uttid']
            phonemes = data["phonemes"]
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
        mel = mel.transpose(0, 1)  # -> [channel=80, time]
        mel_length = mel.shape[1]  # MEL谱长度
        # get mask
        mel_mask = torch.ones_like(mel, dtype=torch.float32)
        # padding
        if mel.shape[1] < self.input_max_length:
            zeros_pad = torch.zeros([80, self.input_max_length - mel.shape[1]], dtype=torch.float32)
            mel = torch.concat([mel, zeros_pad], dim=1)
            mask_zeros = torch.zeros_like(zeros_pad, dtype=torch.float32)
            mel_mask = torch.concat([mel_mask, mask_zeros], dim=1)
        elif mel.shape[1] > self.input_max_length:
            mel = mel[:, self.input_max_length]
            mel_mask = mel_mask[:, self.input_max_length]

        # get F0
        f0 = np.log(np.maximum(f0, 1e-10))
        f0 = torch.tensor(f0, dtype=torch.float32)
        f0_length = len(f0)
        # padding
        if len(f0) < self.input_max_length:
            zeros_pad = torch.zeros([self.input_max_length - len(f0)], dtype=torch.float32)
            f0 = torch.concat([f0, zeros_pad], dim=0)
            f0 *= mel_mask[0, :]
        elif len(f0) > self.input_max_length:
            f0 = f0[:self.input_max_length]

        # get energy
        energy = np.log(np.maximum(energy, 1e-10))
        energy = torch.tensor(energy, dtype=torch.float32)
        energy_length = len(energy)
        # padding
        if len(energy) < self.input_max_length:
            zeros_pad = torch.zeros([self.input_max_length - len(energy)], dtype=torch.float32)
            energy = torch.concat([energy, zeros_pad], dim=0)
            energy *= mel_mask[0, :]
        elif len(energy) > self.input_max_length:
            energy = energy[:self.input_max_length]

        # padding audio
        audio = torch.tensor(audio, dtype=torch.float32)
        seconds = len(audio) / self.sample_rate  # 语音时长，秒
        if len(audio) < self.input_max_length * self.hop_size:
            zeros_pad = torch.zeros([self.input_max_length * self.hop_size - len(audio)], dtype=torch.float32)
            audio = torch.concat([audio, zeros_pad], dim=0)
        elif len(audio) > self.input_max_length * self.hop_size:
            audio = audio[:self.input_max_length * self.hop_size]

        assert len(energy) == len(f0) == mel.shape[1] == len(audio) // self.hop_size

        return mel, f0, energy, audio, mel_mask, seconds, mel_length, f0_length, energy_length

    def load_syn_mel(self, uttid):
        """ 根据 uttid 找出对应的合成Mel谱； """

        syn_mel_path = os.path.join(self.syn_mel_path, uttid + ".npy")
        if not os.path.exists(syn_mel_path):
            return None

        syn_mel = np.load(syn_mel_path)  # [channel=80, time]
        syn_mel = torch.tensor(syn_mel, dtype=torch.float32)

        # padding
        if syn_mel.shape[1] < self.input_max_length:
            zeros_pad = torch.zeros([80, self.input_max_length - syn_mel.shape[1]], dtype=torch.float32)
            syn_mel = torch.concat([syn_mel, zeros_pad], dim=1)
        elif syn_mel.shape[1] > self.input_max_length:
            raise ValueError(f"读取到的Mel谱长度为 {syn_mel.shape[1]}，超过预设的最大长度 {self.input_max_length} ,"
                             f"需要增大 self.input_max_length .")

        return syn_mel

    def __iter__(self):
        """ 模型加载数据集的入口； """
        if self.shuffle is True:
            random.Random(self.epoch).shuffle(self.data_list)  # 按照epoch设置random的随机种子，保证可复现

        for data in self.data_list:
            (data['mel'], data['f0'], data['energy'], data['audio'], data['mel_mask'], data['seconds'],
             data['mel_length'], data['f0_length'], data['energy_length']) = self.get_features(data['path'])

            if data['mel_type'] == "syn":
                data['mel'] = self.load_syn_mel(data['uttid'])
                if data['mel'] is None:
                    continue

            data['phonemes'] = ""  # 置空，否则还需要将音素 padding 至相同长度；
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
    conf_file = "../examples/Yuanshen/configs/fs+hifi/demo-2.yaml"
    with open(conf_file, 'r', encoding='utf-8') as r1:
        configs = yaml.load(r1, Loader=yaml.FullLoader)

    configs['batch_size'] = 2
    # configs["train_data"] = configs["valid_data"]

    # 测试 dataloader 的功能
    train_data_loader = get_tts_dataloader(
            data_path=configs["train_data"],
            data_conf=configs,
            model_type="acoustic model",
            data_type="train",
        )

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
