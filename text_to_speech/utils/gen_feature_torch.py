""" 生成 Mel谱、F0 等语音合成任务所需的特征； """

from typing import Dict
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


class AudioFeatureExtractorTorch:
    def __init__(self, conf: Dict):
        """ 提取语音数据的特征，Mel谱、F0等；基于GPU torch库； """
        self.sample_rate = conf['sample_rate']  # 采样率，默认 16K Hz，如果输入数据不是这个采样率，就会重采样；
        self.hop_size = conf['hop_size']  # 每多少个点计算一次FFT；需要能被 sample_rate 整除；
        self.fft_size = conf['fft_size']
        self.win_length = conf['win_length']
        self.window = conf['window']
        self.num_mels = conf['num_mels']

        self.mel_f_min = conf['mel_f_min']  # Mel谱频率的最小值
        self.mel_f_max = conf['mel_f_max']  # Mel谱频率的最大值

        self.spec_extractor = torchaudio.transforms.Spectrogram(
            n_fft=self.fft_size,
            win_length=self.win_length,
            hop_length=self.hop_size,
            window_fn=torch.hann_window,
        )

        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.fft_size,
            win_length=self.win_length,
            hop_length=self.hop_size,
            f_min=self.mel_f_min,
            f_max=self.mel_f_max,
            n_mels=self.num_mels,
            window_fn=torch.hann_window,
        )

    def forward_mel_f0_energy(self, audio_path: str):
        """ FastSpeech 声学模型提取特征的入口； """
        audio = self.load_audio(audio_path)
        res = [
            audio,                  # [time, ]
            self.get_mel(audio),     # [frames, num_mel=80]
            self.get_f0(audio),      # [frames, ]
            self.get_energy(audio),  # [frames, ]
        ]
        return res

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """ 读取音频文件 """
        audio, rate = torchaudio.load(audio_path)
        return audio

    def get_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """ 根据 torch格式的语音，生成Mel谱； """
        mel = self.mel_extractor(audio)
        return mel

    def get_f0(self, audio: torch.Tensor) -> torch.Tensor:
        """ 根据 torch格式的语音，生成 F0； """
        frame_time = self.hop_size / self.sample_rate  # 0.0125 fot 16K Hz

        f0 = F.detect_pitch_frequency(
            audio,
            sample_rate=self.sample_rate,
            frame_time=frame_time,
            win_length=self.win_length,
            freq_low=80,
            freq_high=8000,
        )
        return f0

    def get_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """ 根据 torch格式的语音，生成 energy； """
        spec = self.spec_extractor(audio)
        energy = torch.sqrt(torch.sum(spec ** 2, dim=1))
        return energy


if __name__ == "__main__":
    conf = {
        'sample_rate': 16000,
        'hop_size': 200,
        'fft_size': 2048,
        'win_length': 800,
        'window': 'hann',
        'num_mels': 80,
        'mel_f_min': 0,
        'mel_f_max': 8000,
    }

    audio_feature_extractor_torch = AudioFeatureExtractorTorch(conf)

    path = "G:\\tts_train_data\\20231201_16K_Yuanshen34+Aishell3useful84_no-loudnorm\\Abeiduo\\Abeiduo-0000_kw7rtfpo4dmxbdmgx4ryfezznc6oo4i_000000.wav"
    # audio = audio_feature_extractor_torch.load_audio(path)
    # mel = audio_feature_extractor_torch.get_mel(audio)
    # f0 = audio_feature_extractor_torch.get_f0(audio)

    features = audio_feature_extractor_torch.forward_mel_f0_energy(path)
    # TODO 解决F0长度不一致的问题、continuous F0、to GPU

    pass