""" 生成 Mel谱、F0 等语音合成任务所需的特征； """

from typing import Dict
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from scipy.interpolate import interp1d

import torch
import torchaudio
import torchaudio.functional as F


class AudioFeatureExtractor:
    def __init__(self, conf: Dict):
        """ 提取语音数据的特征，Mel谱、F0等；基于CPU librosa库； """
        self.sample_rate = conf['sample_rate']  # 采样率，默认 16K Hz，如果输入数据不是这个采样率，就会重采样；
        self.hop_size = conf['hop_size']  # 每多少个点计算一次FFT；需要能被 sample_rate 整除；
        self.fft_size = conf['fft_size']
        self.win_length = conf['win_length']
        self.window = conf['window']
        self.num_mels = conf['num_mels']

        self.mel_f_min = conf['mel_f_min']  # Mel谱频率的最小值
        self.mel_f_max = conf['mel_f_max']  # Mel谱频率的最大值

    def forward_mel_f0_energy(self, audio_path: str):
        """ FastSpeech 声学模型提取特征的入口； """
        audio = self.load_audio(audio_path)
        res = [
            audio,                  # [time, ]
            self.gen_mel(audio),     # [frames, num_mel=80]
            self.gen_f0(audio),      # [frames, ]
            self.gen_energy(audio),  # [frames, ]
        ]
        return res

    def load_audio(self, audio_path: str) -> np.array:
        """ 读取语音数据；如果和预设的采样率不一致，就会重采样； """
        audio, rate = sf.read(audio_path)

        if rate != self.sample_rate:
            audio = librosa.resample(audio, self.sample_rate)

        return np.array(audio, dtype=np.double)

    def gen_spec(self, audio: np.array) -> np.array:
        """ 生成频谱； """
        stft = librosa.stft(
            audio,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
        )
        spec, _ = librosa.magphase(stft)
        return spec

    def gen_mel(self, audio: np.array) -> np.array:
        """ 生成 Mel谱； """
        spec = self.gen_spec(audio)
        # get mel spec
        mel_filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.fft_size,
            n_mels=self.num_mels,
            fmin=self.mel_f_min,
            fmax=self.mel_f_max,
        )
        mel = np.log10(
            np.maximum(
                np.dot(
                    mel_filters,
                    spec,
                ),
                1e-10,
            )
        ).T
        return mel

    def gen_f0(self, audio: np.array, continuous: bool = True) -> np.array:
        """ 计算：F0； """
        # audio: float64
        f0, t = pw.dio(
            x=audio,
            fs=self.sample_rate,
            f0_ceil=self.mel_f_max,
            frame_period=1000 * self.hop_size / self.sample_rate,
        )
        f0 = pw.stonemask(x=audio, f0=f0, temporal_positions=t, fs=self.sample_rate)

        if continuous is True:
            f0 = continuous_f0(f0)
        return f0

    def gen_energy(self, audio: np.array) -> np.array:
        """ 计算 energy； """
        spec = self.gen_spec(audio)
        energy = np.sqrt(np.sum(spec ** 2, axis=0))
        return energy


def continuous_f0(f0: np.array) -> np.array:
    """ 连续化 f0 """
    if sum(f0) == 0:
        return f0

    # pad start and end zero-value
    for i in range(len(f0)):
        if f0[i] != 0:
            f0[:i] = f0[i]
            break
    for j in range(len(f0)-1, -1, -1):
        if f0[j] != 0:
            f0[j:] = f0[j]
            break

    # continuous
    non_zero_index = []
    for i in range(len(f0)):
        if f0[i] != 0:
            non_zero_index.append(i)

    interp_fn = interp1d(non_zero_index, f0[non_zero_index])
    f0 = interp_fn(np.arange(0, len(f0)))
    return f0


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

        self.device = "cuda"

        self.spec_extractor = torchaudio.transforms.Spectrogram(
            n_fft=self.fft_size,
            win_length=self.win_length,
            hop_length=self.hop_size,
            window_fn=torch.hann_window,
        ).to(self.device)

        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.fft_size,
            win_length=self.win_length,
            hop_length=self.hop_size,
            f_min=self.mel_f_min,
            f_max=self.mel_f_max,
            n_mels=self.num_mels,
            window_fn=torch.hann_window,
        ).to(self.device)

    def forward_mel_f0_energy(self, audio_path: str):
        """ FastSpeech 声学模型提取特征的入口； """
        audio = self.load_audio(audio_path)
        res = [
            audio.squeeze(0),                  # [time, ]
            self.get_mel(audio).squeeze(0).transpose(0, 1),     # [frames, num_mel=80]
            self.get_f0(audio).squeeze(0),      # [frames, ]
            self.get_energy(audio).squeeze(0),  # [frames, ]
        ]
        return res

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """ 读取音频文件 """
        audio, rate = torchaudio.load(audio_path)
        audio = audio.to(self.device)
        return audio

    def get_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """ 根据 torch格式的语音，生成Mel谱； """
        mel = self.mel_extractor(audio)
        return mel

    def get_f0_error(self, audio: torch.Tensor) -> torch.Tensor:
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

    def get_f0(self, audio: torch.tensor, continuous: bool = True) -> torch.tensor:
        """ 计算：F0； """
        audio = audio.detach().squeeze(0).cpu().numpy()  # to float64
        audio = audio.astype(np.float64)
        f0, t = pw.dio(
            x=audio,
            fs=self.sample_rate,
            f0_ceil=self.mel_f_max,
            frame_period=1000 * self.hop_size / self.sample_rate,
        )
        f0 = pw.stonemask(x=audio, f0=f0, temporal_positions=t, fs=self.sample_rate)

        if continuous is True:
            f0 = continuous_f0(f0)
        f0 = torch.tensor(f0, dtype=torch.float32, device=self.device).unsqueeze(0)
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
    path = "G:\\tts_train_data\\20231201_16K_Yuanshen34+Aishell3useful84_no-loudnorm\\Abeiduo\\Abeiduo-0000_kw7rtfpo4dmxbdmgx4ryfezznc6oo4i_000000.wav"

    audio_feature_extractor_torch = AudioFeatureExtractorTorch(conf)
    features = audio_feature_extractor_torch.forward_mel_f0_energy(path)
    # TODO 解决F0长度不一致的问题、continuous F0、to GPU

    # audio_feature_extractor = AudioFeatureExtractor(conf)
    # features = audio_feature_extractor.forward_mel_f0_energy(path)

    pass