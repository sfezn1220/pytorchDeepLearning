"""关于音频处理的一些工具脚本；"""
import os
import shutil

import librosa
import soundfile as sf
from pydub import AudioSegment  # ref: https://github.com/jiaaro/pydub/blob/master/API.markdown
import pyloudnorm as pyln


def copy_and_rename(read_file: str, write_file: str) -> bool:
    """ 重命名、mp3 to wav """
    try:
        audio_data = AudioSegment.from_file(read_file)
        audio_data.export(write_file, format="wav")

        return True

    except Exception as e:
        print(f"copy_and_rename Error: {e}")
        return False


def resample(read_file: str, write_file: str = "", sample_rate: int = 48000) -> bool:
    """重采样"""
    try:
        if len(write_file) < 1:
            write_file = read_file
            ori_read_file = read_file.replace(".wav", ".tmp.wav")
            shutil.copy(write_file, ori_read_file)
        else:
            ori_read_file = read_file

        audio_data = AudioSegment.from_file(ori_read_file, format="wav")
        audio_data = audio_data.set_frame_rate(sample_rate)

        audio_data.export(write_file, format="wav")

        if os.path.exists(ori_read_file):
            os.remove(ori_read_file)

        return True

    except Exception as e:
        print(f"Resample Error: {e}")
        return False


def channels_to_mono(read_file: str, write_file: str = "") -> bool:
    """双声道 -> 单声道"""
    try:
        if len(write_file) < 1:
            write_file = read_file
            ori_read_file = read_file.replace(".wav", ".tmp.wav")
            shutil.copy(write_file, ori_read_file)
        else:
            ori_read_file = read_file

        sample_rate = librosa.get_samplerate(ori_read_file)

        audio_data = AudioSegment.from_file(ori_read_file, frame_rate=sample_rate, format="wav")
        # audio_data = audio_data.set_channels(1)  # 这种方法会造成 ffmpeg 错误；
        audio_data = audio_data.split_to_mono()[0]

        audio_data.export(write_file, format="wav")

        if os.path.exists(ori_read_file):
            os.remove(ori_read_file)

        return True

    except Exception as e:
        print(f"Channels To Mono Error: {e}")
        return False


def loudness_norm(read_file: str, write_file: str) -> bool:
    """响度归一化；  ref: https://github.com/csteinmetz1/pyloudnorm"""
    try:
        data, rate = sf.read(read_file)

        # measure the loudness first
        meter = pyln.Meter(rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(data)

        # loudness normalize audio to -12 dB LUFS
        loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)

        sf.write(write_file, loudness_normalized_audio, rate)

        return True

    except Exception as e:
        print(f"Loudness Norm Error: {e}")
        return False


def file2sample_rate(file: str) -> int:
    """输入音频路径，输出采样率，整数"""
    sample_rate = int(
        librosa.get_samplerate(file)
    )
    return sample_rate


def file2dur(file: str) -> tuple[float, str]:
    """输入音频路径，输出时长（str格式，保留两位小数）"""

    sample_rate = librosa.get_samplerate(file)
    audio_data, sample_rate = librosa.load(file, sr=sample_rate)

    # audio_data_mono = librosa.to_mono(audio_data)
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    duration_str = str(round(duration, 2))

    return duration, duration_str
