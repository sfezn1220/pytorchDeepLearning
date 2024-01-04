""" 合并：声学模型 & 声码器； """

import os
import copy
import soundfile as sf

import torch

from bin.base_executor import BaseExecutor
from text_to_speech.fastspeech2.dataset import get_tts_dataloader
from text_to_speech.fastspeech2.fastspeech2 import FastSpeech2
from text_to_speech.hifigan.hifigan import HiFiGAN

from text_to_speech.text_precess import TextFrontEnd  # 文本前端模型


class JointTTS(BaseExecutor):
    """ 合并：声学模型 & 声码器； """

    def __init__(self,
                 conf_file: str,
                 acoustic_model: str = "fastspeech2",
                 vocoder: str = "hifigan"
                 ):
        """ 合并：声学模型 & 声码器； """
        super().__init__(conf_file, name="joint-" + acoustic_model + "-" + vocoder)

        if acoustic_model.lower() == "fastspeech2":
            self.acoustic_model = FastSpeech2(self.trainer_conf).to(self.device)
        else:
            raise ValueError(f"acoustic_model.lower() must in ['fastspeech2'] now.")

        if vocoder.lower() == "hifigan":
            self.vocoder = HiFiGAN(self.trainer_conf, device=self.device).to(self.device)
        else:
            raise ValueError(f"vocoder.lower() must in ['hifigan'] now.")

        self.sample_rate = 16000

        self.load_models()

        self.text_processor = TextFrontEnd(phoneme_map_file=self.trainer_conf['phoneme_map'])  # 文本前端模型

    def load_models(self):
        """ 依次读取：声学模型、声码器； """
        acoustic_model_dict = torch.load("F:\\models_tts_fs+hifi\\base-2\\fastspeech2-model_epoch-0101.pth")
        self.acoustic_model.load_state_dict(acoustic_model_dict["model_state_dict"])

        vocoder_model_dict = torch.load("F:\\models_tts_fs+hifi\\base-2\\hifigan-model_epoch-0082.pth")
        self.vocoder.load_state_dict(vocoder_model_dict["model_state_dict"])

    def test(self, text, spk_id):
        """ 模型推理部分； """

        self.acoustic_model.eval()
        self.vocoder.eval()

        # 文本 -> 音素
        phoneme_ids = self.text_processor.text2phoneme_ids(text)  # 文本前端模型：文本 -> 音素ID
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.int32)
        phoneme_ids = phoneme_ids.to(self.device)
        phoneme_ids = phoneme_ids.unsqueeze(0)

        spk_id = torch.tensor(spk_id, dtype=torch.int32)
        spk_id = spk_id.to(self.device)
        spk_id = spk_id.unsqueeze(0)

        # 声学模型
        mel_after, mel_before, f0_predict, energy_predict, duration_predict = \
            self.acoustic_model(phoneme_ids, spk_id)

        # 声码器
        audio_gen = self.vocoder.inference(mel_after)

        if len(audio_gen.shape) == 3:
            audio_gen = audio_gen.squeeze(1)  # [batch, 1, time] -> [batch, time]
        if len(audio_gen.shape) == 2:
            audio_gen = audio_gen.squeeze(1)  # [batch, time] -> [time, ]

        audio_gen = audio_gen.detach().squeeze(0).cpu().numpy()

        save_dir = os.path.join(self.ckpt_path, "gen_audios")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        sf.write(
            file=os.path.join(save_dir, str(text) + ".wav"),
            data=audio_gen,
            samplerate=self.sample_rate,
        )
