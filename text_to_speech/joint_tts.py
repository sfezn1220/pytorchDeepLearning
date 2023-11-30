""" 合并：声学模型 & 声码器； """

import os
import copy
import soundfile as sf

from bin.base_executor import BaseExecutor
from text_to_speech.fastspeech2.fastspeech_dataset import get_tts_dataloader
from text_to_speech.fastspeech2.fastspeech_models import FastSpeech2
from text_to_speech.hifigan.hifigan import HiFiGAN


class JointTTS(BaseExecutor):
    """ 合并：声学模型 & 声码器； """  # TODO  还需要调试；

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

        # 读取 configs.yaml
        test_data_conf = copy.deepcopy(self.trainer_conf)
        test_data_conf["batch_size"] = 1
        # valid_data_conf['shuffle'] = False

        self.sample_rate = self.trainer_conf['sample_rate']

        self.test_data_loader = get_tts_dataloader(
            data_path=test_data_conf["test_data"],
            data_conf=test_data_conf,
            model_type="acoustic model",
            data_type="test",
        )

    def test(self):
        """ 模型推理部分； """

        self.acoustic_model.eval()
        self.vocoder.eval()

        for batch_idx, batch in enumerate(self.test_data_loader):

            uttids = batch["uttid"]
            phoneme_ids = batch["phoneme_ids"].to(self.device)
            spk_id = batch["spk_id"].to(self.device)
            duration_gt = batch["duration"].to(self.device)

            # 声学模型
            mel_after, mel_before, f0_predict, energy_predict, duration_predict = self.acoustic_model(phoneme_ids, spk_id, duration_gt)

            # 声码器
            audio_gen = self.vocoder.inference(mel_before)

            if len(audio_gen.shape) == 3:
                audio_gen = audio_gen.squeeze(1)  # [batch, 1, time] -> [batch, time]
            if len(audio_gen.shape) == 2:
                audio_gen = audio_gen.squeeze(1)  # [batch, time] -> [time, ]

            audio_gen = audio_gen.detach().cpu().numpy()

            save_dir = os.path.join(self.ckpt_path, "gen_audios")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            sf.write(
                file=os.path.join(save_dir, str(spk_id.detach().cpu()) + ".wav"),
                data=audio_gen,
                samplerate=self.sample_rate,
            )
