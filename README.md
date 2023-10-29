使用 pytorch，尝试从头搭建一版深度学习模型；

---

## 1、demo 搭建一版图像分类模型 ing

---

## 2、搭建一版语音合成模型 TODO

sample:

<audio id="audio" controls="" preload="none">
      <source id="wav" src="wavs_sample.Babala-0000_hxj4a3e2aw2oundfm7ddctfs8nqzake_000077.wav">
</audio>

---

## 训练环境安装

```shell
# 新建虚拟环境
conda create -n torch
```

```shell
# 进入虚拟环境
conda activate torch
```

```shell
# 安装 torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install scipy>=1.2.0
conda install matplotlib
conda install -c conda-forge sox librosa
pip install soundfile
conda install -c conda-forge pyworld
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install tqdm
conda install pydub -c conda-forge
conda install Pillow

pip install pyloudnorm jieba pyyaml sox
pip install pypinyin
pip install imageio
pip install textgrid
```

---

## MFA

### MFA环境安装

```shell
# 新建虚拟环境
conda create -n mfa
```

```shell
# 进入虚拟环境
conda activate mfa
```

```shell
# 安装官方的MFA工具包
# conda install -c conda-forge montreal-forced-aligner

conda install -c conda-forge kaldi 
conda install -c conda-forge sox librosa
conda install -c conda-forge biopython praatio tqdm requests colorama pyyaml pynini openfst baumwelch ngram

conda install -c conda-forge montreal-forced-aligner
```

### MFA训练
```shell
# 进入虚拟环境
conda activate mfa
```

```shell
# mfa train  "G:\Yuanshen\3.loudnorm_16K_version-2.0-copy-for-mfa"  "G:\Yuanshen\lexicon.txt"  "G:\Yuanshen\acoustic_model.zip"  --output_directory "G:\Yuanshen\3.loudnorm_16K_version-2.0-copy-for-mfa_output"  --num_jobs 4  --clean
mfa train  "G:\Yuanshen\3.loudnorm_16K_version-2.0"  "G:\Yuanshen\lexicon.txt"  "G:\Yuanshen\3.loudnorm_16K_version-2.0-mfa_output\acoustic_model.zip"  --output_directory "G:\Yuanshen\3.loudnorm_16K_version-2.0-mfa_output"  --num_jobs 4  --clean
```