""" HiFiGAN声码器的训练过程； """

import os
import shutil
import torch
import torch.nn as nn
import time
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from text_to_speech.fastspeech2.fastspeech_executor import FastSpeechExecutor


# class HiFiGANExecutor(FastSpeechExecutor):
