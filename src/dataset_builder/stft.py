import torch
from torch_stft import STFT
import numpy as np
import librosa
import matplotlib.pyplot as plt


def load_audio(
    path, offset=0.0, duration=None
):  # offset and duration is already handled on VideoSplitter side.
    return librosa.load(path, offset=offset, duration=duration)[0]


def transform_stft(
    audio,
    hop_length=256,
    win_length=1024,
    window="hann",
    device="cpu",
    get_tensor=False,
):
    # hop_length is overlap length of window. Generally it use win_length//4
    filter_length = win_length

    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)

    stft = STFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    ).to(device)

    magnitude, phase = stft.transform(audio)
    if get_tensor is True:
        return magnitude, phase

    return magnitude.cpu().numpy(), phase.cpu().numpy()


def inverse_stft(
    magnitude,
    phase,
    hop_length=256,
    win_length=1024,
    window="hann",
    device="cpu",
    get_tensor=False,
):
    # hop_length is overlap length of window. Generally it use win_length//4
    filter_length = win_length

    magnitude = torch.FloatTensor(magnitude).to(device)
    phase = torch.FloatTensor(phase).to(device)

    stft = STFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    ).to(device)

    reconstructed_audio = stft.inverse(magnitude, phase)

    if get_tensor is True:
        return reconstructed_audio

    return reconstructed_audio.cpu().numpy()
