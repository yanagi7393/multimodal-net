import torch
from torch_stft import STFT
import numpy as np
import librosa
import matplotlib.pyplot as plt
from .utils import load_audio


def transform_stft(
    audio_list,
    hop_length=256,
    win_length=1024,
    window="hann",
    device="cpu",
    input_tensor=False,
    output_tensor=False,
):
    # hop_length is overlap length of window. Generally it use win_length//4
    filter_length = win_length

    if input_tensor is False:
        audio_list = [torch.FloatTensor(audio) for audio in audio_list]

    audio_list = [audio.unsqueeze(0) for audio in audio_list]
    audio_cat = torch.cat(audio_list, dim=0)
    audio_cat = audio_cat.to(device)

    stft = STFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    ).to(device)

    magnitudes, phases = stft.transform(audio_cat)
    magnitudes = torch.split(magnitudes, dim=0)
    phases = torch.split(phases, dim=0)

    if output_tensor is True:
        return magnitudes, phases

    return (
        [magnitude.cpu().numpy() for magnitude in magnitudes],
        [phase.cpu().numpy() for phase in phases],
    )


def inverse_stft(
    magnitude_list,
    phase_list,
    hop_length=256,
    win_length=1024,
    window="hann",
    device="cpu",
    input_tensor=False,
    output_tensor=False,
):
    # hop_length is overlap length of window. Generally it use win_length//4
    filter_length = win_length

    if input_tensor is False:
        magnitude_list = [torch.FloatTensor(magnitude) for magnitude in magnitude_list]
        phase_list = [torch.FloatTensor(phase) for phase in phase_list]

    magnitude_cat = torch.cat(magnitude_list, dim=0)
    phase_cat = torch.cat(phase_list, dim=0)

    stft = STFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    ).to(device)

    reconstructed_audio = stft.inverse(magnitude_cat, phase_cat)
    reconstructed_audio = torch.split(reconstructed_audio, dim=0)

    if output_tensor is True:
        return reconstructed_audio

    return [r_audio.cpu().numpy() for r_audio in reconstructed_audio]
