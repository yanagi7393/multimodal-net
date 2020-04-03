import torch
from torch_stft import STFT
import numpy as np
import librosa
import matplotlib.pyplot as plt


def transform_stft(
    audio_list,
    hop_length=512,
    win_length=2048,
    window="hann",
    device="cpu",
    input_tensor=False,
    output_tensor=False,
    stft=None,
):
    # hop_length is overlap length of window. Generally it use win_length//4
    filter_length = win_length

    if input_tensor is False:
        audio_list = [torch.FloatTensor(audio) for audio in audio_list]

    audio_list = [audio.unsqueeze(0) for audio in audio_list]
    audio_cat = torch.cat(audio_list, dim=0)
    audio_cat = audio_cat.to(device)

    if stft is None:
        stft = STFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        ).to(device)

    magnitudes, phases = stft.transform(audio_cat)
    magnitudes = torch.split(magnitudes, split_size_or_sections=1, dim=0)
    phases = torch.split(phases, split_size_or_sections=1, dim=0)

    if output_tensor is True:
        return magnitudes, phases

    return (
        [magnitude.cpu().numpy() for magnitude in magnitudes],
        [phase.cpu().numpy() for phase in phases],
    )


def inverse_stft(
    magnitude_list,
    phase_list,
    hop_length=512,
    win_length=2048,
    window="hann",
    device="cpu",
    input_tensor=False,
    output_tensor=False,
    stft=None,
):
    # hop_length is overlap length of window. Generally it use win_length//4
    filter_length = win_length

    if input_tensor is False:
        magnitude_list = [torch.FloatTensor(magnitude) for magnitude in magnitude_list]
        phase_list = [torch.FloatTensor(phase) for phase in phase_list]

    magnitude_cat = torch.cat(magnitude_list, dim=0)
    phase_cat = torch.cat(phase_list, dim=0)

    if stft is None:
        stft = STFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        ).to(device)

    reconstructed_audio = stft.inverse(magnitude_cat, phase_cat)
    reconstructed_audio = torch.split(
        reconstructed_audio, split_size_or_sections=1, dim=0
    )

    if output_tensor is True:
        return reconstructed_audio

    return [r_audio.cpu().numpy() for r_audio in reconstructed_audio]
