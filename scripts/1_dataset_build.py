import numpy as np
from src.dataset_builder.stft import load_audio, transform_stft
from src.dataset_builder.phase_helper import instantaneous_frequency
from src.dataset_builder.spectrogram_helper import specgrams_to_melspecgrams


DEFAULT_CONFIG = {
    "hop_length": 256,
    "win_length": 1024,
    "window":"hann",
}

def expand(mat):
    expand_vec = np.expand_dims(mat[:,-1], axis=1)
    expanded = np.hstack((mat, expand_vec, expand_vec))
    return expanded


def dataset_build(audio_path, frame_path, device="cpu"):
    # store to hdf5

    # Audio part
    audio = load_audio(path=audio_path)
    magnitude, phase = transform_stft(
        audio=audio,
        hop_length=DEFAULT_CONFIG['hop_length'],
        win_length=DEFAULT_CONFIG['win_length'],
        window=DEFAULT_CONFIG['window'],
        device="cpu",
        get_tensor=False,
    )

    log_magnitude = np.log(magnitude + 1.0e-6)[:DEFAULT_CONFIG['window']]
    log_magnitude = expand(log_magnitude)

    IF = instantaneous_frequency(angle, time_axis=1)[:DEFAULT_CONFIG['window']]
    IF = expand(IF)

    assert log_magnitude.shape == (DEFAULT_CONFIG["win_length"], 128)
    assert IF.shape == (DEFAULT_CONFIG["win_length"], 128)

    log_mel_magnitude_spectrograms, mel_instantaneous_frequencies = specgrams_to_melspecgrams(log_magnitude, IF)

    # Frame part
    ...