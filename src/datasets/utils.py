import librosa
import math
import numpy as np


def load_audio(
    path, offset=0.0, duration=None, sr=16000, mono=True,
):  # offset and duration is already handled on VideoSplitter side.
    audio = librosa.load(path, sr=sr, mono=mono)[0]

    start_index = math.ceil(offset * sr)
    if duration is None:
        return audio[start_index:]

    end_index = start_index + (duration * sr)

    if len(audio) < end_index:
        n_tiles = math.ceil((end_index - len(audio)) / len(audio)) + 1
        audio = np.tile(audio, n_tiles)

    return audio[start_index:end_index]
