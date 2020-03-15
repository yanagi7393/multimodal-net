from src.dataset_builder.stft import load_audio, transform_stft
from src.preprocessor import ?


def expand(mat):
    expand_vec = np.expand_dims(mat[:,125], axis=1)
    expanded = np.hstack((mat, expand_vec, expand_vec))
    return expanded


def dataset_build():
    # Audio part
    audio_path = ...
    audio = load_audio(path=audio_path)
    magnitude, phase = transform_stft(
        audio=audio,
        hop_length=256,
        win_length=1024,
        window="hann",
        device="cpu",
        get_tensor=False,
    )

    # magnitude = np.log(magnitude+ 1.0e-6)[:1024]
    # IF = phase_operation.instantaneous_frequency(angle,time_axis=1)[:1024]

    # magnitude = expand(magnitude)
    # IF = expand(IF)

    # logmelmag, mel_p = spec_helper.specgrams_to_melspecgrams(magnitude, IF)
    
    # assert magnitude.shape ==(1024, 128)
    # assert IF.shape ==(1024, 128)
