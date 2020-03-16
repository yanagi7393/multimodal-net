import numpy as np
from dataset_builder.stft import load_audio, transform_stft
from dataset_builder.phase_helper import instantaneous_frequency
from dataset_builder.spectrogram_helper import specgrams_to_melspecgrams
from dataset_builder.modules.rawdata_loader import RawDataset
from torch.utils.data import DataLoader

DEFAULT_CONFIG = {
    "hop_length": 256,
    "win_length": 1024,
    "window": "hann",
    "batch_size": 64,
    "shuffle": True
}


def expand(mat):
    expand_vec = np.expand_dims(mat[:, -1], axis=1)
    expanded = np.hstack((mat, expand_vec, expand_vec))
    return expanded


def video_to_datasets(video_path_list, device="cpu"):
    raw_data = RawDataset(video_path_list=video_path_list, transform={})
    data_loader = DataLoader(dataset=raw_data, batch_size=DEFAULT_CONFIG['batch_size'], shuffle=DEFAULT_CONFIG["shuffle"])

    # Audio Part
    for frame_list, audio_list in data_loader:
        magnitude_list, phase_list = transform_stft(
            audio_list=audio_list,
            hop_length=DEFAULT_CONFIG["hop_length"],
            win_length=DEFAULT_CONFIG["win_length"],
            window=DEFAULT_CONFIG["window"],
            device=device,
            input_tensor=True,
            output_tensor=False,
        )

        # None handle
        magnitude_list = [magnitude for magnitude in magnitude_list if magnitude is not None]
        phase_list = [phase for phase in phase_list if phase is not None]

        log_magnitude_list = [np.log(magnitude + 1.0e-6)[: DEFAULT_CONFIG["window"]] for magnitude in magnitude_list]
        log_magnitude_list = [expand(log_magnitude) for log_magnitude in log_magnitude_list]

        IFs = [instantaneous_frequency(phase, time_axis=1)[: DEFAULT_CONFIG["window"]] for phase in phase_list]
        IFs = [expand(IF) for IF in IFs]

        # check one value
        assert log_magnitude_list[-1].shape == (DEFAULT_CONFIG["win_length"], 128)
        assert IFs[-1].shape == (DEFAULT_CONFIG["win_length"], 128)

        melspecgrams_outputs = [specgrams_to_melspecgrams(log_magnitude, IF) for log_magnitude, IF in zip(log_magnitude_list, IFs)]
        log_mel_magnitude_spectrograms = [item[0] for item in melspecgrams_outputs]
        mel_instantaneous_frequencies = [item[1] for item in melspecgrams_outputs]

        # Frame part
        ...