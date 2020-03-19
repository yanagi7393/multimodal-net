import numpy as np
from dataset_builder.stft import load_audio, transform_stft
from dataset_builder.phase_helper import instantaneous_frequency
from dataset_builder.spectrogram_helper import specgrams_to_melspecgrams
from dataset_builder.modules.rawdata_loader import RawDataset
from torch.utils.data import DataLoader
from dataset_builder.utils import parallelize
import os

DEFAULT_CONFIG = {
    "hop_length": 256,
    "win_length": 1024,
    "window": "hann",
    "batch_size": 64,
    "shuffle": False
}


def expand(mat):
    expand_vec = np.expand_dims(mat[:, -1], axis=1)
    expanded = np.hstack((mat, expand_vec, expand_vec))
    return expanded

def save_files(frame, audio, log_mel_spec, mel_if, dirs, file_index):
    np.save(frame, os.path.join(dirs['frame'], f'{file_index}_frame.npy'))
    np.save(audio, os.path.join(dirs['audio'], f'{file_index}_audio.npy'))
    np.save(log_mel_spec, os.path.join(dirs['log_mel_spec'], f'{file_index}_log_mel_spec.npy'))
    np.save(mel_if, os.path.join(dirs['mel_if'], f'{file_index}_mel_if.npy'))

def video_to_datasets(video_path_list, save_dir="./dataset", device="cpu"):
    dirs = {data_type : os.path.join(save_dir, data_type) for data_type in  ["frame", "audio", "log_mel_spec", "mel_if"]}

    for _, dir in dirs.items():
        os.makedirs(dir, exist_ok=True)

    raw_data = RawDataset(video_path_list=video_path_list, transform={})
    data_loader = DataLoader(dataset=raw_data, batch_size=DEFAULT_CONFIG['batch_size'], shuffle=DEFAULT_CONFIG["shuffle"])

    i = 0
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
        log_mel_spec_list = [item[0] for item in melspecgrams_outputs]
        mel_if_list = [item[1] for item in melspecgrams_outputs]

        # Save
        frame_list = [frame.cpu().numpy() for frame in frame_list]
        audio_list = [audio.cpu().numpy() for audio in audio_list]

        params = [{"frame": frame, "audio": audio, "log_mel_spec": log_mel_spec, "mel_if": mel_if, "dirs": dirs, "file_index": i + idx} for idx, (frame, audio, log_mel_spec, mel_if) in enumerate(zip([frame_list, audio_list, log_mel_spec_list, mel_if_list]))]

        # save files with parallel
        parallelize(func=save_files, params=params, n_jobs=64)

        i += len(frame_list)