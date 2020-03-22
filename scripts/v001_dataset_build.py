import os
import numpy as np
from src.dataset_builder.stft import transform_stft
from src.dataset_builder.phase_helper import instantaneous_frequency
from src.dataset_builder.spectrogram_helper import specgrams_to_melspecgrams
from src.datasets.rawdataset import RawDataset
from torch.utils.data import DataLoader
from src.dataset_builder.utils import parallelize
from src.datasets.dataset import FILENAME_TEMPLATE
from src.dataset_builder.video_splitter import VideoSplitter


DEFAULT_CONFIG = {
    "hop_length": 512,
    "win_length": 2048,
    "window": "hann",
    "batch_size": 128,
    "shuffle": False,
    "n_jobs": 64,
}


def expand(mat):
    expand_vec = np.expand_dims(mat[:, -1], axis=1)
    expanded = np.hstack((mat, expand_vec, expand_vec))
    return expanded


def save_files(data_dict, dirs, file_index):
    for data_type in FILENAME_TEMPLATE.keys():
        np.save(
            os.path.join(
                dirs[data_type], FILENAME_TEMPLATE[data_type].format(file_index)
            ),
            data_dict[data_type],
        )


def stft_to_mel(magnitude, phase):
    # magnitude process
    magnitude = np.squeeze(magnitude)[: DEFAULT_CONFIG["win_length"] // 2]
    log_magnitude = np.log(magnitude + 1.0e-6)
    log_magnitude = expand(log_magnitude)

    # phase process
    phase = np.squeeze(phase)
    IF = instantaneous_frequency(phase, time_axis=1)[
        : DEFAULT_CONFIG["win_length"] // 2
    ]
    IF = expand(IF)

    # check one value
    assert log_magnitude.shape == (DEFAULT_CONFIG["win_length"] // 2, 128)
    assert IF.shape == (DEFAULT_CONFIG["win_length"] // 2, 128)

    melspecgrams = specgrams_to_melspecgrams(log_magnitude, IF)
    return melspecgrams


def video_to_datasets(video_path_list, save_dir="./dataset", device="cpu"):
    dirs = {
        data_type: os.path.join(save_dir, data_type)
        for data_type in FILENAME_TEMPLATE.keys()
    }

    for _, dir in dirs.items():
        os.makedirs(dir, exist_ok=True)

    raw_data = RawDataset(video_path_list=video_path_list, transforms={})
    data_loader = DataLoader(
        dataset=raw_data,
        batch_size=DEFAULT_CONFIG["batch_size"],
        shuffle=DEFAULT_CONFIG["shuffle"],
        pin_memory=True,
        num_workers=DEFAULT_CONFIG["batch_size"] // 2,
    )

    i = 0
    # Audio Part
    for frame_list, audio_list in data_loader:
        frame_list = [frame for frame in frame_list if frame is not "NODATA"]
        audio_list = [audio for audio in audio_list if audio is not "NODATA"]
        if len(frame_list) == 0 or len(audio_list) == 0:
            continue

        magnitude_list, phase_list = transform_stft(
            audio_list=audio_list,
            hop_length=DEFAULT_CONFIG["hop_length"],
            win_length=DEFAULT_CONFIG["win_length"],
            window=DEFAULT_CONFIG["window"],
            device=device,
            input_tensor=True,
            output_tensor=False,
        )

        stft_to_mel_params = [
            {"magnitude": magnitude, "phase": phase}
            for magnitude, phase in zip(magnitude_list, phase_list)
        ]
        melspecgrams = parallelize(
            func=stft_to_mel, params=stft_to_mel_params, n_jobs=DEFAULT_CONFIG["n_jobs"]
        )

        log_mel_spec_list = [item[0] for item in melspecgrams]
        mel_if_list = [item[1] for item in melspecgrams]

        # Save
        frame_list = [frame.cpu().numpy() for frame in frame_list]
        audio_list = [audio.cpu().numpy() for audio in audio_list]

        params = [
            {
                "data_dict": {
                    "frame": frame,
                    "audio": audio,
                    "log_mel_spec": log_mel_spec,
                    "mel_if": mel_if,
                },
                "dirs": dirs,
                "file_index": i + idx,
            }
            for idx, (frame, audio, log_mel_spec, mel_if) in enumerate(
                zip(frame_list, audio_list, log_mel_spec_list, mel_if_list)
            )
        ]

        # save files with parallel
        parallelize(func=save_files, params=params, n_jobs=DEFAULT_CONFIG["n_jobs"])

        i += len(frame_list)
