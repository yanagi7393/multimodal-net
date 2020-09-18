from torch.utils.data import Dataset
from typing import Dict, List, Callable
from glob import glob
import os
import numpy as np


FILENAME_TEMPLATE = {
    "frame": "{}_frame.npy",
    "audio": "{}_audio.npy",
    "log_mel_spec": "{}_log_mel_spec.npy",
    "mel_if": "{}_mel_if.npy",
    "audio_label": "{}_audio_label.npy",
}


class Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Dict[str, Callable],
        load_files: List[str] = ["frame", "log_mel_spec", "mel_if", "audio_label"],
    ):
        self.filename_template = {
            data_type: FILENAME_TEMPLATE[data_type] for data_type in load_files
        }
        self.dirs = {
            data_type: os.path.join(data_dir, data_type)
            for data_type in self.filename_template.keys()
        }
        self.transforms = transforms
        self.data_length = len(glob(os.path.join(list(self.dirs.values())[0], "*.npy")))

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        data_dict = {
            data_type: np.load(
                os.path.join(
                    self.dirs[data_type], self.filename_template[data_type].format(idx)
                )
            )
            for data_type in self.filename_template.keys()
        }

        # astype -> np.uint8
        for data_type in self.filename_template.keys():
            if data_type in ["frame"]:
                data_dict[data_type] = data_dict[data_type].astype(np.uint8)

        # astype -> float32
        for data_type in self.filename_template.keys():
            if data_type in ["log_mel_spec", "mel_if"]:
                data_dict[data_type] = data_dict[data_type].astype("float32")

        # transform
        for data_type, transform in self.transforms.items():
            assert data_type in ["frame", "audio", "mel"]
            if data_type == "mel":
                (data_dict["log_mel_spec"], data_dict["mel_if"]) = transform(
                    spec=data_dict["log_mel_spec"], IF=data_dict["mel_if"]
                )
                continue

            data_dict[data_type] = transform(data_dict[data_type])

        return data_dict
