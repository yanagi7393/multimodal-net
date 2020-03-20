from torch.utils.data import Dataset
from typing import Dict, List, Callable
import os


FILENAME_TEMPLATE = {
    "frame": "{}_frame.npy",
    "audio": "{}_audio.npy",
    "log_mel_spec": "{}_log_mel_spec.npy",
    "mel_if": "{}_mel_if.npy",
}


class Dataset(Dataset):
    def __init__(
        self, data_dir: str, data_length: int, transforms: Dict[str, Callable]
    ):
        self.dirs = {
            data_type: os.path.join(data_dir, data_type)
            for data_type in FILENAME_TEMPLATE.keys()
        }
        self.data_length = data_length
        self.transforms = transforms

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        data_dict = {
            data_type: np.load(
                os.path.join(
                    self.dirs[data_type], FILENAME_TEMPLATE[data_type].format(idx)
                )
            )
            for data_type in ILENAME_TEMPLATE.keys()
        }

        for data_type, transform in self.transforms.items():
            data_dict[data_type] = transform(data_dict[data_type])

        return data_dict
