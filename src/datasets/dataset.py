from torch.utils.data import Dataset
from typing import Dict, List, Callable


class Dataset(Dataset):
    def __init__(self, data_dir: str, data_length: int, transform: Dict[str, Callable]):
        self.data_dir = data_dir
        self.data_length = data_length
        self.transform = transform

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        ...
