from torch.utils.data import Dataset
from typing import Dict, List, Callable
from glob import glob
import os
import numpy as np
from PIL import Image


class IMGDataset(Dataset):
    def __init__(
        self, data_dir: str, data_type: str, transform: Callable,
    ):
        self.transform = transform
        self.file_list = sorted(glob(os.path.join(data_dir, f"*.{data_type}")))
        if len(self.file_list) == 0:
            raise FileNotFoundError("No images found")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        frame = Image.open(self.file_list[idx])

        frame = self.transform(frame)

        return frame
