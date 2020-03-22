from torch.utils.data import Dataset
from src.dataset_builder.video_splitter import VideoSplitter
import tempfile
from PIL import Image
from src.dataset_builder.utils import load_audio
import torchvision
from typing import Dict, List
import numpy as np


class RawDataset(Dataset):
    def __init__(self, video_path_list: List, transforms: Dict = {}, **params):
        self.video_path_list = video_path_list
        self.params = params
        self.transforms = transforms

    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, idx):
        video_splitter = VideoSplitter()

        data_dict = video_splitter.split(
            video_path=self.video_path_list[idx], n_frames=1, **self.params
        )
        if data_dict is None:
            return "NODATA", "NODATA"

        # we use only last value
        frame = data_dict["frames"][-1]
        audio = data_dict["audios"][-1].mean(axis=-1).astype("float32")

        if self.transforms.get("frame"):
            frame = self.transforms.get("frame")(frame)

        if self.transforms.get("audio"):
            audio = self.transforms.get("audio")(audio)

        return frame, audio
