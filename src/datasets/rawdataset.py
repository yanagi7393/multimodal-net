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
        with tempfile.TemporaryDirectory() as temp_dir:
            video_splitter = VideoSplitter(save_dir=temp_dir)

            paths = video_splitter.split(
                video_path=self.video_path_list[idx], n_frames=1, **self.params
            )
            if paths is None:
                return None, None

            image = np.array(Image.open(paths["frame_path"][-1]))
            sound = load_audio(path=paths["sound_path"][-1])

        if self.transforms.get("frame"):
            image = self.transforms.get("frame")(image)

        if self.transforms.get("sound"):
            sound = self.transforms.get("sound")(sound)

        return image, sound
