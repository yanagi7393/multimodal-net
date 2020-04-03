import torchvision
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from typing import Dict, List
from PIL import Image
import numpy as np

from .utils import load_audio
from pathlib import Path


DEFUALT = {
    "offset": 0.0,
    "duration": 4,
    "frame_size": 256,
    "sr": 16000,
    "mono": True,
}


class RawDatasetV2(Dataset):
    def __init__(
        self, frame_path_list: List, audio_path_list: List, transforms: Dict = {}
    ):
        self.frame_path_list = sorted(frame_path_list)
        self.audio_path_list = sorted(audio_path_list)

        assert len(frame_path_list) == len(audio_path_list)
        for frame_file, audio_file in zip(self.frame_path_list, self.audio_path_list):
            assert Path(frame_file).stem == Path(audio_file).stem

        self.transforms = transforms

    def __len__(self):
        return len(self.frame_path_list)

    def __getitem__(self, idx):
        frame = np.array(
            Image.open(self.frame_path_list[idx]).resize(
                (DEFUALT["frame_size"], DEFUALT["frame_size"])
            ),
            np.float32,
        )
        audio = load_audio(
            path=self.audio_path_list[idx],
            offset=DEFUALT["offset"],
            duration=DEFUALT["duration"],
            sr=DEFUALT["sr"],
            mono=DEFUALT["mono"],
        ).astype("float32")

        if self.transforms.get("frame"):
            frame = self.transforms.get("frame")(frame)

        if self.transforms.get("audio"):
            audio = self.transforms.get("audio")(audio)

        return frame, audio


def custom_collate_fn(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return (None, None)

    return default_collate(batch)
