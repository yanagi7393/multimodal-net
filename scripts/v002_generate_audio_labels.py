import panns_inference
from panns_inference import AudioTagging, labels
from datasets.dataset import Dataset
from torch.utils.data import DataLoader
import librosa
import os
from tqdm import tqdm
import numpy as np
import fire


def _build_data_loader(data_dir, batch_size=256, sr=16000):
    transforms = {}
    if sr != 32000:
        transforms["audio"] = lambda audio: librosa.resample(audio, sr, 32000)
        print(f"[!] sr: {sr} -> 32000")

    # DEFINE: DATASETS
    train_dataset = Dataset(
        data_dir=data_dir, transforms=transforms, load_files=["audio"],
    )

    # DEFINE: DATA LOADER
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=batch_size // 2,
    )
    return train_data_loader


def _make_save_dir(data_dir):
    save_dir = os.path.join(data_dir, "audio_label")
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def generate(data_dir, batch_size=256, device="cuda", sr=16000):
    save_dir = _make_save_dir(data_dir=data_dir)

    inferenced_labels = []
    panns = AudioTagging(device=device)
    data_loader = _build_data_loader(data_dir=data_dir, batch_size=batch_size, sr=sr)

    for data_dict in tqdm(data_loader):
        inferences, _ = panns.inference(data_dict["audio"])
        inferenced_labels += inferences.argmax(axis=1).tolist()

    for idx, inferenced_label in enumerate(inferenced_labels):
        np.save(
            os.path.join(save_dir, f"{idx}_audio_label.npy"),
            np.array([inferenced_label]),
        )


if __name__ == "__main__":
    fire.Fire(generate)
