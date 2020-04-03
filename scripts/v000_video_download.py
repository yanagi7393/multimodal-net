import os
from pytube import YouTube
import math
from dataset_builder.utils import parallelize
from urllib.error import HTTPError
from tqdm import tqdm

URL_PREFIX = "https://www.youtube.com/watch?v="


def download(video_code, output_path):
    try:
        yt = YouTube(URL_PREFIX + video_code)
        yt.streams.filter(subtype="mp4").filter(res="360p").order_by("resolution")[
            -1
        ].download(filename=video_code, output_path=output_path)
    except:
        try:
            yt.streams.filter(file_extension="mp4").filter(res="360p").order_by(
                "resolution"
            )[-1].download(filename=video_code, output_path=output_path)
        except HTTPError:
            raise HTTPError
        except:
            pass


def downloads(video_codes, save_dir="./", n_jobs=2):

    for batch_idx in tqdm(range(int(math.ceil(len(video_codes) / n_jobs)))):
        output_path = os.path.join(save_dir, str(batch_idx // 10))

        batch_codes = video_codes[batch_idx * n_jobs : (batch_idx + 1) * n_jobs]
        params = [
            {"video_code": code, "output_path": output_path} for code in batch_codes
        ]
        parallelize(func=download, params=params, n_jobs=n_jobs)
