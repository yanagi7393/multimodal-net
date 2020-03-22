import os
import random
from pathlib import Path
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import crop
from .utils import parallelize
import math

DEFUALT = {
    "offset": 10,
    "duration": 4,
    "n_frames": 1,
    "frame_size": 256,
    "n_jobs": 64,
}


class VideoSplitter:
    def __init__(self, save_dir):
        self.frames_dir = os.path.join(save_dir, "frames")
        self.sounds_dir = os.path.join(save_dir, "sounds")

        for _dir in [self.frames_dir, self.sounds_dir]:
            os.makedirs(_dir, exist_ok=True)

    def split(
        self,
        video_path,
        offset=DEFUALT["offset"],
        duration=DEFUALT["duration"],
        n_frames=DEFUALT["n_frames"],
        frame_size=DEFUALT["frame_size"],
    ):
        video_name = Path(video_path).stem

        get_frame_path = lambda index: os.path.join(
            self.frames_dir, video_name + f"-{index}.png"
        )
        sound_path = os.path.join(self.frames_dir, video_name + ".mp3")

        # load video
        video = VideoFileClip(video_path)

        # if whole duration size is smaller than duration, just pass.
        if video.duration < duration:
            return None

        end_time = offset + duration
        end_time = min(end_time, math.floor(video.duration))
        start_time = end_time - duration

        video = video.subclip(start_time, end_time)
        assert video.duration == duration

        # crop and resize
        (w, h) = video.size
        min_size = min(w, h)
        video = crop(
            video, width=min_size, height=min_size, x_center=w / 2, y_center=h / 2
        ).resize(width=frame_size, height=frame_size)

        # frame part
        time_ranges = range(0, duration)
        pick_frame_times = random.sample(time_ranges, n_frames)
        for idx, t in enumerate(pick_frame_times):
            try:
                video.save_frame(get_frame_path(index=idx), t)
                continue
            except OSError:
                try:
                    video.save_frame(get_frame_path(index=idx), time_ranges[-1])
                except:
                    return None

        # audio part
        audio = video.audio

        # Replace the parameter with the location along with filename
        audio.write_audiofile(sound_path)

        video.close()

        return {
            "frame_path": [
                get_frame_path(index=idx) for idx in range(len(pick_frame_times))
            ],
            "sound_path": [sound_path],
        }

    def splits(
        self,
        video_path_list,
        offset=DEFUALT["offset"],
        duration=DEFUALT["duration"],
        n_frames=DEFUALT["n_frames"],
        frame_size=DEFUALT["frame_size"],
        n_jobs=DEFUALT["n_jobs"],
    ):
        params = []

        for video_path in video_path_list:
            param = {
                "video_path": video_path,
                "offset": offset,
                "duration": duration,
                "n_frames": n_frames,
                "frame_size": frame_size,
                "n_jobs": n_jobs,
            }
            params.append(param)

        parallelize(func=self.split, params=params, n_jobs=n_jobs)
