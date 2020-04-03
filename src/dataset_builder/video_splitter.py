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
    "sr": 16000,
    "mutate_end_time": True,
}


class VideoSplitter:
    def split(
        self,
        video_path,
        offset=DEFUALT["offset"],
        duration=DEFUALT["duration"],
        n_frames=DEFUALT["n_frames"],
        frame_size=DEFUALT["frame_size"],
        mutate_end_time=DEFUALT["mutate_end_time"],
    ):
        if not os.path.isfile(video_path):
            raise ValueError(f"FileNotExisted:{video_path}")

        video_name = Path(video_path).stem

        # load video
        video = VideoFileClip(video_path)

        # if whole duration size is smaller than duration, just pass.
        if video.duration < duration:
            return None

        end_time = offset + duration

        if mutate_end_time is False:
            if end_time > math.floor(video.duration):
                return None

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

        frames = []
        for idx, t in enumerate(pick_frame_times):
            try:
                frame = video.get_frame(t)
                frames.append(frame)
                continue
            except:
                try:
                    frame = video.get_frame(time_ranges[-1])
                    frames.append(frame)
                except:
                    return None

        # audio part
        audios = [video.audio.to_soundarray(fps=DEFUALT["sr"])]

        video.close()

        return {
            "frames": frames,
            "audios": audios,
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
            }
            params.append(param)

        outputs = parallelize(func=self.split, params=params, n_jobs=n_jobs)
        return outputs
