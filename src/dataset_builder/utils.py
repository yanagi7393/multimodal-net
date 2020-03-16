from joblib import Parallel, delayed
from typing import Callable, List, Dict
import librosa


def parallelize(func: Callable, params: List[Dict], n_jobs: int = 64) -> List:
    outputs = []

    for param in params:

        outputs.append(delayed(func)(**param))

    outputs = Parallel(n_jobs=n_jobs, verbose=1)(outputs)

    return outputs


def load_audio(
    path, offset=0.0, duration=None
):  # offset and duration is already handled on VideoSplitter side.
    return librosa.load(path, offset=offset, duration=duration)[0]
