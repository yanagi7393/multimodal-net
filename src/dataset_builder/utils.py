from joblib import Parallel, delayed
from typing import Callable, List, Dict


def parallelize(func: Callable, params: List[Dict], n_jobs: int = 64) -> List:
    outputs = []

    for param in params:

        outputs.append(delayed(func)(**param))

    outputs = Parallel(n_jobs=n_jobs, verbose=1)(outputs)

    return outputs
