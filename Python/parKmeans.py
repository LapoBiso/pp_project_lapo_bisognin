import numpy as np
from joblib import Parallel, delayed
import time

def assignmentStep(ds: np.ndarray , centers: np.ndarray, k: int , d: int):
    dists_c = np.sum((ds[:, None, :] - centers[None, :, :])**2, axis=2)
    assign_c = np.argmin(dists_c, axis=1)
    sums_c  = np.zeros((k, d), dtype=ds.dtype)
    count_c = np.zeros(k, dtype=np.int64)
    np.add.at(sums_c,  assign_c, ds)
    np.add.at(count_c, assign_c, 1)
    return sums_c, count_c, assign_c

def pKmeans(ds:np.ndarray, centers: np.ndarray, n_workers=8, max_iter=30):
    with Parallel(n_jobs=n_workers,  backend="threading") as pool:
        k, d = centers.shape
        chunks = np.array_split(ds, n_workers*8)
        for _ in range(max_iter):
            results = pool(delayed(assignmentStep)(c, centers, k , d) for c in chunks)
            sums  = np.sum([r[0] for r in results], axis=0)
            count = np.sum([r[1] for r in results], axis=0)
            nonempty = count > 0
            centers[nonempty] = sums[nonempty] / count[nonempty, None]
        assign = np.concatenate([r[2] for r in results])
    return assign
