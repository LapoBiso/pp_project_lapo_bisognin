from dataclasses import dataclass, field
import numpy as np

def dataGenerator(n: int, c: int, d: int):
    ds = np.empty((n*c,d), dtype=np.float64)
    rng = np.random.default_rng()
    for i in range(c):
        start = i * n
        end = (i + 1) * n
        ds[start:end, :] = rng.normal(loc=4 * i, scale=2, size=(n, d))
    return ds

def initCenters(ds: np.ndarray, c: int, d: int):
    rng = np.random.default_rng()
    centers = np.empty((c,d), dtype=np.float64)
    index: int = 0
    for i in range(c):
        index = rng.integers(low=0, high=ds.shape[0])
        centers[i,:] = ds[index,:]
    return centers
