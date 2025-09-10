import numpy as np

def assignmentStep(ds , centers, k , d):
    dists = np.sum((ds[:, None, :] - centers[None, :, :])**2, axis=2)
    assign = np.argmin(dists, axis=1)
    sums  = np.zeros((k, d), dtype=ds.dtype)
    count = np.zeros(k, dtype=np.int64)
    np.add.at(sums,  assign, ds)
    np.add.at(count, assign, 1)
    return sums, count, assign


def sKmeans(ds, centers, max_iter=30):
    k, d = centers.shape
    for _ in range(max_iter):
        sums, count, assign = assignmentStep(ds, centers, k, d)
        nonempty = count > 0
        centers[nonempty] = sums[nonempty] / count[nonempty, None]
    return assign