import numpy as np


def sp_close(
        t, data, target_sp, skip=0., duration=None, atol=0.2):
    if duration is None:
        duration = np.max(t) - skip
    return np.all(np.sqrt(np.sum(
        np.square(data[(skip < t) & (t <= skip + duration)] - target_sp.v),
        axis=1)) < atol)
