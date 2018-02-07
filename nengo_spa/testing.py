import numpy as np


def sp_close(
        t, data, target_sp, skip=0., duration=None, atol=0.2,
        normalized=False):
    if duration is None:
        duration = np.max(t) - skip
    actual = data[(skip < t) & (t <= skip + duration)]
    expected = target_sp
    if normalized:
        actual /= np.expand_dims(np.linalg.norm(actual, axis=1), 1)
        expected = expected.normalized()
    return np.all(np.sqrt(np.sum(
        np.square(actual - expected.v), axis=1)) < atol)
