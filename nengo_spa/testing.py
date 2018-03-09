"""Support for unit testing SPA models."""

import numpy as np


def sp_close(
        t, data, target_sp, skip=0., duration=None, atol=0.2,
        normalized=False):
    """Test that the RMSE to a Semantic Pointer is below threshold.

    Parameters
    ----------
    t : (T,) array_like
        Time values for data in seconds, usually obtained with
        `nengo.Simulator.trange`.
    data : (T, D) array_like
        Simulation data for *T* timesteps and *D* dimensions.
    target_sp : SemanticPointer
        Target Semantic Pointer.
    skip : float, optional
        Amount of seconds to ignore at the beginning of the data.
    duration : float, optional
        Amount of seconds to consider after the skipped portion.
    atol : float, optional
        Absolute tolerated RMSE.
    normalize : bool, optional
        Whether to normalize the simulation data to unit length in each
        timestep.

    Returns
    -------
    bool
        *True* if *atol* is not exceeded during the considered time interval.
    """
    if duration is None:
        duration = np.max(t) - skip
    actual = data[(skip < t) & (t <= skip + duration)]
    expected = target_sp
    if normalized:
        actual /= np.expand_dims(np.linalg.norm(actual, axis=1), 1)
        expected = expected.normalized()
    return np.all(np.sqrt(np.sum(
        np.square(actual - expected.v), axis=1)) < atol)
