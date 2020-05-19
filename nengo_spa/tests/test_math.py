import numpy as np
import pytest

from nengo_spa.math import prob_cleanup


def test_prob_cleanup():
    pytest.importorskip("scipy")

    assert 1.0 > prob_cleanup(0.7, 64, 10000) > 0.9999
    assert 0.9999 > prob_cleanup(0.6, 64, 10000) > 0.999
    assert 0.99 > prob_cleanup(0.5, 64, 1000) > 0.9

    assert 0.999 > prob_cleanup(0.4, 128, 1000) > 0.997
    assert 0.99 > prob_cleanup(0.4, 128, 10000) > 0.97
    assert 0.9 > prob_cleanup(0.4, 128, 100000) > 0.8
    assert prob_cleanup(0.4, 128, np.inf) == 0.0

    with pytest.raises(ArithmeticError):
        prob_cleanup(0.2, 2025, np.inf)
