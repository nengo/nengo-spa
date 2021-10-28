import numpy as np

from nengo_spa.typechecks import is_integer, is_number


def test_is_integer():
    assert is_integer(1)
    assert is_integer(np.array(1))
    assert not is_integer(np.array([1]))
    assert not is_integer(1.0)
    assert not is_integer(np.array(1.0))
    assert not is_integer(np.array([1.0]))
    assert not is_integer(object())


def test_is_number():
    assert is_number(1)
    assert is_number(np.array(1))
    assert is_number(1.0)
    assert is_number(np.array(1.0))
    assert not is_number(np.array([1.0]))
    assert not is_number(np.array([1]))
    assert not is_number(object())
