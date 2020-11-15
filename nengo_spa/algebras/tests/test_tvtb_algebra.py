import numpy as np
import pytest

from nengo_spa.algebras.tvtb_algebra import TvtbAlgebra, TvtbSign
from nengo_spa.vector_generation import UnitLengthVectors


def test_is_singleton():
    assert TvtbAlgebra() is TvtbAlgebra()


def test_is_valid_dimensionality():
    assert not TvtbAlgebra().is_valid_dimensionality(-1)
    assert not TvtbAlgebra().is_valid_dimensionality(0)
    assert not TvtbAlgebra().is_valid_dimensionality(15)
    assert not TvtbAlgebra().is_valid_dimensionality(24)
    assert TvtbAlgebra().is_valid_dimensionality(1)
    assert TvtbAlgebra().is_valid_dimensionality(16)
    assert TvtbAlgebra().is_valid_dimensionality(25)


@pytest.mark.parametrize(
    "sign",
    [TvtbSign(-1), TvtbSign(1)],
)
def test_sign_and_abs(sign):
    pytest.importorskip("scipy")
    d = 16
    algebra = TvtbAlgebra()
    base = next(UnitLengthVectors(d))
    # TODO replace the following line with the canonical methods of generating
    # positive definite vectors for the algebra
    abs_v = np.dot(algebra.get_binding_matrix(base).T, base)
    v = algebra.bind(sign.to_vector(d), abs_v)
    assert algebra.sign(v) == sign
    assert np.allclose(algebra.abs(v), abs_v)
