import numpy as np

from nengo_spa.algebras.vtb_algebra import VtbAlgebra
from nengo_spa.semantic_pointer import SemanticPointer
from nengo_spa.vector_generation import UnitLengthVectors


def test_is_singleton():
    assert VtbAlgebra() is VtbAlgebra()


def test_is_valid_dimensionality():
    assert not VtbAlgebra().is_valid_dimensionality(-1)
    assert not VtbAlgebra().is_valid_dimensionality(0)
    assert not VtbAlgebra().is_valid_dimensionality(15)
    assert not VtbAlgebra().is_valid_dimensionality(24)
    assert VtbAlgebra().is_valid_dimensionality(1)
    assert VtbAlgebra().is_valid_dimensionality(16)
    assert VtbAlgebra().is_valid_dimensionality(25)


def test_get_swapping_matrix(rng):
    gen = UnitLengthVectors(64, rng=rng)
    a = SemanticPointer(next(gen), algebra=VtbAlgebra()).v
    b = SemanticPointer(next(gen), algebra=VtbAlgebra()).v

    m = VtbAlgebra().get_swapping_matrix(64)
    assert np.allclose(VtbAlgebra().bind(a, b), np.dot(m, VtbAlgebra().bind(b, a)))
