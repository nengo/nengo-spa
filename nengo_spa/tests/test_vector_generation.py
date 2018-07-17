import numpy as np

from nengo_spa.algebras import CircularConvolutionAlgebra
from nengo_spa.pointer import SemanticPointer
from nengo_spa.vector_generation import (
    AxisAlignedVectors, ExpectedUnitLengthVectors, OrthonormalVectors,
    UnitLengthVectors, UnitaryVectors)


def test_axis_aligned_pointers():
    for i, p in enumerate(AxisAlignedVectors(5)):
        assert np.all(p == np.eye(5)[i])


def test_unit_length_pointers(rng):
    g = UnitLengthVectors(64, rng)
    for i in range(10):
        assert np.allclose(np.linalg.norm(next(g)), 1.)


def test_unitary_pointers(rng):
    algebra = CircularConvolutionAlgebra()
    g = UnitaryVectors(64, algebra, rng)
    a = SemanticPointer(next(g), algebra)
    b = SemanticPointer(next(g), algebra)
    c = SemanticPointer(next(g), algebra)
    assert np.allclose(a.compare(c), (a * b).compare(c * b))


def test_orthonormal_pointers(rng):
    g = OrthonormalVectors(32, rng)
    vectors = np.array(list(g))
    assert len(vectors) == 32
    assert np.allclose(np.dot(vectors.T, vectors), np.eye(32))


def test_expected_unit_length_vectors(rng):
    g = ExpectedUnitLengthVectors(64, rng)
    assert np.abs(np.mean(
        [np.linalg.norm(next(g)) for i in range(50)]) - 1.) < 0.1
