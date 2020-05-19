import numpy as np
import pytest

from nengo_spa.algebras.base import AbstractAlgebra
from nengo_spa.vector_generation import UnitLengthVectors


@pytest.mark.parametrize("d", [16, 25])
def test_make_unitary(algebra, d, rng):
    a = next(UnitLengthVectors(d, rng=rng))
    b = algebra.make_unitary(a)
    for i in range(3):
        assert np.allclose(1, np.linalg.norm(a))
        assert np.allclose(1, np.linalg.norm(b))
        a = algebra.bind(a, b)
        assert np.allclose(1, np.linalg.norm(a))


def test_superpose(algebra, rng):
    gen = UnitLengthVectors(16, rng=rng)
    a = next(gen)
    b = next(gen)
    # Orthogonalize
    b -= np.dot(a, b) * a
    b /= np.linalg.norm(b)

    r = algebra.superpose(a, b)
    for v in (a, b):
        assert np.dot(v, r / np.linalg.norm(r)) > 0.6


@pytest.mark.parametrize("d", [25, 36])
def test_binding_and_invert(algebra, d, rng):
    dissimilarity_passed = 0
    unbinding_passed = 0
    for i in range(10):
        gen = UnitLengthVectors(d, rng=rng)
        a = next(gen)
        b = next(gen)
        bound = algebra.bind(a, b)
        r = algebra.bind(bound, algebra.invert(b))
        for v in (a, b):
            dissimilarity_passed += np.dot(v, bound / np.linalg.norm(b)) < 0.7
        unbinding_passed += np.dot(a, r / np.linalg.norm(r)) > 0.6
    assert dissimilarity_passed >= 2 * 8
    assert unbinding_passed >= 8


def test_dimensionality_mismatch_exception(algebra):
    with pytest.raises(ValueError):
        algebra.bind(np.ones(16), np.ones(25))
    with pytest.raises(ValueError):
        algebra.superpose(np.ones(16), np.ones(25))


def test_get_binding_matrix(algebra, rng):
    gen = UnitLengthVectors(16, rng=rng)
    a = next(gen)
    b = next(gen)

    m = algebra.get_binding_matrix(b)

    assert np.allclose(algebra.bind(a, b), np.dot(m, a))


def test_get_inversion_matrix(algebra, rng):
    a = next(UnitLengthVectors(16, rng=rng))
    m = algebra.get_inversion_matrix(16)
    assert np.allclose(algebra.invert(a), np.dot(m, a))


def test_absorbing_element(algebra, rng):
    a = next(UnitLengthVectors(16, rng=rng))
    try:
        p = algebra.absorbing_element(16)
        r = algebra.bind(a, p)
        r /= np.linalg.norm(r)
        assert np.allclose(p, r) or np.allclose(p, -r)
    except NotImplementedError:
        pass


def test_identity_element(algebra, rng):
    a = next(UnitLengthVectors(16, rng=rng))
    p = algebra.identity_element(16)
    assert np.allclose(algebra.bind(a, p), a)


def test_zero_element(algebra, rng):
    a = next(UnitLengthVectors(16, rng=rng))
    p = algebra.zero_element(16)
    assert np.all(p == 0.0)
    assert np.allclose(algebra.bind(a, p), 0.0)


def test_isinstance_check(algebra):
    assert isinstance(algebra, AbstractAlgebra)


class DummyAlgebra:
    def is_valid_dimensionality(self, d):
        pass

    def make_unitary(self, v):
        pass

    def superpose(self, a, b):
        pass

    def bind(self, a, b):
        pass

    def invert(self, v):
        pass

    def get_binding_matrix(self, v, swap_inputs=False):
        pass

    def get_inversion_matrix(self, d):
        pass

    def implement_superposition(self, n_neurons_per_d, d, n):
        pass

    def implement_binding(self, n_neurons_per_d, d, unbind_left, unbind_right):
        pass

    def absorbing_element(self, d):
        pass

    def identity_element(self, d):
        pass

    def zero_element(self, d):
        pass


@pytest.mark.filterwarnings("ignore:.*do not rely on pure duck-typing")
def test_isinstance_ducktyping_check():
    assert isinstance(DummyAlgebra(), AbstractAlgebra)
