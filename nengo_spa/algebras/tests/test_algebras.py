import numpy as np
import pytest

from nengo_spa.pointer import SemanticPointer


@pytest.mark.parametrize('d', [65, 100])
def test_make_unitary(algebra, d, rng):
    a = SemanticPointer(d, rng=rng).v
    b = algebra.make_unitary(a)
    for i in range(3):
        assert np.allclose(1, np.linalg.norm(a))
        assert np.allclose(1, np.linalg.norm(b))
        a = algebra.bind(a, b)
        assert np.allclose(1, np.linalg.norm(a))


def test_superpose(algebra, rng):
    a = SemanticPointer(10, rng).v
    b = np.array(SemanticPointer(10, rng).v)
    # Orthogonalize
    b -= np.dot(a, b) * a
    b /= np.linalg.norm(b)

    r = algebra.superpose(a, b)
    for v in (a, b):
        assert np.dot(v, r / np.linalg.norm(r)) > 0.6


@pytest.mark.parametrize('d', [64, 65])
def test_binding_and_invert(algebra, d, rng):
    a = SemanticPointer(d, rng=rng).v
    b = SemanticPointer(d, rng=rng).v
    bound = algebra.bind(a, b)
    r = algebra.bind(bound, algebra.invert(b))
    for v in (a, b):
        assert np.dot(v, bound / np.linalg.norm(b)) < 0.7
    assert np.dot(a, r / np.linalg.norm(r)) > 0.6


def test_get_binding_matrix(algebra, rng):
    a = SemanticPointer(50, rng).v
    b = SemanticPointer(50, rng).v

    m = algebra.get_binding_matrix(b)

    assert np.allclose(algebra.bind(a, b), np.dot(m, a))


def test_absorbing_element(algebra, rng):
    a = SemanticPointer(64, rng).v
    p = algebra.absorbing_element(64)
    r = algebra.bind(a, p)
    r /= np.linalg.norm(r)
    assert np.allclose(p, r) or np.allclose(p, -r)


def test_identity_element(algebra, rng):
    a = SemanticPointer(64, rng).v
    p = algebra.identity_element(64)
    assert np.allclose(algebra.bind(a, p), a)


def test_zero_element(algebra, rng):
    a = SemanticPointer(64, rng).v
    p = algebra.zero_element(64)
    assert np.all(p == 0.)
    assert np.allclose(algebra.bind(a, p), 0.)
