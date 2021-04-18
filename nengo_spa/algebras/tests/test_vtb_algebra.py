import numpy as np
import pytest

from nengo_spa.algebras.vtb_algebra import VtbAlgebra, VtbProperties, VtbSign
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


@pytest.mark.parametrize(
    "sign",
    [VtbSign(-1), VtbSign(1)],
)
def test_sign_and_abs(sign):
    pytest.importorskip("scipy")
    d = 16
    algebra = VtbAlgebra()
    abs_v = algebra.create_vector(d, {VtbProperties.POSITIVE})
    v = algebra.bind(sign.to_vector(d), abs_v)
    assert algebra.sign(v) == sign
    assert np.allclose(algebra.abs(v), abs_v)


def test_sign_repr():
    sign = VtbSign(-1)
    assert eval(repr(sign)) == sign


def test_create_positive_vector(rng):
    pytest.importorskip("scipy")
    algebra = VtbAlgebra()
    v = algebra.create_vector(16, {VtbProperties.POSITIVE}, rng=rng)
    assert len(v) == 16
    assert algebra.sign(v).is_positive()


def test_create_unitary_vector(rng):
    algebra = VtbAlgebra()
    v = algebra.create_vector(16, {VtbProperties.UNITARY}, rng=rng)
    assert len(v) == 16
    assert np.allclose(algebra.make_unitary(v), v)


@pytest.mark.filterwarnings("ignore:.*only positive unitary vector")
def test_create_positive_unitary_vector(rng):
    algebra = VtbAlgebra()
    v = algebra.create_vector(16, {VtbProperties.UNITARY, VtbProperties.POSITIVE})
    assert len(v) == 16
    assert algebra.sign(v).is_positive()
    assert np.allclose(algebra.make_unitary(v), v)


def test_create_vector_with_invalid_property():
    with pytest.raises(ValueError):
        VtbAlgebra().create_vector(16, "foo")


def test_fractional_binding_power_of_non_positive_vector_raises(rng):
    pytest.importorskip("scipy")
    algebra = VtbAlgebra()
    v = algebra.bind(
        algebra.create_vector(16, {VtbProperties.POSITIVE}, rng=rng),
        VtbSign(-1).to_vector(16),
    )
    assert algebra.sign(v).is_negative()
    with pytest.raises(ValueError):
        algebra.binding_power(v, 0.5)


def test_fractional_binding_power_with_positive_exponent_lower_1(rng):
    pytest.importorskip("scipy")

    algebra = VtbAlgebra()
    v = algebra.create_vector(16, {VtbProperties.POSITIVE}, rng=rng)
    assert np.allclose(v, algebra.binding_power(algebra.binding_power(v, 2.5), 1 / 2.5))
