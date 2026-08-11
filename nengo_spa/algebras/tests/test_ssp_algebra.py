import numpy as np
import pytest

from nengo_spa.algebras.ssp_algebra import SspAlgebra, SspProperties
from nengo_spa.algebras.hrr_algebra import HrrSign
from nengo_spa.vector_generation import UnitLengthVectors


def test_is_singleton():
    assert SspAlgebra() is SspAlgebra()


@pytest.mark.parametrize(
    "sign,d",
    [
        (HrrSign(1, 0), 15),
        (HrrSign(-1, 0), 15),
        (HrrSign(1, 1), 16),
        (HrrSign(1, -1), 16),
        (HrrSign(-1, 1), 16),
        (HrrSign(-1, -1), 16),
    ],
)
def test_sign_and_abs(sign, d):
    algebra = SspAlgebra()
    abs_v = algebra.abs(next(UnitLengthVectors(d)))
    v = algebra.bind(sign.to_vector(d), abs_v)
    assert algebra.sign(v) == sign
    assert np.allclose(algebra.abs(v), abs_v)


def test_create_positive_vector(rng):
    algebra = SspAlgebra()
    v = algebra.create_vector(16, {SspProperties.POSITIVE}, rng=rng)
    assert len(v) == 16
    assert algebra.sign(v).is_positive()


def test_create_unitary_vector(rng):
    algebra = SspAlgebra()
    v = algebra.create_vector(16, {SspProperties.UNITARY}, rng=rng)
    assert len(v) == 16
    assert np.allclose(algebra.make_unitary(v), v)


def test_create_positive_unitary_vector(rng):
    algebra = SspAlgebra()
    v = algebra.create_vector(16, {SspProperties.UNITARY, SspProperties.POSITIVE})
    assert len(v) == 16
    assert algebra.sign(v).is_positive()
    assert np.allclose(algebra.make_unitary(v), v)


def test_create_vector_with_invalid_property():
    with pytest.raises(ValueError):
        SspAlgebra().create_vector(16, "foo")


def test_additional_integer_binding_power_properties(rng):
    algebra = SspAlgebra()
    v = algebra.create_vector(16, {SspProperties.UNITARY}, rng=rng)

    assert np.allclose(
        algebra.bind(algebra.binding_power(v, 2), algebra.binding_power(v, 3)),
        algebra.binding_power(v, 5),
    )
    assert np.allclose(
        algebra.binding_power(algebra.binding_power(v, 2), 3),
        algebra.binding_power(v, 6),
    )

    assert np.allclose(
        algebra.bind(algebra.binding_power(v, 2), algebra.binding_power(v, -4)),
        algebra.binding_power(v, -2),
    )
    assert np.allclose(
        algebra.binding_power(algebra.binding_power(v, -2), 3),
        algebra.binding_power(v, -6),
    )


@pytest.mark.filterwarnings("ignore:.*only positive unitary vector")
def test_additional_fractional_binding_power_properties(rng):
    algebra = SspAlgebra()
    v = algebra.create_vector(
        16, {SspProperties.POSITIVE, SspProperties.UNITARY}, rng=rng
    )

    assert np.allclose(
        algebra.bind(algebra.binding_power(v, 2.2), algebra.binding_power(v, 3.3)),
        algebra.binding_power(v, 5.5),
    )
    assert np.allclose(
        algebra.bind(algebra.binding_power(v, 2.2), algebra.binding_power(v, -4.4)),
        algebra.binding_power(v, -2.2),
    )

    v = algebra.create_vector(16, {SspProperties.POSITIVE}, rng=rng)

    assert np.allclose(
        algebra.bind(algebra.binding_power(v, 2.2), algebra.binding_power(v, 3.3)),
        algebra.binding_power(v, 5.5),
    )


def test_fractional_binding_power_of_non_positive_vector_raises(rng):
    algebra = SspAlgebra()
    v = algebra.bind(
        algebra.create_vector(16, {SspProperties.POSITIVE}, rng=rng),
        HrrSign(-1, 1).to_vector(16),
    )
    assert algebra.sign(v).is_negative()
    with pytest.raises(ValueError):
        algebra.binding_power(v, 0.5)
