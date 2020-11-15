import numpy as np
import pytest

from nengo_spa.algebras.hrr_algebra import HrrAlgebra, HrrSign
from nengo_spa.vector_generation import UnitLengthVectors


def test_is_singleton():
    assert HrrAlgebra() is HrrAlgebra()


@pytest.mark.parametrize(
    "sign,d,roll",
    [
        (HrrSign(1, 0), 15, 0),
        (HrrSign(-1, 0), 15, 0),
        (HrrSign(1, 1), 16, 0),
        (HrrSign(1, -1), 16, 1),
        (HrrSign(-1, 1), 16, 1),
        (HrrSign(-1, -1), 16, 0),
    ],
)
def test_sign_and_abs(sign, d, roll):
    algebra = HrrAlgebra()
    abs_v = algebra.abs(next(UnitLengthVectors(d)))
    v = algebra.bind(sign.to_vector(d), abs_v)
    assert algebra.sign(v) == sign
    assert np.allclose(algebra.abs(v), np.roll(abs_v, 2 * roll))


class TestHrrSign:
    @pytest.mark.parametrize("sign", [HrrSign(1, 1), HrrSign(1, 0)])
    def test_positive(self, sign):
        assert sign.is_positive()
        assert not sign.is_negative()
        assert not sign.is_zero()
        assert not sign.is_indefinite()

    @pytest.mark.parametrize(
        "sign", [HrrSign(1, -1), HrrSign(-1, 1), HrrSign(-1, -1), HrrSign(-1, 0)]
    )
    def test_negative(self, sign):
        assert not sign.is_positive()
        assert sign.is_negative()
        assert not sign.is_zero()
        assert not sign.is_indefinite()

    def test_zero(self):
        sign = HrrSign(0, 0)
        assert not sign.is_positive()
        assert not sign.is_negative()
        assert sign.is_zero()
        assert not sign.is_indefinite()

    @pytest.mark.parametrize(
        "dc_sign, nyquist_sign", [(0, 1), (0, -1), (0.5, 0), (1, 2)]
    )
    def test_invalid(self, dc_sign, nyquist_sign):
        with pytest.raises(ValueError):
            HrrSign(dc_sign, nyquist_sign)

    @pytest.mark.parametrize(
        "sign,expected",
        [
            (HrrSign(0, 0), [0, 0, 0, 0]),
            (HrrSign(1, 1), [1, 0, 0, 0]),
            (HrrSign(1, -1), [0, 1, 0, 0]),
            (HrrSign(-1, 1), [0, -1, 0, 0]),
            (HrrSign(-1, -1), [-1, 0, 0, 0]),
            (HrrSign(0, 0), [0, 0, 0]),
            (HrrSign(1, 0), [1, 0, 0]),
            (HrrSign(-1, 0), [-1, 0, 0]),
        ],
    )
    def test_to_vector(self, sign, expected):
        assert HrrAlgebra().sign(expected) == sign  # sanity check
        assert np.allclose(sign.to_vector(len(expected)), expected)

    @pytest.mark.parametrize("sign,d", [(HrrSign(1, 0), 4), (HrrSign(1, 1), 3)])
    def test_invalid_to_vector(self, sign, d):
        with pytest.raises(ValueError):
            sign.to_vector(d)

    def test_equality(self):
        assert HrrSign(1, 1) == HrrSign(1, 1)
        assert HrrSign(-1, 1) == HrrSign(-1, 1)
        assert HrrSign(0, 0) == HrrSign(0, 0)
        assert HrrSign(1, -1) == HrrSign(1, -1)

        assert HrrSign(1, 1) != HrrSign(1, -1)
        assert HrrSign(1, 1) != HrrSign(0, 0)

        assert HrrSign(1, 1) != (1, 1)
