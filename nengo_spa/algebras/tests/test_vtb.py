import numpy as np

from nengo_spa.algebras.vtb import VtbAlgebra
from nengo_spa.pointer import SemanticPointer


def test_is_singleton():
    assert VtbAlgebra() is VtbAlgebra()


def test_get_swapping_matrix(rng):
    a = SemanticPointer(64, rng, algebra=VtbAlgebra()).v
    b = SemanticPointer(64, rng, algebra=VtbAlgebra()).v

    m = VtbAlgebra().get_swapping_matrix(64)
    assert np.allclose(
        VtbAlgebra().bind(a, b), np.dot(m, VtbAlgebra().bind(b, a)))
