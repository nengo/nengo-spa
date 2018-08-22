from nengo_spa.algebras.hrr_algebra import HrrAlgebra


def test_is_singleton():
    assert HrrAlgebra() is HrrAlgebra()
