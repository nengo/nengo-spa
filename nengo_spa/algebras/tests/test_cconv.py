from nengo_spa.algebras.cconv import CircularConvolutionAlgebra


def test_is_singleton():
    assert CircularConvolutionAlgebra() is CircularConvolutionAlgebra()
