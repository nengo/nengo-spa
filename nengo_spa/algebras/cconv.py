from nengo.utils.compat import range
import numpy as np


class CircularConvolutionAlgebra(object):
    """Circular convolution algebra.

    Uses element-wise addition for superposition, circular convolution for
    binding with an approximate inverse.
    """

    @classmethod
    def make_unitary(cls, v):
        fft_val = np.fft.fft(v)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
        fft_unit = fft_val / fft_norms
        return np.array((np.fft.ifft(fft_unit, n=len(v))).real)

    @classmethod
    def superpose(cls, a, b):
        return a + b

    @classmethod
    def bind(cls, a, b):
        n = len(a)
        if len(b) != n:
            raise ValueError("Inputs must have same length.")
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=n)

    @classmethod
    def invert(cls, v):
        return v[-np.arange(len(v))]

    @classmethod
    def get_binding_matrix(cls, v):
        D = len(v)
        T = []
        for i in range(D):
            T.append([v[(i - j) % D] for j in range(D)])
        return np.array(T)
