import nengo
from nengo.utils.compat import range
import numpy as np

from nengo_spa.networks.circularconvolution import CircularConvolution


class CircularConvolutionAlgebra(object):
    """Circular convolution algebra.

    Uses element-wise addition for superposition, circular convolution for
    binding with an approximate inverse.
    """

    @classmethod
    def is_valid_dimensionality(cls, d):
        return d > 0

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

    @classmethod
    def get_inversion_matrix(cls, d):
        return np.eye(d)[-np.arange(d)]

    @classmethod
    def implement_superposition(cls, n_neurons_per_d, d, n):
        node = nengo.Node(size_in=d)
        return node, n * (node,), node

    @classmethod
    def implement_binding(cls, n_neurons_per_d, d, invert_a, invert_b):
        net = CircularConvolution(n_neurons_per_d, d, invert_a, invert_b)
        return net, (net.input_a, net.input_b), net.output

    @classmethod
    def absorbing_element(cls, d):
        return np.ones(d) / np.sqrt(d)

    @classmethod
    def identity_element(cls, d):
        data = np.zeros(d)
        data[0] = 1.
        return data

    @classmethod
    def zero_element(cls, d):
        return np.zeros(d)
