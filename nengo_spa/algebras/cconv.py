import nengo
from nengo.utils.compat import range
import numpy as np

from nengo_spa.algebras.base import AbstractAlgebra
from nengo_spa.networks.circularconvolution import CircularConvolution


class _CircularConvolutionAlgebra(AbstractAlgebra):
    """Circular convolution algebra.

    Uses element-wise addition for superposition, circular convolution for
    binding with an approximate inverse.
    """

    def is_valid_dimensionality(self, d):
        return d > 0

    def make_unitary(self, v):
        fft_val = np.fft.fft(v)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
        fft_unit = fft_val / fft_norms
        return np.array((np.fft.ifft(fft_unit, n=len(v))).real)

    def superpose(self, a, b):
        return a + b

    def bind(self, a, b):
        n = len(a)
        if len(b) != n:
            raise ValueError("Inputs must have same length.")
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=n)

    def invert(self, v):
        return v[-np.arange(len(v))]

    def get_binding_matrix(self, v, swap_inputs=False):
        D = len(v)
        T = []
        for i in range(D):
            T.append([v[(i - j) % D] for j in range(D)])
        return np.array(T)

    def get_inversion_matrix(self, d):
        return np.eye(d)[-np.arange(d)]

    def implement_superposition(self, n_neurons_per_d, d, n):
        node = nengo.Node(size_in=d)
        return node, n * (node,), node

    def implement_binding(self, n_neurons_per_d, d, unbind_left, unbind_right):
        net = CircularConvolution(
            n_neurons_per_d, d, unbind_left, unbind_right)
        return net, (net.input_a, net.input_b), net.output

    def absorbing_element(self, d):
        return np.ones(d) / np.sqrt(d)

    def identity_element(self, d):
        data = np.zeros(d)
        data[0] = 1.
        return data

    def zero_element(self, d):
        return np.zeros(d)


CircularConvolutionAlgebra = _CircularConvolutionAlgebra()
