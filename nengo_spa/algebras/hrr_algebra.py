import nengo
from nengo.utils.compat import range
import numpy as np

from nengo_spa.algebras.base import AbstractAlgebra
from nengo_spa.networks.circularconvolution import CircularConvolution


class HrrAlgebra(AbstractAlgebra):
    r"""Holographic Reduced Representations (HRRs) algebra.

    Uses element-wise addition for superposition, circular convolution for
    binding with an approximate inverse.

    The circular convolution :math:`c` of vectors :math:`a` and :math:`b`
    is given by

    .. math:: c[i] = \sum_j a[j] b[i - j]

    where negative indices on :math:`b` wrap around to the end of the vector.

    This computation can also be done in the Fourier domain,

    .. math:: c = DFT^{-1} ( DFT(a) \odot DFT(b) )

    where :math:`DFT` is the Discrete Fourier Transform operator, and
    :math:`DFT^{-1}` is its inverse.

    Circular convolution as a binding operation is associative, commutative,
    distributive.

    More information on circular convolution as a binding operation can be
    found in [plate2003]_.

    .. [plate2003] Plate, Tony A. Holographic Reduced Representation:
       Distributed Representation for Cognitive Structures. Stanford, CA: CSLI
       Publications, 2003.
    """

    _instance = None

    def __new__(cls):
        if type(cls._instance) is not cls:
            cls._instance = super(HrrAlgebra, cls).__new__(cls)
        return cls._instance

    def is_valid_dimensionality(self, d):
        """Checks whether *d* is a valid vector dimensionality.

        For circular convolution all positive numbers are valid
        dimensionalities.

        Parameters
        ----------
        d : int
            Dimensionality

        Returns
        -------
        bool
            *True*, if *d* is a valid vector dimensionality for the use with
            the algebra.
        """
        return d > 0

    def make_unitary(self, v):
        fft_val = np.fft.fft(v)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
        invalid = fft_norms <= 0.
        fft_val[invalid] = 1.
        fft_norms[invalid] = 1.
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
        r"""Return the standard absorbing element of dimensionality *d*.

        An absorbing element will produce a scaled version of itself when bound
        to another vector. The standard absorbing element is the absorbing
        element with norm 1.

        The absorbing element for circular convolution is the vector
        :math:`(1, 1, \dots, 1)^{\top} / \sqrt{d}`.

        Parameters
        ----------
        d : int
            Vector dimensionality.

        Returns
        -------
        (d,) ndarray
            Standard absorbing element.
        """
        return np.ones(d) / np.sqrt(d)

    def identity_element(self, d):
        r"""Return the identity element of dimensionality *d*.

        The identity does not change the vector it is bound to.

        The identity element for circular convolution is the vector
        :math:`(1, 0, \dots, 0)^{\top}`.

        Parameters
        ----------
        d : int
            Vector dimensionality.

        Returns
        -------
        (d,) ndarray
            Identity element.
        """
        data = np.zeros(d)
        data[0] = 1.
        return data

    def zero_element(self, d):
        """Return the zero element of dimensionality *d*.

        The zero element produces itself when bound to a different vector.
        For circular convolution this is the zero vector.

        Parameters
        ----------
        d : int
            Vector dimensionality.

        Returns
        -------
        (d,) ndarray
            Zero element.
        """
        return np.zeros(d)
