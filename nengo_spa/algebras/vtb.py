import nengo
from nengo.utils.compat import range
import numpy as np

from nengo_spa.algebras.base import AbstractAlgebra
from nengo_spa.networks.vtb import VTB


class _VtbAlgebra(AbstractAlgebra):
    """Circular convolution algebra.

    Uses element-wise addition for superposition, circular convolution for
    binding with an approximate inverse.
    """

    def is_valid_dimensionality(self, d):
        if d < 1:
            return False
        sub_d = np.sqrt(d)
        return sub_d * sub_d == d

    def _get_sub_d(self, d):
        sub_d = int(np.sqrt(d))
        if sub_d * sub_d != d:
            raise ValueError("Vector dimensionality must be a square number.")
        return sub_d

    def make_unitary(self, v):
        sub_d = self._get_sub_d(len(v))
        m = np.array(v.reshape((sub_d, sub_d)))
        for i in range(1, sub_d):
            y = -np.dot(m[:i, i:], m[i, i:])
            A = m[:i, :i]
            m[i, :i] = np.linalg.solve(A, y)
        m /= np.linalg.norm(m, axis=1)[:, None]
        m /= np.sqrt(sub_d)
        return m.flatten()

    def superpose(self, a, b):
        return a + b

    def bind(self, a, b):
        d = len(a)
        if len(b) != d:
            raise ValueError("Inputs must have same length.")
        m = self.get_binding_matrix(b)
        return np.dot(m, a)

    def invert(self, v):
        sub_d = self._get_sub_d(len(v))
        return v.reshape((sub_d, sub_d)).T.flatten()

    def get_binding_matrix(self, v, swap_inputs=False):
        sub_d = self._get_sub_d(len(v))
        m = np.sqrt(sub_d) * np.kron(
            np.eye(sub_d), v.reshape((sub_d, sub_d)))
        if swap_inputs:
            m = np.dot(self.get_swapping_matrix(len(v)), m)
        return m

    def get_swapping_matrix(self, d):
        sub_d = self._get_sub_d(d)
        m = np.zeros((d, d))
        for i in range(d):
            j = i // sub_d + sub_d * (i % sub_d)
            m[i, j] = 1.
        return m

    def get_inversion_matrix(self, d):
        sub_d = self._get_sub_d(d)
        m = np.zeros((d, d))
        for i in range(d):
            j = sub_d * i
            m[j % d + j // d, i] = 1.
        return m

    def implement_superposition(self, n_neurons_per_d, d, n):
        node = nengo.Node(size_in=d)
        return node, n * (node,), node

    def implement_binding(self, n_neurons_per_d, d, unbind_left, unbind_right):
        net = VTB(n_neurons_per_d, d, unbind_left, unbind_right)
        return net, (net.input_left, net.input_right), net.output

    def absorbing_element(self, d):
        raise NotImplementedError(
            "VtbAlgebra does not have any absorbing elements.")

    def identity_element(self, d):
        sub_d = self._get_sub_d(d)
        return (np.eye(sub_d) / d**0.25).flatten()

    def zero_element(self, d):
        return np.zeros(d)


VtbAlgebra = _VtbAlgebra()
