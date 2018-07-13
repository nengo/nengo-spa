import nengo
from nengo.dists import CosineSimilarity
from nengo.utils.compat import range
import numpy as np

from nengo_spa.networks.matrix_multiplication import MatrixMult


class VtbAlgebra(object):
    """Circular convolution algebra.

    Uses element-wise addition for superposition, circular convolution for
    binding with an approximate inverse.
    """

    @classmethod
    def is_valid_dimensionality(cls, d):
        if d < 1:
            return False
        sub_d = np.sqrt(d)
        return sub_d * sub_d == d

    @classmethod
    def _get_sub_d(cls, d):
        sub_d = int(np.sqrt(d))
        if sub_d * sub_d != d:
            raise ValueError("Vector dimensionality must be a square number.")
        return sub_d

    @classmethod
    def make_unitary(cls, v):
        sub_d = cls._get_sub_d(len(v))
        m = np.array(v.reshape((sub_d, sub_d)))
        for i in range(1, sub_d):
            y = -np.dot(m[:i, i:], m[i, i:])
            A = m[:i, :i]
            m[i, :i] = np.linalg.solve(A, y)
        m /= np.linalg.norm(m, axis=1)[:, None]
        m /= np.sqrt(sub_d)
        return m.flatten()

    @classmethod
    def superpose(cls, a, b):
        return a + b

    @classmethod
    def bind(cls, a, b):
        d = len(a)
        if len(b) != d:
            raise ValueError("Inputs must have same length.")
        sub_d = cls._get_sub_d(d)
        m = cls.get_binding_matrix(b)
        return np.dot(m, a)

    @classmethod
    def invert(cls, v):
        sub_d = cls._get_sub_d(len(v))
        return v.reshape((sub_d, sub_d)).T.flatten()

    @classmethod
    def get_binding_matrix(cls, v, swap_inputs=False):
        sub_d = cls._get_sub_d(len(v))
        m = np.sqrt(sub_d) * np.kron(
            np.eye(sub_d), v.reshape((sub_d, sub_d)))
        if swap_inputs:
            m = np.dot(cls.get_swapping_matrix(len(v)), m)
        return m

    @classmethod
    def get_swapping_matrix(cls, d):
        sub_d = cls._get_sub_d(d)
        m = np.zeros((d, d))
        for i in range(d):
            j = i // sub_d + sub_d * (i % sub_d)
            m[i, j] = 1.
        return m

    @classmethod
    def get_inversion_matrix(cls, d):
        sub_d = cls._get_sub_d(d)
        m = np.zeros((d, d))
        for i in range(d):
            j = sub_d * i
            m[j % d + j // d, i] = 1.
        return m

    @classmethod
    def implement_superposition(cls, n_neurons_per_d, d, n):
        node = nengo.Node(size_in=d)
        return node, n * (node,), node

    @classmethod
    def implement_binding(cls, n_neurons_per_d, d, unbind_left, unbind_right):
        sub_d = cls._get_sub_d(d)
        shape_left = (sub_d, sub_d)
        shape_right = (sub_d, 1)

        with nengo.Network() as net:
            net.input_left = nengo.Node(size_in=d)
            net.input_right = nengo.Node(size_in=d)
            net.output = nengo.Node(size_in=d)

            net.mat = nengo.Node(size_in=d)
            net.vec = nengo.Node(size_in=d)

            if unbind_left and unbind_right:
                raise ValueError("Cannot unbind both sides at the same time.")
            elif unbind_left:
                nengo.Connection(
                    net.input_left, net.mat,
                    transform=cls.get_inversion_matrix(d), synapse=None)
                nengo.Connection(
                    net.input_right, net.vec,
                    transform=cls.get_swapping_matrix(d), synapse=None)
            else:
                nengo.Connection(net.input_left, net.vec, synapse=None)
                if unbind_right:
                    tr = cls.get_inversion_matrix(d)
                else:
                    tr = 1.
                nengo.Connection(
                    net.input_right, net.mat, transform=tr, synapse=None)

            with nengo.Config(nengo.Ensemble) as cfg:
                cfg[nengo.Ensemble].intercepts = CosineSimilarity(d+2)
                cfg[nengo.Ensemble].eval_points = CosineSimilarity(d+2)
                net.matmuls = [
                    MatrixMult(n_neurons_per_d, shape_left, shape_right)
                    for i in range(sub_d)]

            for i in range(sub_d):
                mm = net.matmuls[i]
                sl = slice(i * sub_d, (i + 1) * sub_d)
                nengo.Connection(net.mat, mm.input_left, synapse=None)
                nengo.Connection(net.vec[sl], mm.input_right, synapse=None)
                nengo.Connection(
                    mm.output, net.output[sl], transform=np.sqrt(sub_d),
                    synapse=None)
        return net, (net.input_left, net.input_right), net.output

    @classmethod
    def absorbing_element(cls, d):
        raise NotImplementedError(
            "VtbAlgebra does not have any absorbing elements.")

    @classmethod
    def identity_element(cls, d):
        sub_d = cls._get_sub_d(d)
        return (np.eye(sub_d) / d**0.25).flatten()

    @classmethod
    def zero_element(cls, d):
        return np.zeros(d)
