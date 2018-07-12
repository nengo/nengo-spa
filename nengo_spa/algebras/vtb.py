from nengo.utils.compat import range
import numpy as np


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
    def get_binding_matrix(cls, v):
        sub_d = cls._get_sub_d(len(v))
        return np.sqrt(sub_d) * np.kron(
            np.eye(sub_d), v.reshape((sub_d, sub_d)))

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
        raise NotImplementedError()

    @classmethod
    def implement_binding(cls, n_neurons_per_d, d, invert_a, invert_b):
        raise NotImplementedError()

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
