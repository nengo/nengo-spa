import nengo
import numpy as np
from nengo.dists import CosineSimilarity
from nengo.exceptions import ValidationError

from nengo_spa.networks.matrix_multiplication import MatrixMult


def calc_sub_d(dimensions):
    sub_d = int(np.sqrt(dimensions))
    if sub_d * sub_d != dimensions:
        raise ValidationError("Dimensions must be a square number.", "dimensions")
    return sub_d


def inversion_matrix(dimensions):
    sub_d = calc_sub_d(dimensions)
    m = np.zeros((dimensions, dimensions))
    for i in range(dimensions):
        j = sub_d * i
        m[j % dimensions + j // dimensions, i] = 1.0
    return m


class TVTB(nengo.Network):
    r"""Compute transposed vector-derived transformation binding (TVTB).

    VTB uses elementwise addition for superposition. The binding operation
    :math:`\mathcal{B}(x, y)` is defined as

    .. math::

       \mathcal{B}(x, y) := V_y^T x = \left[\begin{array}{ccc}
           V_y'^T &      0 &      0 \\
              0   & V_y'^T &      0 \\
              0   &      0 & V_y'^T
           \end{array}\right] x

    with

    .. math::

       V_y' = d^{\frac{1}{4}} \left[\begin{array}{cccc}
           y_1            & y_2            & \dots  & y_{d'}  \\
           y_{d' + 1}     & y_{d' + 2}     & \dots  & y_{2d'} \\
           \vdots         & \vdots         & \ddots & \vdots  \\
           y_{d - d' + 1} & y_{d - d' + 2} & \dots  & y_d
       \end{array}\right]

    and

    .. math:: d'^2 = d.

    The approximate inverse :math:`y^+` for :math:`y` is permuting the elements
    such that :math:`V_{y^+} = V_y^T`.

    Note that TVTB requires the vector dimensionality to be square.

    The TVTB binding operation is neither associative nor commutative.
    In contrast to VTB, however, TVTB has two-sided identities and inverses.
    Other properties are equivalent to VTB.

    .. seealso::
        `.VtbAlgebra`, `.VTB`

    Parameters
    ----------
    n_neurons : int
        Number of neurons to use in each product computation.
    dimensions : int
        The number of dimensions of the input and output vectors. Needs to be a
        square number.
    unbind_left : bool
        Whether to unbind the left input vector from the right input vector.
    unbind_right : bool
        Whether to unbind the right input vector from the left input vector.
    **kwargs : dict
        Keyword arguments to pass through to the `nengo.Network` constructor.

    Attributes
    ----------
    input_left : nengo.Node
        The left operand vector to be bound.
    input_right : nengo.Node
        The right operand vector to be bound.
    mat : nengo.Node
        Representation of the matrix :math:`V_y'`.
    vec : nengo.Node
        Representation of the vector :math:`y`.
    matmuls : list
        Matrix multiplication networks.
    output : nengo.Node
        The resulting bound vector.
    """

    def __init__(
        self, n_neurons, dimensions, unbind_left=False, unbind_right=False, **kwargs
    ):
        super().__init__(**kwargs)

        sub_d = calc_sub_d(dimensions)

        shape_left = (sub_d, sub_d)
        shape_right = (sub_d, 1)

        with self:
            self.input_left = nengo.Node(size_in=dimensions)
            self.input_right = nengo.Node(size_in=dimensions)
            self.output = nengo.Node(size_in=dimensions)

            self.mat = nengo.Node(size_in=dimensions)
            self.vec = nengo.Node(size_in=dimensions)

            if unbind_left and unbind_right:
                raise ValueError("Cannot unbind both sides at the same time.")
            elif unbind_left:
                nengo.Connection(
                    self.input_left,
                    self.mat,
                    transform=inversion_matrix(dimensions),
                    synapse=None,
                )
                nengo.Connection(
                    self.input_right,
                    self.vec,
                    synapse=None,
                )
            else:
                nengo.Connection(self.input_left, self.vec, synapse=None)
                if unbind_right:
                    tr = inversion_matrix(dimensions)
                else:
                    tr = 1.0
                nengo.Connection(self.input_right, self.mat, transform=tr, synapse=None)

            with nengo.Config(nengo.Ensemble) as cfg:
                cfg[nengo.Ensemble].intercepts = CosineSimilarity(dimensions + 2)
                cfg[nengo.Ensemble].eval_points = CosineSimilarity(dimensions + 2)
                self.matmuls = [
                    MatrixMult(n_neurons, shape_left, shape_right) for i in range(sub_d)
                ]

            for i in range(sub_d):
                mm = self.matmuls[i]
                sl = slice(i * sub_d, (i + 1) * sub_d)
                nengo.Connection(
                    self.mat,
                    mm.input_left,
                    transform=inversion_matrix(dimensions),
                    synapse=None,
                )
                nengo.Connection(self.vec[sl], mm.input_right, synapse=None)
                nengo.Connection(
                    mm.output, self.output[sl], transform=np.sqrt(sub_d), synapse=None
                )
