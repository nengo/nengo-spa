import nengo
from nengo.dists import CosineSimilarity
from nengo.exceptions import ValidationError
import numpy as np

from nengo_spa.networks.matrix_multiplication import MatrixMult


def calc_sub_d(dimensions):
    sub_d = int(np.sqrt(dimensions))
    if sub_d * sub_d != dimensions:
        raise ValidationError(
            "Dimensions must be a square number.", 'dimensions')
    return sub_d


def inversion_matrix(dimensions):
    sub_d = calc_sub_d(dimensions)
    m = np.zeros((dimensions, dimensions))
    for i in range(dimensions):
        j = sub_d * i
        m[j % dimensions + j // dimensions, i] = 1.
    return m


def swapping_matrix(dimensions):
    sub_d = calc_sub_d(dimensions)
    m = np.zeros((dimensions, dimensions))
    for i in range(dimensions):
        j = i // sub_d + sub_d * (i % sub_d)
        m[i, j] = 1.
    return m


def VTB(n_neurons, dimensions, unbind_left=False, unbind_right=False,
        **kwargs):
    r"""Compute vector-derived transformation binding (VTB).

    VTB uses elementwise addition for superposition. The binding operation
    :math:`\mathcal{B}(x, y)` is defined as

    .. math::

       \mathcal{B}(x, y) := V_y x = \left[\begin{array}{ccc}
           V_y' &    0 &    0 \\
              0 & V_y' &    0 \\
              0 &    0 & V_y'
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
    such that :math:`V_{y^+} = V_y`.

    Note that VTB requires the vector dimensionality to be square.

    The VTB binding operation is neither associative nor commutative.

    Publications with further information are forthcoming.

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

    Returns
    -------
    nengo.Network
        The newly built product network with attributes:

         * **input_left** (`nengo.Node`): The left operand vector to be bound.
         * **input_right** (`nengo.Node`): The right operand vector to be
           bound.
         * **mat** (`nengo.Node`): Representation of the matrix :math:`V_y'`.
         * **vec** (`nengo.Node`): Representation of the vector :math:`y`.
         * **matmuls** (`list`): Matrix multiplication networks.
         * **output** (`nengo.Node`): The resulting bound vector.
    """
    sub_d = calc_sub_d(dimensions)

    shape_left = (sub_d, sub_d)
    shape_right = (sub_d, 1)

    with nengo.Network(**kwargs) as net:
        net.input_left = nengo.Node(size_in=dimensions)
        net.input_right = nengo.Node(size_in=dimensions)
        net.output = nengo.Node(size_in=dimensions)

        net.mat = nengo.Node(size_in=dimensions)
        net.vec = nengo.Node(size_in=dimensions)

        if unbind_left and unbind_right:
            raise ValueError("Cannot unbind both sides at the same time.")
        elif unbind_left:
            nengo.Connection(
                net.input_left, net.mat,
                transform=inversion_matrix(dimensions), synapse=None)
            nengo.Connection(
                net.input_right, net.vec,
                transform=swapping_matrix(dimensions), synapse=None)
        else:
            nengo.Connection(net.input_left, net.vec, synapse=None)
            if unbind_right:
                tr = inversion_matrix(dimensions)
            else:
                tr = 1.
            nengo.Connection(
                net.input_right, net.mat, transform=tr, synapse=None)

        with nengo.Config(nengo.Ensemble) as cfg:
            cfg[nengo.Ensemble].intercepts = CosineSimilarity(
                dimensions + 2)
            cfg[nengo.Ensemble].eval_points = CosineSimilarity(
                dimensions + 2)
            net.matmuls = [
                MatrixMult(n_neurons, shape_left, shape_right)
                for i in range(sub_d)]

        for i in range(sub_d):
            mm = net.matmuls[i]
            sl = slice(i * sub_d, (i + 1) * sub_d)
            nengo.Connection(net.mat, mm.input_left, synapse=None)
            nengo.Connection(net.vec[sl], mm.input_right, synapse=None)
            nengo.Connection(
                mm.output, net.output[sl], transform=np.sqrt(sub_d),
                synapse=None)

    return net
