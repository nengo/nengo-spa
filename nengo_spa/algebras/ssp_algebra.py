import nengo
import numpy as np

from nengo_spa.algebras import HrrAlgebra

from nengo_spa.algebras.base import (
    AbstractAlgebra,
    AbstractSign,
    CommonProperties,
    ElementSidedness,
)
from nengo_spa.networks.circularconvolution import CircularConvolution


class SspAlgebra(HrrAlgebra):
    r"""
    Spatial Semantic Pointer Algebra - a restriction on the HRR algebra to 
    ensure vectors are always unitary and positive.

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
            cls._instance = super(SspAlgebra, cls).__new__(cls)
        return cls._instance

    def create_vector(self, d, properties, *, rng=None):
        """
        Create a vector fulfilling given properties in the algebra.

        Parameters
        ----------
        d : int
            Vector dimensionality
        properties : set of str
            Definition of properties for the vector to fulfill. Valid set
            elements are constants defined in `.SspProperties`.

            SSPs are always UNITARY and POSITIVE
        rng : numpy.random.RandomState, optional
            The random number generator to use to create the vector.

        Returns
        -------
        ndarray
            Random vector with desired properties.
        """
        properties = set(properties)

        if rng is None:
            rng = np.random.RandomState()

        v = rng.randn(d)
        v /= np.linalg.norm(v)

        v = self.abs(v)
        v = self.make_unitary(v)


        if SspProperties.POSITIVE in properties:
            properties.remove(SspProperties.POSITIVE)
        if SspProperties.UNITARY in properties:
            properties.remove(SspProperties.UNITARY)

        if len(properties) > 0:
            raise ValueError("Invalid properties: " + ", ".join(properties))

        return v



class SspProperties:
    """Vector properties supported by the `.SspAlgebra`."""

    UNITARY = CommonProperties.UNITARY
    """A unitary vector does not change the length of a vector it is bound to."""

    POSITIVE = CommonProperties.POSITIVE
    """
    A positive vector does not change the sign of a vector it is bound to.

    A positive vector allows for fractional binding powers.
    """
