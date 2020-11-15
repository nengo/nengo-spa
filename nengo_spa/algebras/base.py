import warnings
from abc import ABCMeta
from enum import Enum


class ElementSidedness(Enum):
    """The side in a binary operation for which a special element's properties hold."""

    LEFT = "left"
    RIGHT = "right"
    TWO_SIDED = "two-sided"


class _DuckTypedABCMeta(ABCMeta):
    def __instancecheck__(cls, instance):
        if super().__instancecheck__(instance):
            return True

        for member in dir(cls):
            if member.startswith("_"):
                continue
            if not hasattr(instance, member) or not hasattr(
                getattr(instance, member), "__self__"
            ):
                return False
        warnings.warn(
            DeprecationWarning(
                "Please do not rely on pure duck-typing for {clsname}. "
                "Explicitly register your class {userclass} as a virtual subclass "
                "of {clsname} or derive from it.".format(
                    clsname=cls.__name__, userclass=instance.__class__.__name__
                )
            )
        )
        return True


class AbstractAlgebra(metaclass=_DuckTypedABCMeta):
    """Abstract base class for algebras.

    Custom algebras can be defined by implementing the interface of this
    abstract base class.
    """

    def is_valid_dimensionality(self, d):
        """Checks whether *d* is a valid vector dimensionality.

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
        raise NotImplementedError()

    def make_unitary(self, v):
        """Returns a unitary vector based on the vector *v*.

        A unitary vector does not change the length of a vector it is bound to.

        Parameters
        ----------
        v : (d,) ndarray
            Vector to base unitary vector on.

        Returns
        -------
        ndarray
            Unitary vector.
        """
        raise NotImplementedError()

    def superpose(self, a, b):
        """Returns the superposition of *a* and *b*.

        This is commonly elementwise addition.

        Parameters
        ----------
        a : (d,) ndarray
            Left operand in superposition.
        b : (d,) ndarray
            Right operand in superposition.

        Returns
        -------
        (d,) ndarray
            Superposed vector.
        """
        raise NotImplementedError()

    def bind(self, a, b):
        """Returns the binding of *a* and *b*.

        The resulting vector should in most cases be dissimilar to both inputs.

        Parameters
        ----------
        a : (d,) ndarray
            Left operand in binding.
        b : (d,) ndarray
            Right operand in binding.

        Returns
        -------
        (d,) ndarray
            Bound vector.
        """
        raise NotImplementedError()

    def invert(self, v, sidedness=ElementSidedness.TWO_SIDED):
        """Invert vector *v*.

        A vector bound to its inverse will result in the identity vector.

        Some algebras might not have an inverse only on specific sides. In that
        case a *NotImplementedError* may be raised for non-existing inverses.

        Parameters
        ----------
        v : (d,) ndarray
            Vector to invert.
        sidedness : ElementSidedness, optional
            Side in the binding operation on which the returned value acts as
            inverse.

        Returns
        -------
        (d,) ndarray
            Inverted vector.
        """
        raise NotImplementedError()

    def get_binding_matrix(self, v, swap_inputs=False):
        """Returns the transformation matrix for binding with a fixed vector.

        Parameters
        ----------
        v : (d,) ndarray
            Fixed vector to derive binding matrix for.
        swap_inputs : bool, optional
            By default the matrix will be such that *v* becomes the *right*
            operand in the binding. By setting *swap_inputs*, the matrix will
            be such that *v* becomes the *left* operand. For binding operations
            that are commutative (such as circular convolution), this has no
            effect.

        Returns
        -------
        (d, d) ndarray
            Transformation matrix to perform binding with *v*.
        """
        raise NotImplementedError()

    def get_inversion_matrix(self, d, sidedness=ElementSidedness.TWO_SIDED):
        """Returns the transformation matrix for inverting a vector.

        Some algebras might not have an inverse only on specific sides. In that
        case a *NotImplementedError* may be raised for non-existing inverses.

        Parameters
        ----------
        d : int
            Vector dimensionality (determines the matrix size).
        sidedness : ElementSidedness, optional
            Side in the binding operation on which a transformed vectors acts as
            inverse.

        Returns
        -------
        (d, d) ndarray
            Transformation matrix to invert a vector.
        """
        raise NotImplementedError()

    def implement_superposition(self, n_neurons_per_d, d, n):
        """Implement neural network for superposing vectors.

        Parameters
        ----------
        n_neurons_per_d : int
            Neurons to use per dimension.
        d : int
            Dimensionality of the vectors.
        n : int
            Number of vectors to superpose in the network.

        Returns
        -------
        tuple
            Tuple *(net, inputs, output)* where *net* is the implemented
            `nengo.Network`, *inputs* a sequence of length *n* of inputs to the
            network, and *output* the network output.
        """
        raise NotImplementedError()

    def implement_binding(self, n_neurons_per_d, d, unbind_left, unbind_right):
        """Implement neural network for binding vectors.

        Parameters
        ----------
        n_neurons_per_d : int
            Neurons to use per dimension.
        d : int
            Dimensionality of the vectors.
        unbind_left : bool
            Whether the left input should be unbound from the right input.
        unbind_right : bool
            Whether the right input should be unbound from the left input.

        Returns
        -------
        tuple
            Tuple *(net, inputs, output)* where *net* is the implemented
            `nengo.Network`, *inputs* a sequence of the left and the right
            input in that order, and *output* the network output.
        """
        raise NotImplementedError()

    def absorbing_element(self, d, sidedness=ElementSidedness.TWO_SIDED):
        """Return the standard absorbing element of dimensionality *d*.

        An absorbing element will produce a scaled version of itself when bound
        to another vector. The standard absorbing element is the absorbing
        element with norm 1.

        Some algebras might not have an absorbing element other than the zero
        vector. In that case a *NotImplementedError* may be raised.

        Parameters
        ----------
        d : int
            Vector dimensionality.
        sidedness : ElementSidedness, optional
            Side in the binding operation on which the element absorbs.

        Returns
        -------
        (d,) ndarray
            Standard absorbing element.
        """
        raise NotImplementedError()

    def identity_element(self, d, sidedness=ElementSidedness.TWO_SIDED):
        """Return the identity element of dimensionality *d*.

        The identity does not change the vector it is bound to.

        Some algebras might not have an identity element. In that case a
        *NotImplementedError* may be raised.

        Parameters
        ----------
        d : int
            Vector dimensionality.
        sidedness : ElementSidedness, optional
            Side in the binding operation on which the element acts as identity.

        Returns
        -------
        (d,) ndarray
            Identity element.
        """
        raise NotImplementedError()

    def zero_element(self, d, sidedness=ElementSidedness.TWO_SIDED):
        """Return the zero element of dimensionality *d*.

        The zero element produces itself when bound to a different vector.
        Usually this will be the zero vector.

        Some algebras might not have a zero element. In that case a
        *NotImplementedError* may be raised.

        Parameters
        ----------
        d : int
            Vector dimensionality.
        sidedness : ElementSidedness, optional
            Side in the binding operation on which the element acts as zero.

        Returns
        -------
        (d,) ndarray
            Zero element.
        """
        raise NotImplementedError()
