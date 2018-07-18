import nengo
from nengo.exceptions import ValidationError
from nengo.utils.compat import is_array, is_number
import numpy as np

from nengo_spa.algebras.cconv import CircularConvolutionAlgebra
from nengo_spa.ast.base import Fixed, infer_types, TypeCheckedBinaryOp
from nengo_spa.types import TAnyVocab, TScalar, TVocabulary


class SemanticPointer(Fixed):
    """A Semantic Pointer, based on Holographic Reduced Representations.

    Operators are overloaded so that ``+`` and ``-`` are addition,
    ``*`` is circular convolution, and ``~`` is the inversion operator.

    Parameters
    ----------
    data : array_like
        The vector constituting the Semantic Pointer.
    rng : numpy.random.RandomState, optional
        Random number generator used for random generation of a Semantic
        Pointer.
    vocab : Vocabulary, optional
        Vocabulary that the Semantic Pointer is considered to be part of.
        Mutually exclusive with the *algebra* argument.
    algebra : AbstractAlgebra, optional
        Algebra used to perform vector symbolic operations on the Semantic
        Pointer. Defaults to `.CircularConvolutionAlgebra`. Mutually exclusive
        with the *vocab* argument.

    Attributes
    ----------
    v : array_like
        The vector constituting the Semantic Pointer.
    algebra : AbstractAlgebra
        Algebra that defines the vector symbolic operations on this Semantic
        Pointer.
    vocab : Vocabulary or None
        The vocabulary the this Semantic Pointer is considered to be part of.
    """

    def __init__(self, data, rng=None, vocab=None, algebra=None):
        super(SemanticPointer, self).__init__(
            TAnyVocab if vocab is None else TVocabulary(vocab))
        self.algebra = self._get_algebra(vocab, algebra)

        if rng is None:
            rng = np.random

        self.v = np.array(data, dtype=float)
        if len(self.v.shape) != 1:
            raise ValidationError("'data' must be a vector", 'data', self)
        self.v.setflags(write=False)

        self.vocab = vocab

    def _get_algebra(cls, vocab, algebra):
        if algebra is None:
            if vocab is None:
                algebra = CircularConvolutionAlgebra()
            else:
                algebra = vocab.algebra
        elif vocab is not None and vocab.algebra is not algebra:
            raise ValueError(
                "vocab and algebra argument are mutually exclusive")
        return algebra

    def evaluate(self):
        return self

    def connect_to(self, sink, **kwargs):
        return nengo.Connection(self.construct(), sink, **kwargs)

    def construct(self):
        return nengo.Node(
            self.v, label="Semantic Pointer ({}d)".format(len(self)))

    def normalized(self):
        """Normalize the Semantic Pointer and return it as a new object.

        The original object is not modified.
        """
        nrm = np.linalg.norm(self.v)
        if nrm > 0:
            return SemanticPointer(
                self.v / nrm, vocab=self.vocab, algebra=self.algebra)

    def unitary(self):
        """Make the Semantic Pointer unitary and return it as a new object.

        The original object is not modified.

        A unitary Semantic Pointer has the property that it does not change
        the length of Semantic Pointers it is bound with using circular
        convolution.
        """
        return SemanticPointer(
            self.algebra.make_unitary(self.v), vocab=self.vocab,
            algebra=self.algebra)

    def copy(self):
        """Return another semantic pointer with the same data."""
        return SemanticPointer(
            data=self.v, vocab=self.vocab, algebra=self.algebra)

    def length(self):
        """Return the L2 norm of the vector."""
        return np.linalg.norm(self.v)

    def __len__(self):
        """Return the number of dimensions in the vector."""
        return len(self.v)

    def __str__(self):
        return str(self.v)

    @TypeCheckedBinaryOp(Fixed)
    def __add__(self, other):
        return self._add(other, swap=False)

    @TypeCheckedBinaryOp(Fixed)
    def __radd__(self, other):
        return self._add(other, swap=True)

    def _add(self, other, swap=False):
        type_ = infer_types(self, other)
        vocab = None if type_ == TAnyVocab else type_.vocab
        if vocab is None:
            self._ensure_algebra_match(other)
        a, b = self.v, other.evaluate().v
        if swap:
            a, b = b, a
        return SemanticPointer(
            data=self.algebra.superpose(a, b), vocab=vocab,
            algebra=self.algebra)

    def __neg__(self):
        return SemanticPointer(
            data=-self.v, vocab=self.vocab, algebra=self.algebra)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        """Multiplication of two SemanticPointers is circular convolution.

        If multiplied by a scalar, we do normal multiplication.
        """
        return self._mul(other, swap=False)

    def __rmul__(self, other):
        """Multiplication of two SemanticPointers is circular convolution.

        If multiplied by a scalar, we do normal multiplication.
        """
        return self._mul(other, swap=True)

    def _mul(self, other, swap=False):
        if is_array(other):
            raise TypeError(
                "Multiplication of Semantic Pointers with arrays in not "
                "allowed.")
        elif is_number(other):
            return SemanticPointer(
                data=self.v * other, vocab=self.vocab, algebra=self.algebra)
        elif isinstance(other, Fixed):
            if other.type == TScalar:
                return SemanticPointer(
                    data=self.v * other.evaluate(), vocab=self.vocab,
                    algebra=self.algebra)
            else:
                return self._bind(other, swap=swap)
        else:
            return NotImplemented

    def __invert__(self):
        """Return a reorganized vector that acts as an inverse for convolution.

        This reorganization turns circular convolution into circular
        correlation, meaning that ``A*B*~B`` is approximately ``A``.

        For the vector ``[1, 2, 3, 4, 5]``, the inverse is ``[1, 5, 4, 3, 2]``.
        """
        return SemanticPointer(
            data=self.algebra.invert(self.v), vocab=self.vocab,
            algebra=self.algebra)

    def bind(self, other):
        """Return the binding of two SemanticPointers."""
        return self._bind(other, swap=False)

    def rbind(self, other):
        """Return the binding of two SemanticPointers."""
        return self._bind(other, swap=True)

    def _bind(self, other, swap=False):
        type_ = infer_types(self, other)
        vocab = None if type_ == TAnyVocab else type_.vocab
        if vocab is None:
            self._ensure_algebra_match(other)
        a, b = self.v, other.evaluate().v
        if swap:
            a, b = b, a
        return SemanticPointer(
            data=self.algebra.bind(a, b), vocab=vocab, algebra=self.algebra)

    def get_binding_matrix(self, swap_inputs=False):
        """Return the matrix that does a binding with this vector.

        This should be such that
        ``A*B == dot(A.get_binding_matrix(), B.v)``.
        """
        return self.algebra.get_binding_matrix(self.v, swap_inputs=swap_inputs)

    def dot(self, other):
        """Return the dot product of the two vectors."""
        if isinstance(other, Fixed):
            infer_types(self, other)
            other = other.evaluate().v
        return np.dot(self.v, other)

    def __matmul__(self, other):
        return self.dot(other)

    def compare(self, other):
        """Return the similarity between two SemanticPointers.

        This is the normalized dot product, or (equivalently), the cosine of
        the angle between the two vectors.
        """
        if isinstance(other, SemanticPointer):
            infer_types(self, other)
            other = other.evaluate().v
        scale = np.linalg.norm(self.v) * np.linalg.norm(other)
        if scale == 0:
            return 0
        return np.dot(self.v, other) / scale

    def reinterpret(self, vocab):
        return SemanticPointer(self.v, vocab=vocab)

    def translate(self, vocab, populate=None, keys=None, solver=None):
        tr = self.vocab.transform_to(vocab, populate, solver)
        return SemanticPointer(np.dot(tr, self.evaluate().v), vocab=vocab)

    def distance(self, other):
        """Return a distance measure between the vectors.

        This is ``1-cos(angle)``, so that it is 0 when they are identical, and
        the distance gets larger as the vectors are farther apart.
        """
        return 1 - self.compare(other)

    def mse(self, other):
        """Return the mean-squared-error between two vectors."""
        if isinstance(other, SemanticPointer):
            infer_types(self, other)
            other = other.evaluate().v
        return np.sum((self.v - other)**2) / len(self.v)

    def _ensure_algebra_match(self, other):
        """Check the algebra of the *other*.

        If the *other* parameter is a `SemanticPointer` and uses a different
        algebra, a `TypeError` will be raised.
        """
        if isinstance(other, SemanticPointer):
            if self.algebra is not other.algebra:
                raise TypeError(
                    "Operation not supported for SemanticPointer with "
                    "different algebra.")


class Identity(SemanticPointer):
    """Identity element.

    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary, optional
        Vocabulary that the Semantic Pointer is considered to be part of.
        Mutually exclusive with the *algebra* argument.
    algebra : AbstractAlgebra, optional
        Algebra used to perform vector symbolic operations on the Semantic
        Pointer. Defaults to `.CircularConvolutionAlgebra`. Mutually exclusive
        with the *vocab* argument.
    """

    def __init__(self, n_dimensions, vocab=None, algebra=None):
        data = self._get_algebra(vocab, algebra).identity_element(n_dimensions)
        super(Identity, self).__init__(data, vocab=vocab, algebra=algebra)


class AbsorbingElement(SemanticPointer):
    r"""Absorbing element.

    If :math:`z` denotes the absorbing element, :math:`v \circledast z = c z`,
    where :math:`v` is a Semantic Pointer and :math:`c` is a real-valued
    scalar. Furthermore :math:`\|z\| = 1`.

    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary, optional
        Vocabulary that the Semantic Pointer is considered to be part of.
        Mutually exclusive with the *algebra* argument.
    algebra : AbstractAlgebra, optional
        Algebra used to perform vector symbolic operations on the Semantic
        Pointer. Defaults to `.CircularConvolutionAlgebra`. Mutually exclusive
        with the *vocab* argument.
    """
    def __init__(self, n_dimensions, vocab=None, algebra=None):
        data = self._get_algebra(vocab, algebra).absorbing_element(
            n_dimensions)
        super(AbsorbingElement, self).__init__(
            data, vocab=vocab, algebra=algebra)


class Zero(SemanticPointer):
    """Zero element.

    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary, optional
        Vocabulary that the Semantic Pointer is considered to be part of.
        Mutually exclusive with the *algebra* argument.
    algebra : AbstractAlgebra, optional
        Algebra used to perform vector symbolic operations on the Semantic
        Pointer. Defaults to `.CircularConvolutionAlgebra`. Mutually exclusive
        with the *vocab* argument.
    """
    def __init__(self, n_dimensions, vocab=None, algebra=None):
        data = self._get_algebra(vocab, algebra).zero_element(n_dimensions)
        super(Zero, self).__init__(data, vocab=vocab, algebra=algebra)
