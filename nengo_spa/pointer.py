import nengo
from nengo.exceptions import ValidationError
from nengo.utils.compat import is_integer, is_number, range
import numpy as np

from nengo_spa.ast.base import Fixed, infer_types, TypeCheckedBinaryOp
from nengo_spa.types import TAnyVocab, TScalar, TVocabulary


class SemanticPointer(Fixed):
    """A Semantic Pointer, based on Holographic Reduced Representations.

    Operators are overloaded so that ``+`` and ``-`` are addition,
    ``*`` is circular convolution, and ``~`` is the inversion operator.

    Parameters
    ----------
    data : int or array_like
        The vector constituting the Semantic Pointer. If an integer is given,
        a random unit-length vector will be generated.
    rng : numpy.random.RandomState, optional
        Random number generator used for random generation of a Semantic
        Pointer.
    vocab : Vocabulary
        Vocabulary that the Semantic Pointer is considered to be part of.

    Attributes
    ----------
    v : array_like
        The vector constituting the Semantic Pointer.
    vocab : Vocabulary or None
        The vocabulary the this Semantic Pointer is considered to be part of.
    """

    def __init__(self, data, rng=None, vocab=None):
        super(SemanticPointer, self).__init__(
            TAnyVocab if vocab is None else TVocabulary(vocab))

        if rng is None:
            rng = np.random

        if is_integer(data):
            if data < 1:
                raise ValidationError("Number of dimensions must be a "
                                      "positive int", attr='data', obj=self)

            self.v = rng.randn(data)
            self.v /= np.linalg.norm(self.v)
        else:
            self.v = np.array(data, dtype=float)
            if len(self.v.shape) != 1:
                raise ValidationError("'data' must be a vector", 'data', self)
        self.v.setflags(write=False)

        self.vocab = vocab

    def evaluate(self):
        return self

    def connect_to(self, sink, **kwargs):
        return nengo.Connection(self.construct(), sink, **kwargs)

    def construct(self):
        return nengo.Node(self.v)

    def normalized(self):
        """Normalize the Semantic Pointer and return it as a new object.

        The original object is not modified.
        """
        nrm = np.linalg.norm(self.v)
        if nrm > 0:
            return SemanticPointer(self.v / nrm, vocab=self.vocab)

    def unitary(self):
        """Make the Semantic Pointer unitary and return it as a new object.

        The original object is not modified.

        A unitary Semantic Pointer has the property that it does not change
        the length of Semantic Pointers it is bound with using circular
        convolution.
        """
        fft_val = np.fft.fft(self.v)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
        fft_unit = fft_val / fft_norms
        return SemanticPointer(np.array((np.fft.ifft(
            fft_unit, n=len(self))).real), vocab=self.vocab)

    def copy(self):
        """Return another semantic pointer with the same data."""
        return SemanticPointer(data=self.v, vocab=self.vocab)

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
        type_ = infer_types(self, other)
        vocab = None if type_ == TAnyVocab else type_.vocab
        return SemanticPointer(data=self.v + other.evaluate().v, vocab=vocab)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return SemanticPointer(data=-self.v, vocab=self.vocab)

    @TypeCheckedBinaryOp(Fixed)
    def __sub__(self, other):
        type_ = infer_types(self, other)
        vocab = None if type_ == TAnyVocab else type_.vocab
        return SemanticPointer(data=self.v - other.evaluate().v, vocab=vocab)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        """Multiplication of two SemanticPointers is circular convolution.

        If multiplied by a scalar, we do normal multiplication.
        """
        if is_number(other):
            return SemanticPointer(data=self.v * other, vocab=self.vocab)
        elif isinstance(other, Fixed):
            if other.type == TScalar:
                return SemanticPointer(
                    data=self.v * other.evaluate(), vocab=self.vocab)
            else:
                return self.convolve(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Multiplication of two SemanticPointers is circular convolution.

        If multiplied by a scalar, we do normal multiplication.
        """
        return self.__mul__(other)

    def __invert__(self):
        """Return a reorganized vector that acts as an inverse for convolution.

        This reorganization turns circular convolution into circular
        correlation, meaning that ``A*B*~B`` is approximately ``A``.

        For the vector ``[1, 2, 3, 4, 5]``, the inverse is ``[1, 5, 4, 3, 2]``.
        """
        return SemanticPointer(
            data=self.v[-np.arange(len(self))], vocab=self.vocab)

    def convolve(self, other):
        """Return the circular convolution of two SemanticPointers."""
        type_ = infer_types(self, other)
        vocab = None if type_ == TAnyVocab else type_.vocab
        n = len(self)
        x = np.fft.irfft(
            np.fft.rfft(self.v) * np.fft.rfft(other.evaluate().v), n=n)
        return SemanticPointer(data=x, vocab=vocab)

    def get_convolution_matrix(self):
        """Return the matrix that does a circular convolution by this vector.

        This should be such that
        ``A*B == dot(A.get_convolution_matrix(), B.v)``.
        """
        D = len(self.v)
        T = []
        for i in range(D):
            T.append([self.v[(i - j) % D] for j in range(D)])
        return np.array(T)

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


class Identity(SemanticPointer):
    """Circular convolution identity.

    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary
        Vocabulary that the Semantic Pointer is considered to be part of.
    """

    def __init__(self, n_dimensions, vocab=None):
        data = np.zeros(n_dimensions)
        data[0] = 1.
        super(Identity, self).__init__(data, vocab=vocab)


class AbsorbingElement(SemanticPointer):
    r"""Circular convolution absorbing element.

    If :math:`z` denotes the absorbing element, :math:`v \circledast z = c z`,
    where :math:`v` is a Semantic Pointer and :math:`c` is a real-valued
    scalar.

    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary
        Vocabulary that the Semantic Pointer is considered to be part of.
    """
    def __init__(self, n_dimensions, vocab=None):
        data = np.ones(n_dimensions) / np.sqrt(n_dimensions)
        super(AbsorbingElement, self).__init__(data, vocab=vocab)


class Zero(SemanticPointer):
    """An all zero Semantic Pointer.

    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary
        Vocabulary that the Semantic Pointer is considered to be part of.
    """
    def __init__(self, n_dimensions, vocab=None):
        super(Zero, self).__init__(np.zeros(n_dimensions), vocab=vocab)
