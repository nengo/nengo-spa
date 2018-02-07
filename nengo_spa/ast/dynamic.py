"""AST classes for dynamic operations (i.e. their output changes over time)."""

import weakref

import nengo
from nengo.utils.compat import is_number
import numpy as np

from nengo_spa.ast.base import infer_types, Node, TypeCheckedBinaryOp
from nengo_spa.ast.symbolic import FixedScalar, PointerSymbol, Symbol
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import TAnyVocab, TScalar, TAnyVocabOfDim, TVocabulary


BasalGangliaRealization = None
BindRealization = None
DotProductRealization = None
ProductRealization = None
ScalarRealization = None
StateRealization = None
ThalamusRealization = None


input_network_registry = weakref.WeakKeyDictionary()
input_vocab_registry = weakref.WeakKeyDictionary()
output_vocab_registry = weakref.WeakKeyDictionary()


def as_node(obj):
    if is_number(obj):
        obj = FixedScalar(obj)
    return obj


binary_node_op = TypeCheckedBinaryOp(Node, as_node)


class DynamicNode(Node):
    """Base class for AST node with an output that changes over time."""

    def __invert__(self):
        if not hasattr(self.type, 'dimensions'):
            raise SpaTypeError(
                "Cannot invert semantic pointer of unknown dimensionality.")
        dimensions = self.type.dimensions
        transform = np.eye(dimensions)[-np.arange(dimensions)]
        return Transformed(self, transform, self.type)

    def __neg__(self):
        return Transformed(self, transform=-1, type_=self.type)

    @binary_node_op
    def __add__(self, other):
        type_ = infer_types(self, other)
        return Summed((self, other), type_)

    @binary_node_op
    def __radd__(self, other):
        return self + other

    @binary_node_op
    def __sub__(self, other):
        return self + (-other)

    @binary_node_op
    def __rsub__(self, other):
        return (-self) + other

    def _mul_with_fixed(self, other):
        infer_types(self, other)
        if other.type == TScalar:
            tr = other.value
        elif self.type == TScalar and other.type == TAnyVocab:
            raise SpaTypeError(
                "Cannot infer vocabulary for fixed pointer when multiplying "
                "with scalar.")
        elif isinstance(other.type, TVocabulary):
            if self.type == TScalar:
                tr = other.evaluate().v
            else:
                tr = other.evaluate().get_convolution_matrix()
        else:
            raise AssertionError("Unexpected node type in multiply.")
        return Transformed(self, tr, self.type)

    def _mul_with_dynamic(self, other, swap_inputs=False):
        type_ = infer_types(self, other)
        if type_ == TScalar:
            mul = ProductRealization()
        elif self.type == TScalar or other.type == TScalar:
            raise NotImplementedError(
                "Dynamic scaling of semantic pointer not implemented.")
        else:
            mul = BindRealization(self.type.vocab)

        if swap_inputs:
            a, b = other, self
        else:
            a, b = self, other
        a.connect_to(mul.input_a)
        b.connect_to(mul.input_b)
        return ModuleOutput(mul.output, type_)

    @binary_node_op
    def __mul__(self, other):
        if isinstance(other, Symbol):
            return self._mul_with_fixed(other)
        else:
            return self._mul_with_dynamic(other)

    @binary_node_op
    def __rmul__(self, other):
        if isinstance(other, Symbol):
            return self._mul_with_fixed(other)
        else:
            return self._mul_with_dynamic(other, swap_inputs=True)

    @binary_node_op
    def dot(self, other):
        type_ = infer_types(self, other)

        if self.type == TScalar or other.type == TScalar:
            raise SpaTypeError("Cannot do a dot product with a scalar.")

        if isinstance(other, PointerSymbol):
            tr = np.atleast_2d(other.evaluate().v)
            return Transformed(self, tr, TScalar)
        else:
            net = DotProductRealization(type_.vocab)
            self.connect_to(net.input_a)
            other.connect_to(net.input_b)
            return ModuleOutput(net.output, TScalar)

    def __matmul__(self, other):
        return self.dot(other)

    def rdot(self, other):
        return self.dot(other)

    def __rmatmul__(self, other):
        return self.rdot(other)

    def reinterpret(self, vocab=None):
        return Transformed(
            self, np.eye(self.type.dimensions),
            TAnyVocabOfDim(self.type.dimensions)
            if vocab is None else TVocabulary(vocab))

    def translate(self, vocab, populate=None, keys=None, solver=None):
        tr = self.type.vocab.transform_to(vocab, populate, keys, solver)
        return Transformed(self, tr, TVocabulary(vocab))


class Transformed(DynamicNode):
    """AST node representing a transform.

    Parameters
    ----------
    source : NengoObject
        Nengo object providing the output to apply the transform to.
    transform : ndarray
        Transform to apply.
    type_ : nengo_spa.types.Type
        The resulting type.
    """

    def __init__(self, source, transform, type_):
        super(Transformed, self).__init__(type_=type_)
        self.source = source
        self.transform = transform

    def connect_to(self, sink, **kwargs):
        if 'transform' in kwargs:
            transform = np.dot(kwargs.pop('transform'), self.transform)
        else:
            transform = self.transform
        return self.source.connect_to(sink, transform=transform, **kwargs)

    def construct(self):
        if self.type == TScalar:
            size_in = 1
        else:
            size_in = self.type.vocab.dimensions
        node = nengo.Node(size_in=size_in)
        self.connect_to(node, synapse=None)
        return node


class Summed(DynamicNode):
    """AST node representing the sum of multiple nodes.

    Parameters
    ----------
    sources : sequence of NengoObject
        Sequence of Nengo objects providing outputs to be summed.
    type_ : nengo_spa.types.Type
        The resulting type.
    """

    def __init__(self, sources, type_):
        super(Summed, self).__init__(type_=type_)
        self.sources = sources

    def connect_to(self, sink, **kwargs):
        for s in self.sources:
            s.connect_to(sink, **kwargs)

    def construct(self):
        dimensions = 1 if self.type == TScalar else self.type.dimensions
        node = nengo.Node(size_in=dimensions)
        self.connect_to(node, synapse=None)
        return node


class ModuleOutput(DynamicNode):
    """AST node representing the output of a SPA module.

    Parameters
    ----------
    output : NengoObject
        Nengo object providing the module output.
    type_ : nengo_spa.types.Type
        The resulting type.
    """

    def __init__(self, output, type_):
        super(ModuleOutput, self).__init__(type_=type_)
        self.output = output

    def construct(self):
        return self.output

    def connect_to(self, sink, **kwargs):
        nengo.Connection(self.output, sink, **kwargs)
