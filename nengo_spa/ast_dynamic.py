import nengo
from nengo.utils.compat import is_number
import numpy as np

from nengo_spa.ast2 import infer_types, Node
from nengo_spa.ast_symbolic import FixedScalar, PointerSymbol, Symbol
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import TInferVocab, TScalar, TVocabulary


DotProductRealization = None
BindRealization = None
ProductRealization = None


def as_node(obj):
    if is_number(obj):
        obj = FixedScalar(obj)
    return obj


class DynamicNode(Node):
    def __invert__(self):
        # FIXME alternate binding operators
        vocab = self.type.vocab
        transform = np.eye(vocab.dimensions)[-np.arange(vocab.dimensions)]
        return Transformed(self.output, transform, self.type)

    def __neg__(self):
        return Transformed(self.output, transform=-1, type_=self.type)

    def __add__(self, other):
        other = as_node(other)
        if not isinstance(other, Node):
            return NotImplemented
        type_ = infer_types(self, other)
        return Summed((self, other), type_)

    def __radd__(self, other):
        other = as_node(other)
        if not isinstance(other, Node):
            return NotImplemented
        return self + other

    def __sub__(self, other):
        other = as_node(other)
        if not isinstance(other, Node):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        other = as_node(other)
        if not isinstance(other, Node):
            return NotImplemented
        return (-self) + other

    def _mul_with_fixed(self, other):
        type_ = infer_types(self, other)
        if other.type == TScalar:
            tr = other.value
        elif self.type == TScalar and other.type == TInferVocab:
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
        return Transformed(self.construct(), tr, self.type)

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

    def __mul__(self, other):
        other = as_node(other)
        if not isinstance(other, Node):
            return NotImplemented

        if isinstance(other, Symbol):
            return self._mul_with_fixed(other)
        else:
            return self._mul_with_dynamic(other)

    def __rmul__(self, other):
        other = as_node(other)
        if not isinstance(other, Node):
            return NotImplemented

        if isinstance(other, Symbol):
            return self._mul_with_fixed(other)
        else:
            return self._mul_with_dynamic(other, swap_inputs=True)

    def dot(self, other):
        other = as_node(other)
        if not isinstance(other, Node):
            raise NotImplementedError()
        type_ = infer_types(self, other)

        if self.type == TScalar or other.type == TScalar:
            raise NotImplementedError()  # FIXME better error?

        if isinstance(other, PointerSymbol):
            tr = np.atleast_2d(other.evaluate().v)
            return Transformed(self.construct(), tr, TScalar)
        else:
            net = DotProductRealization(type_.vocab)
            self.connect_to(net.input_a)
            other.connect_to(net.input_b)
            return ModuleOutput(net.output, TScalar)

    def rdot(self, other):
        return self.dot(other)

    def translate(self, vocab, populate=None, keys=None, solver=None):
        tr = self.type.vocab.transform_to(vocab, populate, solver)
        return Transformed(self.construct(), tr, TVocabulary(vocab))


class Transformed(DynamicNode):
    def __init__(self, source, transform, type_):
        super(Transformed, self).__init__(type_=type_)
        self.source = source
        self.transform = transform

    def connect_to(self, sink):
        # FIXME connection params
        return nengo.Connection(self.source, sink, transform=self.transform)

    def construct(self):
        if self.type == TScalar:
            size_in = 1
        else:
            size_in = self.type.vocab.dimensions
        node = nengo.Node(size_in=size_in)
        self.connect_to(node)
        return node


class Summed(DynamicNode):
    def __init__(self, sources, type_):
        super(Summed, self).__init__(type_=type_)
        self.sources = sources

    def connect_to(self, sink):
        for s in self.sources:
            s.connect_to(sink)

    def construct(self):
        node = nengo.Node(size_in=self.type.vocab.dimensions)
        self.connect_to(node)
        return node


class ModuleOutput(DynamicNode):
    def __init__(self, output, type_):
        super(ModuleOutput, self).__init__(type_=type_)
        self.output = output

    def construct(self):
        return self.output

    def connect_to(self, sink):
        nengo.Connection(self.output, sink)


class ModuleInput(object):
    def __init__(self, input_, type_):
        self.input = input_
        self.type = type_

    def __rrshift__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        infer_types(self, other)
        other.connect_to(self.input)
