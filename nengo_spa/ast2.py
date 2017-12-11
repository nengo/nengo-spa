import weakref

import nengo
from nengo.utils.compat import is_number
import numpy as np

from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import TInferVocab, TScalar, TVocabulary


input_network_registry = weakref.WeakKeyDictionary()
input_vocab_registry = weakref.WeakKeyDictionary()
output_vocab_registry = weakref.WeakKeyDictionary()

DotProductRealization = None
BindRealization = None
ProductRealization = None


def coerce_types(*types):
    if all(t == TScalar for t in types):
        return TScalar

    defined = [t for t in types if isinstance(t, TVocabulary)]
    if len(defined) > 0:
        if all(t == defined[0] for t in defined):
            return defined[0]
        else:
            raise SpaTypeError("Vocabulary mismatch.")
    else:
        return TInferVocab


def infer_types(*nodes):
    type_ = coerce_types(*[n.type for n in nodes])
    if isinstance(type_, TVocabulary):
        for n in (n for n in nodes if n.type == TInferVocab):
            n.type = type_
    return type_


def as_node(obj):
    if is_number(obj):
        obj = FixedScalar(obj)
    return obj


class Node(object):
    def __init__(self, type_):
        self.type = type_

    def connect_to(self, sink):
        raise NotImplementedError()

    def construct(self):
        raise NotImplementedError()


class FixedNode(Node):
    def evaluate(self):
        raise NotImplementedError()

    @property
    def expr(self):
        raise NotImplementedError()


class FixedScalar(FixedNode):
    def __init__(self, value):
        super(FixedScalar, self).__init__(type_=TScalar)
        self.value = value

    def connect_to(self, sink):
        return nengo.Connection(self.construct(), sink)

    def construct(self):
        return nengo.Node(self.value, label=str(self.value))

    def evaluate(self):
        return self.value

    @property
    def expr(self):
        return repr(self.value)


class FixedPointer(FixedNode):
    def __init__(self, expr, type_=TInferVocab):
        super(FixedPointer, self).__init__(type_=type_)
        self._expr = expr

    def connect_to(self, sink):
        return nengo.Connection(self.construct(), sink)

    def construct(self):
        return nengo.Node(self.evaluate().v, label=self.expr)

    def evaluate(self):
        return self.type.vocab.parse(self.expr)

    @property
    def expr(self):
        return self._expr

    def __invert__(self):
        return FixedPointer('~' + self.expr, self.type)

    def __neg__(self):
        return FixedPointer('-' + self.expr, self.type)

    def __add__(self, other):
        other = as_node(other)
        if not isinstance(other, FixedPointer):
            return NotImplemented
        type_ = infer_types(self, other)
        return FixedPointer(self.expr + '+' + other.expr, type_)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = as_node(other)
        if not isinstance(other, FixedPointer):
            return NotImplemented
        type_ = infer_types(self, other)
        return FixedPointer(self.expr + '-' + other.expr, type_)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = as_node(other)
        if not isinstance(other, FixedNode):
            return NotImplemented
        type_ = infer_types(self, other)
        return FixedPointer(self.expr + '*' + other.expr, type_)

    def __rmul__(self, other):
        other = as_node(other)
        if not isinstance(other, FixedNode):
            return NotImplemented
        type_ = infer_types(self, other)
        return FixedPointer(other.expr + '*' + self.expr, type_)

    def dot(self, other):
        other = as_node(other)
        if not isinstance(other, FixedPointer):
            raise NotImplementedError()
        type_ = infer_types(self, other)
        return FixedScalar(self.evaluate().dot(other.evaluate()))

    def rdot(self, other):
        return self.dot(other)

    # FIXME needs specific pointers
    # def translate(self, vocab, populate=None, keys=None, solver=None):
        # tr = self.type.vocab.transform_to(vocab, populate, solver)
        # return FixedPointer(np.dot(tr, self.evaluate().v), TVocabulary(vocab))

    def __repr__(self):
        return "FixedPointer({!r}, {!r})".format(self.expr, self.type)


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

        if isinstance(other, FixedNode):
            return self._mul_with_fixed(other)
        else:
            return self._mul_with_dynamic(other)

    def __rmul__(self, other):
        other = as_node(other)
        if not isinstance(other, Node):
            return NotImplemented

        if isinstance(other, FixedNode):
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

        if isinstance(other, FixedPointer):
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
