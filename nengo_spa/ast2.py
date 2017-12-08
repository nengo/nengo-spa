import weakref

import nengo
from nengo.utils.compat import is_number
import numpy as np

from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import TInferVocab, TScalar, TVocabulary


input_network_registry = weakref.WeakKeyDictionary()
input_vocab_registry = weakref.WeakKeyDictionary()
output_vocab_registry = weakref.WeakKeyDictionary()

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


def as_node(obj):
    if is_number(obj):
        obj = FixedScalar(obj)
    return obj


class Node(object):
    def __init__(self, type_):
        self.type = type_

    def infer_types(self, context_type):
        pass

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

    def infer_types(self, type_):
        if self.type == TInferVocab:
            self.type = type_

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
        if not isinstance(other, FixedNode):
            return NotImplemented
        type_ = coerce_types(self.type, other.type)
        return FixedPointer(self.expr + '+' + other.expr, type_)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = as_node(other)
        if not isinstance(other, FixedNode):
            return NotImplemented
        type_ = coerce_types(self.type, other.type)
        return FixedPointer(self.expr + '-' + other.expr, type_)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = as_node(other)
        if not isinstance(other, FixedNode):
            return NotImplemented
        type_ = coerce_types(self.type, other.type)
        return FixedPointer(self.expr + '*' + other.expr, type_)

    def __rmul__(self, other):
        other = as_node(other)
        if not isinstance(other, FixedNode):
            return NotImplemented
        type_ = coerce_types(self.type, other.type)
        return FixedPointer(other.expr + '*' + self.expr, type_)

    def __repr__(self):
        return "FixedPointer({!r}, {!r})".format(self.expr, self.type_)


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
        other.infer_types(self.type)
        type_ = coerce_types(self.type, other.type)
        return Summed((self, other), type_)

    def __radd__(self, other):
        other = as_node(other)
        return self + other

    def __sub__(self, other):
        other = as_node(other)
        return self + (-other)

    def __rsub__(self, other):
        other = as_node(other)
        return (-self) + other

    def __mul__(self, other):
        other = as_node(other)
        other.infer_types(self.type)

        if isinstance(other, FixedNode):
            # FIXME check AST type or instance type?
            if other.type == TScalar:
                tr = other.value
            else:
                tr = other.evaluate().get_convolution_matrix()
            return Transformed(self.construct(), tr, self.type)
        else:
            if self.type == TScalar and other.type == TScalar:
                mul = ProductRealization()
            elif self.type == other.type:
                mul = BindRealization(self.type.vocab)
            elif self.type == TScalar or other.type == TScalar:
                raise NotImplementedError(
                    "Dynamic scaling of semantic pointer not implemented.")
            else:
                raise SpaTypeError("Vocabulary mismmatch.")

            self.connect_to(mul.input_a)
            other.connect_to(mul.input_b)
            return ModuleOutput(mul.output, self.type)

    def __rmul__(self, other):
        other = as_node(other)
        other.infer_types(self.type)

        if isinstance(other, FixedNode):
            # FIXME check AST type or instance type?
            if other.type == TScalar:
                tr = other.value
            else:
                tr = other.evaluate().get_convolution_matrix()
            return Transformed(self.construct(), tr, self.type)
        else:
            if self.type == TScalar and other.type == TScalar:
                mul = ProductRealization()
            elif self.type == other.type:
                mul = BindRealization(self.type.vocab)
            elif self.type == TScalar or other.type == TScalar:
                print(self, other, self.type, other.type)
                raise NotImplementedError(
                    "Dynamic scaling of semantic pointer not implemented.")
            else:
                raise SpaTypeError("Vocabulary mismmatch.")

            other.connect_to(mul.input_a)
            self.connect_to(mul.input_b)
            return ModuleOutput(mul.output, self.type)


class Transformed(DynamicNode):
    def __init__(self, source, transform, type_):
        super(Transformed, self).__init__(type_=type_)
        self.source = source
        self.transform = transform

    def connect_to(self, sink):
        # FIXME connection params
        return nengo.Connection(self.source, sink, transform=self.transform)

    def construct(self):
        node = nengo.Node(size_in=self.type.vocab.dimensions)
        self.connect_to(node)
        return node


class Summed(DynamicNode):
    def __init__(self, sources, type_):
        super(Summed, self).__init__(type_=type_)
        self.sources = sources

    def infer_types(self, type_):
        for s in self.sources:
            s.infer_types(type_)

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
