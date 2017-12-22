"""AST classes for symbolic operations."""

import nengo
from nengo.config import Default
import numpy as np

from nengo_spa.ast.base import infer_types, Fixed
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.pointer import SemanticPointer
from nengo_spa.types import TAnyVocab, TScalar, TVocabulary
from nengo.utils.compat import is_number


def as_symbolic_node(obj):
    if is_number(obj):
        obj = FixedScalar(obj)
    return obj


class Symbol(Fixed):
    @property
    def expr(self):
        raise NotImplementedError()


class FixedScalar(Symbol):
    # Does not implement any operators as Python does so for numbers and
    # once a Python number gets converted into an AST node it must have been
    # used with some other non-scalar AST node that will return
    # a non-FixedScalar AST node. Thus, there should never be a situation where
    # an operator is applied to two FixedScalar nodes.

    def __init__(self, value):
        super(FixedScalar, self).__init__(type_=TScalar)
        self.value = value

    def connect_to(self, sink, transform=Default):
        return nengo.Connection(self.construct(), sink, transform=Default)

    def construct(self):
        return nengo.Node(self.value, label=str(self.value))

    def evaluate(self):
        return self.value

    @property
    def expr(self):
        return repr(self.value)


class PointerSymbol(Symbol):
    def __init__(self, expr, type_=TAnyVocab):
        super(PointerSymbol, self).__init__(type_=type_)
        self._expr = expr

    def connect_to(self, sink, transform=Default):
        return nengo.Connection(self.construct(), sink, transform=Default)

    def construct(self):
        return nengo.Node(self.evaluate().v, label=self.expr)

    def evaluate(self):
        if not isinstance(self.type, TVocabulary):
            raise SpaTypeError(
                "Cannot evaluate a symbolic semantic pointer expression "
                "without knowing the vocabulary.")
        return self.type.vocab.parse(self.expr)

    @property
    def expr(self):
        return self._expr

    def __invert__(self):
        return PointerSymbol('~' + self.expr, self.type)

    def __neg__(self):
        return PointerSymbol('-' + self.expr, self.type)

    def __add__(self, other):
        other = as_symbolic_node(other)
        if not isinstance(other, PointerSymbol):
            return NotImplemented
        type_ = infer_types(self, other)
        return PointerSymbol(self.expr + '+' + other.expr, type_)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = as_symbolic_node(other)
        if not isinstance(other, PointerSymbol):
            return NotImplemented
        type_ = infer_types(self, other)
        return PointerSymbol(self.expr + '-' + other.expr, type_)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = as_symbolic_node(other)
        if not isinstance(other, Symbol):
            return NotImplemented
        type_ = infer_types(self, other)
        return PointerSymbol(self.expr + '*' + other.expr, type_)

    def __rmul__(self, other):
        other = as_symbolic_node(other)
        if not isinstance(other, Symbol):
            return NotImplemented
        type_ = infer_types(self, other)
        return PointerSymbol(other.expr + '*' + self.expr, type_)

    def dot(self, other):
        other = as_symbolic_node(other)
        if not isinstance(other, PointerSymbol):
            return NotImplemented
        type_ = infer_types(self, other)
        return FixedScalar(self.evaluate().dot(other.evaluate()))

    def rdot(self, other):
        return self.dot(other)

    def translate(self, vocab, populate=None, keys=None, solver=None):
        tr = self.type.vocab.transform_to(vocab, populate, solver)
        return SemanticPointer(np.dot(tr, self.evaluate().v), vocab=vocab)

    def __repr__(self):
        return "PointerSymbol({!r}, {!r})".format(self.expr, self.type)


class PointerSymbolFactory(object):
    """Provides syntactic sugar to create *PointerSymbol* instances.

    Use the `.sym` instance of this class to create *PointerSymbols* like so::

        sym.foo  # creates PointerSymbol('foo')
    """

    def __getattribute__(self, key):
        return PointerSymbol(key)


sym = PointerSymbolFactory()
