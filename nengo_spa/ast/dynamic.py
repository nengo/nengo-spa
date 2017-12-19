import weakref

import nengo
from nengo.network import Network as NengoNetwork
from nengo.utils.compat import is_number
import numpy as np

from nengo_spa.ast.base import Fixed, infer_types, Node
from nengo_spa.ast.symbolic import FixedScalar, PointerSymbol, Symbol
from nengo_spa.exceptions import SpaTypeError
from nengo_spa.types import TAnyVocab, TScalar, TAnyVocabOfDim, TVocabulary


BasalGangliaRealization = None
ScalarRealization = None
StateRealization = None
ThalamusRealization = None
DotProductRealization = None
BindRealization = None
ProductRealization = None


input_network_registry = weakref.WeakKeyDictionary()
input_vocab_registry = weakref.WeakKeyDictionary()
output_vocab_registry = weakref.WeakKeyDictionary()


def as_node(obj):
    if is_number(obj):
        obj = FixedScalar(obj)
    return obj


class DynamicNode(Node):
    def __invert__(self):
        # FIXME alternate binding operators
        if not hasattr(self.type, 'dimensions'):
            raise SpaTypeError()  # FIXME better error
        dimensions = self.type.dimensions
        transform = np.eye(dimensions)[-np.arange(dimensions)]
        return Transformed(self.output, transform, self.type)

    def __neg__(self):
        return Transformed(self.construct(), transform=-1, type_=self.type)

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
            raise NotImplemented
        type_ = infer_types(self, other)

        if self.type == TScalar or other.type == TScalar:
            raise SpaTypeError()  # FIXME better error?

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

    def reinterpret(self, vocab=None):
        return Transformed(
            self.construct(), np.eye(self.type.dimensions),
            TAnyVocabOfDim(self.type.dimensions)
            if vocab is None else TVocabulary(vocab))

    def translate(self, vocab, populate=None, keys=None, solver=None):
        tr = self.type.vocab.transform_to(vocab, populate, keys, solver)
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
        dimensions = 1 if self.type == TScalar else self.type.dimensions
        node = nengo.Node(size_in=dimensions)
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
        if ActionSelection.active is None:
            infer_types(self, other)
            other.connect_to(self.input)
        else:
            return RoutedConnection(other, self)


class RoutedConnection(object):
    free_floating = set()

    def __init__(self, source, sink):
        self.type = infer_types(source, sink)
        self.source = source
        self.sink = sink
        RoutedConnection.free_floating.add(self)

    def connect_to(self, sink):
        return self.source.connect_to(sink)

    def construct(self):
        return self.connect_to(self.sink.input)

    @property
    def fixed(self):
        return isinstance(self.source, Fixed)

    def transform(self):
        assert self.fixed
        if self.type == TScalar:
            return self.source.evaluate()
        else:
            return np.atleast_2d(self.source.evaluate().v).T


class ActionSelection(object):
    active = None

    def __init__(self):
        self.built = False
        self.bg = None
        self.thalamus = None
        self.utilities = []
        self.actions = []

    def __enter__(self):
        assert not self.built
        if ActionSelection.active is None:
            ActionSelection.active = self
        else:
            raise RuntimeError()  # FIXME better error
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ActionSelection.active = None

        if exc_type is not None:
            return

        if len(RoutedConnection.free_floating) > 0:
            raise RuntimeError()  # FIXME better error

        if len(self.utilities) <= 0:
            return

        self.bg = BasalGangliaRealization(len(self.utilities))
        self.thalamus = ThalamusRealization(len(self.utilities))
        self.thalamus.connect_bg(self.bg)

        for index, utility in enumerate(self.utilities):
            self.bg.connect_input(utility.output, index=index)

        for index, action in enumerate(self.actions):
            for effect in action:
                if effect.fixed:
                    self.thalamus.connect_fixed(
                        index, effect.sink.input, transform=effect.transform())
                else:
                    self.thalamus.construct_gate(
                        index, net=NengoNetwork.context[-1])
                    channel = self.thalamus.construct_channel(
                        effect.sink.input, effect.type)
                    effect.connect_to(channel.input)
                    self.thalamus.connect_gate(index, channel)

        self.built = True

    def add_action(self, condition, *actions):
        assert ActionSelection.active is self
        utility = ScalarRealization()  # FIXME should be node
        self.utilities.append(utility)
        self.actions.append(actions)
        RoutedConnection.free_floating.difference_update(actions)
        return utility


def ifmax(condition, *actions):
    utility = ActionSelection.active.add_action(condition, *actions)
    condition.connect_to(utility.input)
    return utility
