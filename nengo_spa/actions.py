import inspect
import warnings
import weakref

from nengo.base import NengoObject
from nengo.connection import Connection
from nengo.network import Network as NengoNetwork
from nengo.exceptions import NetworkContextError
import numpy as np

from nengo_spa import pointer
from nengo_spa.compiler import ast_nodes as nodes
from nengo_spa.compiler.ast_nodes import AstAccessor, ConstructionContext
from nengo_spa.compiler.ast_types import (
    TAction, TActionSet, TScalar, TEffect, TEffects, TVocabulary)
from nengo_spa.exceptions import (
    SpaConstructionError, SpaParseError, SpaTypeError)
from nengo_spa.modules.basalganglia import BasalGanglia
from nengo_spa.modules.bind import Bind
from nengo_spa.modules.compare import Compare
from nengo_spa.modules.product import Product as ProductModule
from nengo_spa.modules.thalamus import Thalamus
from nengo_spa.network import Network as Network
from nengo_spa.vocab import Vocabulary

action_ops = ["__invert__", "__neg__", "__add__", "__radd__", "__sub__",
              "__rsub__", "__mul__", "__rmul__", "__rshift__", "__rrshift__"]
saved_network_ops = {k: getattr(Network, k, None) for k in action_ops}


def route(a, b):
    eff = Effect(b, a)
    Actions.context.add_effect(eff)
    return eff


def ifmax(cond, *effects):
    Actions.context.add_rule(cond, effects)


def as_node(obj, as_sink=False):
    if isinstance(obj, nodes.Node):
        return obj
    elif obj is None or obj == "0":
        return Zero()
    elif obj == "1":
        return One()
    elif isinstance(obj, str):
        return Symbol(obj)
    elif isinstance(obj, (int, float)):
        return Scalar(obj)
    else:
        name = str(obj)
        return (nodes.Sink(name, obj) if as_sink else
                Module(name, obj))


class Actions(AstAccessor):
    context = None

    def __init__(self):
        self.blocks = []
        self.effects = []
        self.rules = []

        super(Actions, self).__init__(self.blocks)

    def add_block(self, actions, mode="bg"):
        if len(Network.context) <= 0:
            raise NetworkContextError(
                "actions can only be called inside a ``with network:`` block.")
        root_network = Network.context[-1]

        if mode == "bg":
            actions = self._add_bg_block(actions)

        for block in actions:
            block.infer_types(None)
        construction_context = ConstructionContext(root_network)
        for block in actions:
            block.construct(construction_context)
        self.blocks += actions

    def _add_bg_block(self, actions):
        action_nodes = []
        for i, (cond, effects) in enumerate(actions):
            if not isinstance(effects, (tuple, list)):
                effects = [effects]
            effects = Effects(effects)
            for e in effects.effects:
                e.channeled = True
            action_nodes.append(
                Action(cond, effects, index=i))
        return [ActionSet(action_nodes)]

    def add_effect(self, effect):
        self.effects.append(effect)

    def add_rule(self, cond, effects):
        self.effects = [e for e in self.effects if e not in effects]
        self.rules.append((cond, effects))

    def __enter__(self):
        if Actions.context is not None:
            try:
                # TODO: is there a better way to check if it's alive?
                Actions.context.test
                raise SpaConstructionError("Nesting of Action contexts is not "
                                           "supported")
            except ReferenceError:
                # dead context object
                pass
        Actions.context = weakref.proxy(self)

        for k in action_ops:
            setattr(Network, k, getattr(ActionOps, k))
        return self

    def __exit__(self, *args):
        if args[0] is not None:
            return

        if len(self.rules) > 0:
            self.add_block(self.rules, mode="bg")
            self.rules = []

        if len(self.effects) > 0:
            self.add_block(self.effects, mode="cortical")
            self.effects = []

        Actions.context = None

        for k, v in saved_network_ops.items():
            if v is None:
                delattr(Network, k)
            else:
                setattr(Network, k, v)


class ActionOps(object):
    def __invert__(self):
        return ApproxInverse(self)

    def __neg__(self):
        return Negative(self)

    def __add__(self, other):
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __sub__(self, other):
        return Sum(self, Negative(other))

    def __rsub__(self, other):
        return Sum(other, Negative(self))

    def __mul__(self, other):
        return Product(self, other)

    def __rmul__(self, other):
        return Product(other, self)

    def __rshift__(self, other):
        return route(self, other)

    def __rrshift__(self, other):
        return route(other, self)


class Source(nodes.Node, ActionOps):
    """Abstract base class of all AST nodes that can provide some output value.
    """

    def infer_types(self, context_type):
        raise NotImplementedError()

    def construct(self, context):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()


class Scalar(Source):
    """A fixed scalar."""

    def __init__(self, value):
        super(Scalar, self).__init__(staticity=nodes.Node.Staticity.FIXED)
        self.value = value
        self.type = TScalar

    def infer_types(self, context_type):
        pass

    def construct(self, context):
        return nodes.construct_bias(self, self.value, context)

    def evaluate(self):
        return self.value

    def __str__(self):
        return str(self.value)


class Symbol(Source):
    """A fixed semantic pointer identified by its name (`key`).

    The `key` has to start with a capatial letter.
    """

    def __init__(self, key):
        super(Symbol, self).__init__(staticity=nodes.Node.Staticity.FIXED)
        self.validate(key)
        self.key = key

    def validate(self, key):
        if not key[0].isupper():
            raise SpaParseError(
                "Semantic pointers must begin with a capital letter.")

    def infer_types(self, context_type):
        if not isinstance(context_type, TVocabulary):
            raise SpaTypeError("Invalid type.")
        self.type = context_type
        # Make sure that key has been added to vocab after type inference to
        # make translate operations as deterministic as possible (it could
        # differ if at a later point another key would be added).
        self.type.vocab.parse(self.key)

    def construct(self, context):
        value = self.type.vocab[self.key].v
        return nodes.construct_bias(self, value, context)

    def evaluate(self):
        return self.type.vocab[self.key]

    def __str__(self):
        return self.key


class Zero(Source):
    """Zero which can act as scalar or zero vector."""

    def __init__(self):
        super(Zero, self).__init__(staticity=nodes.Node.Staticity.FIXED)

    def infer_types(self, context_type):
        if context_type is None:
            self.type = TScalar
        elif context_type == TScalar or isinstance(context_type, TVocabulary):
            self.type = context_type
        else:
            raise SpaTypeError("Invalid type.")

    def construct(self, context):
        return []

    def evaluate(self):
        if self.type == TScalar:
            return 0
        else:
            return pointer.Zero(self.type.vocab.dimensions)

    def __str__(self):
        return "0"


class One(Source):
    """One which can act as scalar or identity vector."""

    def __init__(self):
        super(One, self).__init__(staticity=nodes.Node.Staticity.FIXED)

    def infer_types(self, context_type):
        if context_type is None:
            self.type = TScalar
        elif context_type == TScalar or isinstance(context_type, TVocabulary):
            self.type = context_type
        else:
            raise SpaTypeError("Invalid type.")

    def construct(self, context):
        return nodes.construct_bias(self, self.evaluate(), context)

    def evaluate(self):
        if self.type == TScalar:
            return 1
        else:
            return pointer.Identity(self.type.vocab.dimensions)

    def __str__(self):
        return "1"


class Module(Source):
    """A SPA network or network output identified by its name.

    This will provide potentially time varying input. This class is not used
    for networks that act as sink.
    """

    def __init__(self, name, obj):
        super(Module, self).__init__(
            staticity=nodes.Node.Staticity.TRANSFORM_ONLY)
        self.name = name
        self._obj = obj

    @property
    def obj(self):
        if not isinstance(self._obj, NengoObject):
            return getattr(self._obj, 'output')
        else:
            return self._obj

    def infer_types(self, context_type):
        try:
            vocab = Network.get_output_vocab(self.obj)
        except KeyError:
            raise SpaTypeError("{} {} is not declared as output.".format(
                self.name, self.obj))
        if vocab is None:
            self.type = TScalar
        else:
            self.type = TVocabulary(vocab)

    def construct(self, context):
        return [nodes.Artifact(self.obj)]

    def evaluate(self):
        raise ValueError("Module cannot be statically evaluated.")

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if isinstance(attr, Network):
            return Module(self.name + '.' + name, attr)
        else:
            return attr

    def __str__(self):
        return self.name


class BinaryNode(Source):
    """Base class for binary operators.

    Attributes
    ----------
    lhs : :class:`Node`
        Left-hand side
    rhs : :class:`Node`
        Right-hand side
    """

    def __init__(self, lhs, rhs, staticity, precedence=0):
        lhs = as_node(lhs)
        rhs = as_node(rhs)

        super(BinaryNode, self).__init__(
            staticity=staticity, precedence=precedence)
        self.lhs = lhs
        self.rhs = rhs

    def infer_types(self, context_type):
        raise NotImplementedError()

    def construct(self, context):
        raise NotImplementedError()

    def _connect_binary_operation(self, context, net):
        with context.root_network:
            for artifact in self.lhs.construct(context):
                self.constructed.append(
                    Connection(
                        artifact.nengo_source, net.input_a,
                        transform=artifact.transform))
            for artifact in self.rhs.construct(context):
                self.constructed.append(
                    Connection(
                        artifact.nengo_source, net.input_b,
                        transform=artifact.transform))


class DotProduct(BinaryNode):
    def __init__(self, lhs, rhs):
        lhs = as_node(lhs)
        rhs = as_node(rhs)

        if not lhs.fixed and not rhs.fixed:
            staticity = nodes.Node.Staticity.DYNAMIC
        else:
            staticity = max(lhs.staticity, rhs.staticity)

        super(DotProduct, self).__init__(lhs, rhs, staticity)
        self.type = TScalar

    def infer_types(self, context_type):
        context_type = nodes.infer_vocab(self.lhs, self.rhs)
        self.lhs.infer_types(context_type)
        self.rhs.infer_types(context_type)
        if not isinstance(self.lhs.type, TVocabulary):
            raise SpaTypeError(
                "First argument of dot product '{}' is not of type "
                "TVocabulary, but {}.".format(self, self.lhs.type))
        if not isinstance(self.rhs.type, TVocabulary):
            raise SpaTypeError(
                "Second argument of dot product '{}' is not of type "
                "TVocabulary, but {}.".format(self, self.rhs.type))
        if self.lhs.type.vocab is not self.rhs.type.vocab:
            raise SpaTypeError(
                "Incompatible types {} and {} in dot product '{}'.".format(
                    self.lhs.type, self.rhs.type, self))

    def construct(self, context):
        if self.fixed:
            return nodes.construct_bias(self, self.evaluate(), context)

        if self.lhs.fixed:
            tr = nodes.value_to_transform(self.lhs.evaluate()).T
            return [x.add_transform(tr)
                    for x in self.rhs.construct(context)]
        if self.rhs.fixed:
            tr = nodes.value_to_transform(self.rhs.evaluate()).T
            return [x.add_transform(tr)
                    for x in self.lhs.construct(context)]

        assert self.lhs.type.vocab is self.rhs.type.vocab
        with context.active_net:
            net = Compare(self.lhs.type.vocab, label=str(self))
            self._connect_binary_operation(context, net)
        self.constructed.append(net)
        return [nodes.Artifact(net.output)]

    def evaluate(self):
        return np.dot(self.lhs.evaluate(), self.rhs.evaluate())

    def __str__(self):
        return 'dot({}, {})'.format(self.lhs, self.rhs)


class BinaryOperation(BinaryNode):
    """Base class for binary operators.

    Attributes
    ----------
    lhs : :class:`Node`
        Left-hand side
    rhs : :class:`Node`
        Right-hand side
    operator : str
        String representation of the operator.
    """

    def __init__(
            self, lhs, rhs, operator, staticity, precedence=2,
            allow_scalar=False):
        super(BinaryOperation, self).__init__(
            lhs, rhs, staticity, precedence=precedence)
        self.operator = operator
        self.allow_scalar = allow_scalar

    def infer_types(self, context_type):
        if context_type is None:
            context_type = nodes.infer_vocab(self.lhs, self.rhs)

        self.lhs.infer_types(context_type)
        self.rhs.infer_types(context_type)

        if self.lhs.type == self.rhs.type:
            self.type = self.lhs.type
        elif self.allow_scalar and self.lhs.type == TScalar:
            self.type = self.rhs.type
        elif self.allow_scalar and self.rhs.type == TScalar:
            self.type = self.lhs.type
        else:
            raise SpaTypeError(
                "Incompatible types {} and {} in operation '{}'.".format(
                    self.lhs.type, self.rhs.type, self))

    def construct(self, context):
        raise NotImplementedError()

    def __str__(self):
        if self.lhs.precedence > self.precedence:
            lhs_str = '({})'.format(self.lhs)
        else:
            lhs_str = str(self.lhs)
        if self.rhs.precedence > self.precedence:
            rhs_str = '({})'.format(self.rhs)
        else:
            rhs_str = str(self.rhs)

        return '{} {} {}'.format(lhs_str, self.operator, rhs_str)


class Product(BinaryOperation):
    def __init__(self, lhs, rhs):
        lhs = as_node(lhs)
        rhs = as_node(rhs)

        if not lhs.fixed and not rhs.fixed:
            staticity = nodes.Node.Staticity.DYNAMIC
        else:
            staticity = max(lhs.staticity, rhs.staticity)

        super(Product, self).__init__(
            lhs, rhs, '*', staticity, allow_scalar=True)

    def construct(self, context):
        if self.fixed:
            return nodes.construct_bias(self, self.evaluate(), context)

        if self.lhs.fixed:
            tr = self.lhs.evaluate()
            artifacts = self.rhs.construct(context)
        if self.rhs.fixed:
            tr = self.rhs.evaluate()
            artifacts = self.lhs.construct(context)

        is_binding = (isinstance(self.lhs.type, TVocabulary) and
                      isinstance(self.rhs.type, TVocabulary))

        if self.lhs.fixed or self.rhs.fixed:
            if is_binding:
                tr = tr.get_convolution_matrix()
            else:
                tr = nodes.value_to_transform(tr)
            return [x.add_transform(tr) for x in artifacts]

        with context.active_net:
            if is_binding:
                net = Bind(self.type.vocab, label=str(self))
            elif self.lhs.type == TScalar and self.rhs.type == TScalar:
                net = ProductModule()
            else:
                raise NotImplementedError(
                    "Dynamic scaling of semantic pointer not implemented.")
        self.constructed.append(net)

        self._connect_binary_operation(context, net)
        return [nodes.Artifact(net.output)]

    def evaluate(self):
        return self.lhs.evaluate() * self.rhs.evaluate()


class Sum(BinaryOperation):
    def __init__(self, lhs, rhs):
        lhs = as_node(lhs)
        rhs = as_node(rhs)
        staticity = min(
            nodes.Node.Staticity.TRANSFORM_ONLY,
            max(lhs.staticity, rhs.staticity))
        super(Sum, self).__init__(lhs, rhs, '+', staticity, precedence=3)

    def construct(self, context):
        if self.fixed:
            return nodes.construct_bias(self, self.evaluate(), context)

        return (self.lhs.construct(context) +
                self.rhs.construct(context))

    def evaluate(self):
        return self.lhs.evaluate() + self.rhs.evaluate()


class UnaryOperation(Source):
    """Base class for unary operators.

    Attributes
    ----------
    source : :class:`Node`
        Node the operator is applied to.
    operator : str
        String representation of the operator.
    """

    def __init__(self, source, operator, precedence=1):
        source = as_node(source)
        super(UnaryOperation, self).__init__(
            staticity=source.staticity, precedence=precedence)
        self.source = source
        self.operator = operator

    def infer_types(self, context_type):
        self.source.infer_types(context_type)
        self.type = self.source.type

    def construct(self, context):
        raise NotImplementedError()

    def __str__(self):
        if self.source.precedence <= self.precedence:
            return self.operator + str(self.source)
        else:
            return self.operator + '(' + str(self.source) + ')'


class Negative(UnaryOperation):
    def __init__(self, source):
        super(Negative, self).__init__(source, '-')

    def construct(self, context):
        if self.fixed:
            return nodes.construct_bias(self, self.evaluate(), context)
        return [x.add_transform(-1) for x in self.source.construct(context)]

    def evaluate(self):
        return -self.source.evaluate()


class ApproxInverse(UnaryOperation):
    def __init__(self, source):
        super(ApproxInverse, self).__init__(source, '~')

    def infer_types(self, context_type):
        super(ApproxInverse, self).infer_types(context_type)
        if not isinstance(self.type, TVocabulary):
            raise SpaTypeError(
                "Cannot apply approximate inverse to '{}' which is not of "
                "type TVocabulary, but {}.".format(self.source, self.type))

    def construct(self, context):
        if self.fixed:
            return nodes.construct_bias(self, self.evaluate(), context)

        d = self.type.vocab.dimensions
        tr = np.eye(d)[-np.arange(d)]
        return [x.add_transform(tr) for x in self.source.construct(context)]

    def evaluate(self):
        return ~self.source.evaluate()


class Reinterpret(Source):
    def __init__(self, source, vocab=None):
        source = as_node(source)
        if vocab is not None and not isinstance(vocab, Vocabulary):
            vocab = as_node(vocab)
        super(Reinterpret, self).__init__(staticity=source.staticity)
        self.source = source
        self.vocab = vocab

    def infer_types(self, context_type):
        if self.vocab is None:
            self.type = context_type
        elif isinstance(self.vocab, Module):
            self.vocab.infer_types(None)
            self.type = self.vocab.type
        else:
            self.type = TVocabulary(self.vocab)
        if not isinstance(self.type, TVocabulary):
            raise SpaTypeError(
                "Cannot infer vocabulary for '{}'.".format(self))

        self.source.infer_types(None)
        if not isinstance(self.source.type, TVocabulary):
            raise SpaTypeError(
                "Cannot reinterpret '{}' because it is not of type "
                "TVocabulary, but {}.".format(self.source, self.source.type))
        if self.source.type.vocab.dimensions != self.type.vocab.dimensions:
            raise SpaTypeError(
                "Cannot reinterpret '{}' with {}-dimensional vocabulary as "
                "{}-dimensional vocabulary.".format(
                    self.source, self.source.type.vocab.dimensions,
                    self.type.vocab.dimensions))

    def construct(self, context):
        return self.source.construct(context)

    def evaluate(self):
        return self.source.evaluate()

    def __str__(self):
        return 'reinterpret({})'.format(self.source)


class Translate(Source):
    def __init__(self, source, vocab=None, populate=None, solver=None):
        source = as_node(source)
        if vocab is not None and not isinstance(vocab, Vocabulary):
            vocab = as_node(vocab)
        super(Translate, self).__init__(staticity=source.staticity)
        self.source = source
        self.vocab = vocab
        self.populate = populate
        self.solver = solver

    def infer_types(self, context_type):
        if self.vocab is None:
            self.type = context_type
        elif isinstance(self.vocab, Module):
            self.vocab.infer_types(None)
            self.type = self.vocab.type
        else:
            self.type = TVocabulary(self.vocab)
        if not isinstance(self.type, TVocabulary):
            raise SpaTypeError(
                "Cannot infer vocabulary for '{}'.".format(self))

        self.source.infer_types(None)
        if not isinstance(self.source.type, TVocabulary):
            raise SpaTypeError(
                "Cannot translate '{}' because it is not of type "
                "TVocabulary, but {}.".format(self.source, self.source.type))

    def construct(self, context):
        tr = self.source.type.vocab.transform_to(
            self.type.vocab, populate=self.populate, solver=self.solver)
        artifacts = self.source.construct(context)
        return [a.add_transform(tr) for a in artifacts]

    def evaluate(self):
        tr = self.source.type.vocab.transform_to(
            self.type.vocab, populate=self.populate)
        return pointer.SemanticPointer(np.dot(tr, self.source.evaluate().v))

    def __str__(self):
        return 'translate({})'.format(self.source)


class Effect(nodes.Node):
    """Assignment of an expression to a SPA network.

    Attributes
    ----------
    sink : :class:`Sink`
        Module that acts as sink.
    source : :class:`Source`
        Source of information to be fed to the sink.
    channeled : bool
        Indicates if information should be passed through an additional
        (inhibitable) channel between the source and sink.
    channel : :class:`nengo.networks.EnsembleArray`
        The channel that was constructed for this effect. Will initially be
        ``None`` and will only be constructed if `channeled` is ``True``.
    """

    def __init__(self, sink, source, channeled=False):
        source = as_node(source)
        super(Effect, self).__init__(staticity=source.staticity)
        self.type = TEffect
        self.sink = as_node(sink, as_sink=True)
        self.source = as_node(source)
        self.channeled = channeled
        self.channel = None

    def infer_types(self, context_type):
        if context_type is not None:
            raise ValueError("Effect only allows a context_type of None.")
        try:
            self.sink.infer_types(None)
            self.source.infer_types(self.sink.type)
        except SpaTypeError as err:
            raise SpaTypeError('In effect "{}": {}'.format(self, err))
        if self.sink.type != self.source.type:
            raise SpaTypeError("Cannot assign {} to {} in '{}'".format(
                self.source.type, self.sink.type, self))

    def construct(self, context):
        assert context.sink is None
        context = context.subcontext(sink=self.sink)

        if self.channeled and self.fixed:
            return []  # Will be implemented in transform from thalamus

        if self.channeled:
            self.channel = context.thalamus.construct_channel(
                context.sink_network, context.sink_input,
                net=context.active_net, label='channel: ' + str(self))
            target = self.channel.input
            connect_fn = context.thalamus.connect
        else:
            target = context.sink_input[0]
            connect_fn = Connection

        artifacts = self.source.construct(context)
        for artifact in artifacts:
            self.constructed.append(connect_fn(
                artifact.nengo_source, target, transform=artifact.transform))
        return []

    def evaluate(self):
        raise ValueError("Effects cannot be statically evaluated.")

    def __str__(self):
        return '{source} -> {sink}'.format(source=self.source, sink=self.sink)


class Effects(nodes.Node):
    """Multiple effects."""

    def __init__(self, effects, name=None):
        super(Effects, self).__init__(
            staticity=max(e.staticity for e in effects)
            if len(effects) > 0 else self.Staticity.FIXED)
        self.type = TEffects
        self.effects = effects
        self.name = name

    def infer_types(self, context_type):
        for e in self.effects:
            e.infer_types(context_type)

    def construct(self, context):
        for effect in self.effects:
            effect.construct(context)
        return []

    def evaluate(self):
        raise ValueError("Effects cannot be statically evaluated.")

    def __str__(self):
        if self.name is None:
            return "\n".join(str(e) for e in self.effects)
        else:
            return (
                "always as {!r}:\n    ".format(self.name) +
                "\n     ".join(str(e) for e in self.effects))


class Action(nodes.Node):
    """A conditional SPA action.

    Attributes
    ----------
    index : int
        Numerical index of the action.
    condition : :class:`Node`
        Condition for the action's effects to become active.
    effets : :class:`Node`
        Effects when the condition is met.
    name : str
        Name of the action.
    """

    def __init__(self, condition, effects, index=0, name=None):
        super(Action, self).__init__(staticity=nodes.Node.Staticity.DYNAMIC)
        self.type = TAction
        self.index = index
        self.condition = as_node(condition)
        self.effects = as_node(effects)
        self.name = name

    @property
    def effect(self):
        warnings.warn(DeprecationWarning("Use the effects attribute instead."))
        return self.effects

    def infer_types(self, context_type):
        if isinstance(self.condition, nodes.Node):
            self.condition.infer_types(context_type)
            if self.condition.type != TScalar:
                raise SpaTypeError(
                    "Condition '{}' does not evaluate to a scalar.".format(
                        self.condition))

        self.effects.infer_types(None)

    def construct(self, context):
        if context.bg is None or context.thalamus is None:
            raise SpaConstructionError(
                "Conditional actions require basal ganglia and thalamus.")

        # construct bg utility
        if self.condition.staticity <= nodes.Node.Staticity.TRANSFORM_ONLY:
            condition_context = context
        else:
            condition_context = context.subcontext(active_net=NengoNetwork(
                label='condition: ' + str(self.condition)))
        condition_artifacts = self.condition.construct(condition_context)

        # construct effects
        self.effects.construct(context)

        # construct thalamus gate
        if not self.effects.fixed:
            gate_label = 'gate[{}]: {}'.format(self.index, self.condition)
            context.thalamus.construct_gate(
                self.index, net=context.thalamus, label=gate_label)

        for artifact in condition_artifacts:
            context.bg.connect_input(
                artifact.nengo_source, artifact.transform, self.index)

        # connect up
        for effect in self.effects.effects:
            if effect.fixed:
                sink = effect.sink.obj
                tr = nodes.value_to_transform(effect.source.evaluate())
                context.thalamus.connect_fixed(self.index, sink, transform=tr)
            else:
                context.thalamus.connect_gate(self.index, effect.channel)

        return []

    def evaluate(self):
        raise NotImplementedError("Cannot evaluate conditional actions.")

    def __str__(self):
        if self.name is not None:
            name_str = " as {!r}".format(self.name)
        else:
            name_str = ""
        if len(self.effects.effects) > 0:
            effect_str = "".join(
                "\n    " + line for line in str(self.effects).split("\n"))
        else:
            effect_str = "\n    pass"
        return 'ifmax {utility}{name}:{effects}\n'.format(
            utility=self.condition, name=name_str, effects=effect_str)


class ActionSet(nodes.Node):
    """A set of actions implemented by one basal ganglia and thalamus.

    If multiple *ActionSets* exist, each creates their own basal ganglia and
    thalamus.
    """

    def __init__(self, actions):
        super(ActionSet, self).__init__(staticity=nodes.Node.Staticity.DYNAMIC)
        self.type = TActionSet
        self.actions = actions
        self.bg = None
        self.thalamus = None

    def infer_types(self, context_type):
        for action in self.actions:
            action.infer_types(context_type)

    def construct(self, context):
        action_count = len(self.actions)
        if action_count <= 0:
            return

        with context.root_network:
            self.bg = BasalGanglia(action_count=action_count)
            self.thalamus = Thalamus(action_count=action_count)
            for i, a in enumerate(self.actions):
                self.thalamus.actions.ensembles[i].label = (
                    'action[{}]: {}'.format(i, a.effects))
            self.thalamus.connect_bg(self.bg)
            self.constructed.append(self.bg)
            self.constructed.append(self.thalamus)

        for action in self.actions:
            action.construct(context.subcontext(
                bg=self.bg, thalamus=self.thalamus))

        return []

    def evaluate(self):
        raise NotImplementedError("Cannot evaluate action sets.")

    def __str__(self):
        action_strings = [str(action) for action in self.actions]
        return "".join(
            action_strings[:1] + ["el" + x for x in action_strings[1:]])
