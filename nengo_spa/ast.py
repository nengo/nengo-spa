"""Definition of abstract AST construction tools."""

import weakref

import nengo
from nengo.base import NengoObject
from nengo.connection import Connection
from nengo.network import Network as NengoNetwork
import numpy as np

from nengo_spa import pointer
from nengo_spa import types
from nengo_spa.exceptions import (
    SpaConstructionError, SpaParseError, SpaTypeError)
from nengo_spa.pointer import SemanticPointer

from nengo_spa.vocab import Vocabulary


input_network_registry = weakref.WeakKeyDictionary()
input_vocab_registry = weakref.WeakKeyDictionary()
output_vocab_registry = weakref.WeakKeyDictionary()


def route(a, b):
    eff = Effect(b, a)
    from nengo_spa.actions import Actions
    if Actions.context is not None and Actions.context() is not None:
        Actions.context().add_effect(eff)

    return eff


def ifmax(cond, *effects):
    effects = Effects(effects)
    for e in effects.effects:
        e.channeled = True
    act = Action(cond, effects)

    from nengo_spa.actions import Actions
    if Actions.context is not None and Actions.context() is not None:
        Actions.context().add_rule(act)

    return act


def as_node(obj, as_sink=False):
    if isinstance(obj, Node):
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
        return (Sink(name, obj) if as_sink else
                Module(name, obj))


def instant_network_construction(fn):
    def op(self, other):
        node = fn(self, other)
        if node.staticity == Node.Staticity.DYNAMIC:
            node.infer_types(None)
            return node.construct(ConstructionContext(
                NengoNetwork.context[-1]))
        else:
            return node
    return op


class SpaOperatorMixin(object):
    def __invert__(self):
        return ApproxInverse(self)

    def __neg__(self):
        return Negative(self)

    @instant_network_construction
    def __add__(self, other):
        return Sum(self, other)

    @instant_network_construction
    def __radd__(self, other):
        return Sum(other, self)

    @instant_network_construction
    def __sub__(self, other):
        return Sum(self, Negative(other))

    @instant_network_construction
    def __rsub__(self, other):
        return Sum(other, Negative(self))

    @instant_network_construction
    def __mul__(self, other):
        return Product(self, other)

    @instant_network_construction
    def __rmul__(self, other):
        return Product(other, self)

    def __rshift__(self, other):
        return route(self, other)

    def __rrshift__(self, other):
        return route(other, self)


class Node(SpaOperatorMixin):
    """Abstract class for a node in the AST.

    Attributes
    ----------
    staticity : int
        Staticity of the node. See `.Node.Staticity` which also lists valid
        values.
    type : :class:`Type`
        Type that this node evaluates to. This will be set to ``None`` until
        the type inference was run.
    """

    class Staticity:
        """Valid staticity values.

        * ``FIXED``: Value of the node is static, i.e. does not change over
          time.
        * ``TRANSFORM_ONLY``: Value of the node changes over time, but can be
          implemented with a transform on existing neural resources.
        * ``DYNAMIC``: Value of the node is fully dynamic and needs additional
          neural resources to be implemented.
        """
        FIXED = 0
        TRANSFORM_ONLY = 1
        DYNAMIC = 2

    def __init__(self, staticity, precedence=0):
        self.staticity = staticity
        self.precedence = precedence
        self.type = None
        self.constructed = []

    @property
    def fixed(self):
        """Indicates whether the node value is static.

        A static node value does not change over time.
        """
        return self.staticity <= self.Staticity.FIXED

    def __hash__(self):
        h = hash(self.__class__)
        for v in self.__dict__.values():
            h ^= hash(v)
        return h

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.__dict__ == other.__dict__)

    def infer_types(self, context_type):
        """Run type inference on this node and its children.

        Will raise a :class:`nengo.exceptions.SpaTypeError` if invalid,
        non-matching, or undefined types are discovered.

        This function is idempotent.

        Parameters
        ----------
        context_type : :class:`Type`
            The type of the context of this node. Allows to infer the type
            from the context if the node has no definitive type on its own.
        """
        raise NotImplementedError()

    def construct(self, context):
        """Construct network components for this node and its children.

        Parameters
        ----------
        context : :class:`ConstructionContext`
            The context in which the network components are constructed.

        Returns
        -------
        list of :class:`Articfact`
            The constructed objects with transforms that should be connected to
            the objects that continue the processing.
        """
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate the value of this node statically.

        This can only be done for ``fixed`` nodes, otherwise a ``ValueError``
        will be raised.
        """
        raise NotImplementedError()


class Sink(Node):
    """SPA network that acts as sink identified by its name."""

    def __init__(self, name, obj):
        super(Sink, self).__init__(staticity=Node.Staticity.DYNAMIC)
        self.name = name
        self._obj = obj

    @property
    def obj(self):
        if not isinstance(self._obj, NengoObject):
            return getattr(self._obj, 'input')
        else:
            return self._obj

    def infer_types(self, context_type):
        try:
            vocab = input_vocab_registry[self.obj]
        except KeyError:
            raise SpaTypeError("{} {} is not declared as input.".format(
                self.name, self.obj))
        if vocab is None:
            self.type = types.TScalar
        else:
            self.type = types.TVocabulary(vocab)

    def construct(self, context):
        return []

    def evaluate(self):
        raise ValueError("Sinks cannot be statically evaluated.")

    def __str__(self):
        return self.name


class Source(Node):
    """Abstract base class of all AST ast that can provide some output value.
    """

    def __init__(self, *args, **kwargs):
        super(Source, self).__init__(*args, **kwargs)

    def infer_types(self, context_type):
        raise NotImplementedError()

    def construct(self, context):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()


class Scalar(Source):
    """A fixed scalar."""

    def __init__(self, value):
        super(Scalar, self).__init__(staticity=Node.Staticity.FIXED)
        self.value = value
        self.type = types.TScalar

    def infer_types(self, context_type):
        pass

    def construct(self, context):
        return construct_bias(self, self.value, context)

    def evaluate(self):
        return self.value

    def __str__(self):
        return str(self.value)


class Symbol(Source):
    """A fixed semantic pointer identified by its name (`key`).

    The `key` has to start with a capatial letter.
    """

    def __init__(self, key):
        super(Symbol, self).__init__(staticity=Node.Staticity.FIXED)
        self.validate(key)
        self.key = key

    def validate(self, key):
        if not key[0].isupper():
            raise SpaParseError(
                "Semantic pointers must begin with a capital letter.")

    def infer_types(self, context_type):
        if not isinstance(context_type, types.TVocabulary):
            raise SpaTypeError("Invalid type.")
        self.type = context_type
        # Make sure that key has been added to vocab after type inference to
        # make translate operations as deterministic as possible (it could
        # differ if at a later point another key would be added).
        self.type.vocab.parse(self.key)

    def construct(self, context):
        value = self.type.vocab[self.key].v
        return construct_bias(self, value, context)

    def evaluate(self):
        return self.type.vocab[self.key]

    def __str__(self):
        return self.key


class Zero(Source):
    """Zero which can act as scalar or zero vector."""

    def __init__(self):
        super(Zero, self).__init__(staticity=Node.Staticity.FIXED)

    def infer_types(self, context_type):
        if context_type is None:
            self.type = types.TScalar
        elif context_type == types.TScalar or isinstance(
                context_type, types.TVocabulary):
            self.type = context_type
        else:
            raise SpaTypeError("Invalid type.")

    def construct(self, context):
        return []

    def evaluate(self):
        if self.type == types.TScalar:
            return 0
        else:
            return pointer.Zero(self.type.vocab.dimensions)

    def __str__(self):
        return "0"


class One(Source):
    """One which can act as scalar or identity vector."""

    def __init__(self):
        super(One, self).__init__(staticity=Node.Staticity.FIXED)

    def infer_types(self, context_type):
        if context_type is None:
            self.type = types.TScalar
        elif context_type == types.TScalar or isinstance(
                context_type, types.TVocabulary):
            self.type = context_type
        else:
            raise SpaTypeError("Invalid type.")

    def construct(self, context):
        return construct_bias(self, self.evaluate(), context)

    def evaluate(self):
        if self.type == types.TScalar:
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
            staticity=Node.Staticity.TRANSFORM_ONLY)
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
            vocab = output_vocab_registry[self.obj]
        except KeyError:
            raise SpaTypeError("{} {} is not declared as output.".format(
                self.name, self.obj))
        if vocab is None:
            self.type = types.TScalar
        else:
            self.type = types.TVocabulary(vocab)

    def construct(self, context):
        return [Artifact(self.obj)]

    def evaluate(self):
        raise ValueError("Module cannot be statically evaluated.")

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
    DotProductRealization = None

    def __init__(self, lhs, rhs):
        lhs = as_node(lhs)
        rhs = as_node(rhs)

        if not lhs.fixed and not rhs.fixed:
            staticity = Node.Staticity.DYNAMIC
        else:
            staticity = max(lhs.staticity, rhs.staticity)

        super(DotProduct, self).__init__(lhs, rhs, staticity)
        self.type = types.TScalar

    def infer_types(self, context_type):
        context_type = infer_vocab(self.lhs, self.rhs)
        self.lhs.infer_types(context_type)
        self.rhs.infer_types(context_type)
        if not isinstance(self.lhs.type, types.TVocabulary):
            raise SpaTypeError(
                "First argument of dot product '{}' is not of type "
                "TVocabulary, but {}.".format(self, self.lhs.type))
        if not isinstance(self.rhs.type, types.TVocabulary):
            raise SpaTypeError(
                "Second argument of dot product '{}' is not of type "
                "TVocabulary, but {}.".format(self, self.rhs.type))
        if self.lhs.type.vocab is not self.rhs.type.vocab:
            raise SpaTypeError(
                "Incompatible types {} and {} in dot product '{}'.".format(
                    self.lhs.type, self.rhs.type, self))

    def construct(self, context):
        if self.fixed:
            return construct_bias(self, self.evaluate(), context)

        if self.lhs.fixed:
            tr = value_to_transform(self.lhs.evaluate()).T
            return [x.add_transform(tr)
                    for x in self.rhs.construct(context)]
        if self.rhs.fixed:
            tr = value_to_transform(self.rhs.evaluate()).T
            return [x.add_transform(tr)
                    for x in self.lhs.construct(context)]

        assert self.lhs.type.vocab is self.rhs.type.vocab
        with context.active_net:
            net = self.DotProductRealization(
                self.lhs.type.vocab, label=str(self))
            self._connect_binary_operation(context, net)
        self.constructed.append(net)
        return net

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
            context_type = infer_vocab(self.lhs, self.rhs)

        self.lhs.infer_types(context_type)
        self.rhs.infer_types(context_type)

        if self.lhs.type == self.rhs.type:
            self.type = self.lhs.type
        elif self.allow_scalar and self.lhs.type == types.TScalar:
            self.type = self.rhs.type
        elif self.allow_scalar and self.rhs.type == types.TScalar:
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
    BindRealization = None
    ProductRealization = None

    def __init__(self, lhs, rhs):
        lhs = as_node(lhs)
        rhs = as_node(rhs)

        if not lhs.fixed and not rhs.fixed:
            staticity = Node.Staticity.DYNAMIC
        else:
            staticity = max(lhs.staticity, rhs.staticity)

        super(Product, self).__init__(
            lhs, rhs, '*', staticity, allow_scalar=True)

    def construct(self, context):
        if self.fixed:
            return construct_bias(self, self.evaluate(), context)

        if self.lhs.fixed:
            tr = self.lhs.evaluate()
            artifacts = self.rhs.construct(context)
        if self.rhs.fixed:
            tr = self.rhs.evaluate()
            artifacts = self.lhs.construct(context)

        is_binding = (isinstance(self.lhs.type, types.TVocabulary) and
                      isinstance(self.rhs.type, types.TVocabulary))

        if self.lhs.fixed or self.rhs.fixed:
            if is_binding:
                tr = tr.get_convolution_matrix()
            else:
                tr = value_to_transform(tr)
            return [x.add_transform(tr) for x in artifacts]

        with context.active_net:
            if is_binding:
                net = self.BindRealization(self.type.vocab, label=str(self))
            elif (self.lhs.type == types.TScalar and
                    self.rhs.type == types.TScalar):
                net = self.ProductRealization()
            else:
                raise NotImplementedError(
                    "Dynamic scaling of semantic pointer not implemented.")
        self.constructed.append(net)

        self._connect_binary_operation(context, net)
        return net

    def evaluate(self):
        return self.lhs.evaluate() * self.rhs.evaluate()


class Sum(BinaryOperation):
    def __init__(self, lhs, rhs):
        lhs = as_node(lhs)
        rhs = as_node(rhs)
        staticity = min(
            Node.Staticity.TRANSFORM_ONLY,
            max(lhs.staticity, rhs.staticity))
        super(Sum, self).__init__(lhs, rhs, '+', staticity, precedence=3)

    def construct(self, context):
        if self.fixed:
            return construct_bias(self, self.evaluate(), context)

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
            return construct_bias(self, self.evaluate(), context)
        return [x.add_transform(-1) for x in self.source.construct(context)]

    def evaluate(self):
        return -self.source.evaluate()


class ApproxInverse(UnaryOperation):
    def __init__(self, source):
        super(ApproxInverse, self).__init__(source, '~')

    def infer_types(self, context_type):
        super(ApproxInverse, self).infer_types(context_type)
        if not isinstance(self.type, types.TVocabulary):
            raise SpaTypeError(
                "Cannot apply approximate inverse to '{}' which is not of "
                "type TVocabulary, but {}.".format(self.source, self.type))

    def construct(self, context):
        if self.fixed:
            return construct_bias(self, self.evaluate(), context)

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
            self.type = types.TVocabulary(self.vocab)
        if not isinstance(self.type, types.TVocabulary):
            raise SpaTypeError(
                "Cannot infer vocabulary for '{}'.".format(self))

        self.source.infer_types(None)
        if not isinstance(self.source.type, types.TVocabulary):
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
            self.type = types.TVocabulary(self.vocab)
        if not isinstance(self.type, types.TVocabulary):
            raise SpaTypeError(
                "Cannot infer vocabulary for '{}'.".format(self))

        self.source.infer_types(None)
        if not isinstance(self.source.type, types.TVocabulary):
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


class Effect(Node):
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
        self.type = types.TEffect
        self.sink = as_node(sink, as_sink=True)
        self.source = as_node(source)
        self.channeled = channeled
        self.channel = None

        if not isinstance(self.sink, Sink):
            raise SpaTypeError("%s is not a valid sink for SPA routing" % sink)
        if not isinstance(self.source, Source):
            raise SpaTypeError("%s is not a valid source for SPA routing" %
                               source)

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


class Effects(Node):
    """Multiple effects."""

    def __init__(self, effects, name=None):
        super(Effects, self).__init__(
            staticity=max(e.staticity for e in effects))
        self.type = types.TEffects
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


class Action(Node):
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
        super(Action, self).__init__(staticity=Node.Staticity.DYNAMIC)
        self.type = types.TAction
        self.index = index
        self.condition = as_node(condition)
        self.effects = as_node(effects)
        self.name = name

    @property
    def effect(self):
        warnings.warn(DeprecationWarning("Use the effects attribute instead."))
        return self.effects

    def infer_types(self, context_type):
        if isinstance(self.condition, Node):
            self.condition.infer_types(context_type)
            if self.condition.type != types.TScalar:
                raise SpaTypeError(
                    "Condition '{}' does not evaluate to a scalar.".format(
                        self.condition))

        self.effects.infer_types(None)

    def construct(self, context):
        if context.bg is None or context.thalamus is None:
            raise SpaConstructionError(
                "Conditional actions require basal ganglia and thalamus.")

        # construct bg utility
        if self.condition.staticity <= Node.Staticity.TRANSFORM_ONLY:
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
                tr = value_to_transform(effect.source.evaluate())
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


class ActionSet(Node):
    """A set of actions implemented by one basal ganglia and thalamus.

    If multiple *ActionSets* exist, each creates their own basal ganglia and
    thalamus.
    """

    BasalGangliaRealization = None
    ThalamusRealization = None

    def __init__(self, actions):
        super(ActionSet, self).__init__(staticity=Node.Staticity.DYNAMIC)
        self.type = types.TActionSet
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
            self.bg = self.BasalGangliaRealization(action_count=action_count)
            self.thalamus = self.ThalamusRealization(action_count=action_count)
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


class ConstructionContext(object):
    """Context in which SPA actions are constructed.

    This primarily provides the SPA networks used to construct certain
    components. All attributes except `root_network` may be ``None`` if these
    are not provided in the current construction context.

    Attributes
    ----------
    root_network : :class:`spa.Module`
        The root network the encapsulated all of the constructed structures.
    bg : :class:`spa.BasalGanglia`
        Module to manage the basal ganglia part of action selection.
    thalamus : :class:`spa.Thalamus`
        Module to manage the thalamus part of action selection.
    sink : :class:`Sink`
        Node in the AST where some result will be send to.
    active_net : class:`nengo.Network`
        Network to add constructed components to.
    """
    __slots__ = [
        'root_network', 'bg', 'thalamus', 'bias', 'sink', 'active_net']

    def __init__(
            self, root_network, bg=None, thalamus=None,
            sink=None, active_net=None):
        self.root_network = root_network
        self.bg = bg
        self.thalamus = thalamus
        self.bias = None
        self.sink = sink
        if active_net is None:
            active_net = root_network
        self.active_net = active_net

    def subcontext(self, bg=None, thalamus=None, sink=None, active_net=None):
        """Creates a subcontext.

        All omitted arguments will be initialized from the parent context.
        """
        if bg is None:
            bg = self.bg
        if thalamus is None:
            thalamus = self.thalamus
        if sink is None:
            sink = self.sink
        if active_net is None:
            active_net = self.active_net
        return self.__class__(
            root_network=self.root_network, bg=bg,
            thalamus=thalamus, sink=sink, active_net=active_net)

    @property
    def sink_network(self):
        return input_network_registry[self.sink.obj]

    @property
    def sink_input(self):
        return self.sink.obj, input_vocab_registry[self.sink.obj]


class Artifact(object):
    """Stores information about Nengo objects constructed from SPA actions.

    This deals with the problem that when we construct the Nengo object we
    have the object itself and know the transform for the outgoing connection,
    but we do not know what to connect to yet. Thus, this class allows to store
    and pass around that information until we know what to connect to.

    Attributes
    ----------
    nengo_source : :class:`nengo.NengoObject`
        Some constructed Nengo object that allows outgoing connections.
    transform : array-like
        Transform to be applied to the outgoing connection from the
        `nengo_source`.
    """

    def __init__(self, nengo_source, transform=1):
        self.nengo_source = nengo_source
        self.transform = transform

    def add_transform(self, tr):
        return Artifact(self.nengo_source, np.dot(tr, self.transform))


def infer_vocab(*nodes):
    """Return the first vocabulary type that can be inferred for one of the
    `nodes`.

    If the context that an operation is embedded in does not provide a
    vocabulary type, it might be possible to infer it from one of the nodes
    in the operation.

    If no vocabulary type can be inferred, ``None`` will be returned.

    Note that this function calls ``infer_types`` on a subset or all `nodes`
    which has side effects!
    """
    for node in nodes:
        try:
            node.infer_types(None)
            if isinstance(node.type, types.TVocabulary):
                return node.type
        except SpaTypeError:
            pass
    return None


def construct_bias(ast_node, value, context):
    """Constructs a bias node (if not existent) and a transform to `value`."""
    with context.active_net:
        if context.bias is None:
            context.bias = nengo.Node([1], label="bias")
    ast_node.constructed.append(context.bias)
    if isinstance(value, SemanticPointer):
        value = value.v
    transform = np.array([value]).T
    return [Artifact(context.bias, transform=transform)]


def value_to_transform(value):
    if isinstance(value, SemanticPointer):
        value = np.atleast_2d(value.v).T
    return np.asarray(value)
