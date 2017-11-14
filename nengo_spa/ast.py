"""Definition of abstract AST construction tools."""

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import nengo
from nengo.base import NengoObject
from nengo.utils.compat import is_integer
import numpy as np

from nengo_spa.exceptions import SpaTypeError
from nengo_spa.network import Network
from nengo_spa.pointer import SemanticPointer


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
        return Network.get_input_network(self.sink.obj)

    @property
    def sink_input(self):
        return self.sink.obj, Network.get_input_vocab(self.sink.obj)


class AstAccessor(Sequence):
    """Provides access to the root AST nodes of build action rules.

    Nodes can either be accessed by their ordinal position in the action rules
    or by the name provided in the action rules.
    """

    def __init__(self, ast):
        self.ast = ast
        self.by_name = {
            node.name: node for node in ast if hasattr(node, 'name')}

    def __len__(self):
        return len(self.ast)

    def __getitem__(self, key):
        if is_integer(key):
            return self.ast[key]
        else:
            return self.by_name[key]

    def __contains__(self, key):
        return key in self.ast or key in self.by_name

    def keys(self):
        return self.by_name.keys()

    def _attr_list(self, attr):
        objs = []
        for act_set in self.ast:
            if hasattr(act_set, attr):
                objs.append(getattr(act_set, attr))
        return objs

    def all_bgs(self):
        return self._attr_list("bg")

    def all_thals(self):
        return self._attr_list("thalamus")


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
            if isinstance(node.type, TVocabulary):
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


class Node(object):
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
            vocab = Network.get_input_vocab(self.obj)
        except KeyError:
            raise SpaTypeError("{} {} is not declared as input.".format(
                self.name, self.obj))
        if vocab is None:
            self.type = TScalar
        else:
            self.type = TVocabulary(vocab)

    def construct(self, context):
        return []

    def evaluate(self):
        raise ValueError("Sinks cannot be statically evaluated.")

    def __str__(self):
        return self.name

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if isinstance(attr, Network):
            return Sink(self.name + '.' + name, attr)
        else:
            return attr


class Type(object):
    """Describes a type.

    Each part of the AST evaluates to some type.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.name)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.name == other.name

    def __ne__(self, other):
        return not self == other


TAction = Type('TAction')
TActionSet = Type('TActionSet')
TScalar = Type('TScalar')
TEffect = Type('TEffect')
TEffects = Type('TEffects')


class TVocabulary(Type):
    """Each vocabulary is treated as its own type.

    All vocabulary types constitute a type class.
    """

    def __init__(self, vocab):
        super(TVocabulary, self).__init__('TVocabulary')
        self.vocab = vocab

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            self.__class__.__name__, self.name, self.vocab)

    def __str__(self):
        return '{}<{}>'.format(self.name, self.vocab)

    def __hash__(self):
        return super(TVocabulary, self).__hash__() ^ hash(self.vocab)

    def __eq__(self, other):
        return (super(TVocabulary, self).__eq__(other) and
                self.vocab is other.vocab)
